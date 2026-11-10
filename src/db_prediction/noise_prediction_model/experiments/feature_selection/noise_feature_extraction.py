"""Training input features and future-noise target extraction.

Physical LowState and measured speed selection, lag definitions, target
alignment, and gap-safe windows are shared by train.py, test.py and prediction.
Acoustic RMS computation itself remains in preprocessing/common_time.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"preprocessing"))
from contact_features import DEFAULT_CONFIG, feature_names as contact_feature_names, feature_dependency, add_contact_features
import numpy as np
import pandas as pd

DEFAULT_LAGS_MS = (0, 20, 60, 100)
DEFAULT_SAMPLE_MS = 20
DEFAULT_HORIZON_MS = 20
TARGET_COLUMN = "audio_dbfs_20ms"

FAMILIES = {"contact_force": "ls_ff_", "acceleration": "ls_acc_", "angular_velocity": "ls_gyro_",
            "orientation": "ls_quaternion_", "joint_position": "ls_q_", "joint_velocity": "ls_dq_",
            "estimated_torque": "ls_tau_", "actual_speed": "sp_vel_"}

FAMILIES.update({prefix.rstrip("_"): prefix for prefix in ("impact_rate6_", "impact_rate_max20_", "impact_integral20_", "contact_on_", "contact_known_", "contact_age_", "contact_landing_valid_", "contact_pre_vz_body_", "contact_pre_valid_")})

def validate_preprocessing_metadata(metadata):
    """Require the causal 20ms RMS contract of preprocessing/preprocessing.py."""
    if metadata.get('robot_interpolation') != 'previous':
        raise ValueError('Rebuild input with causal previous-value resampling')
    window=metadata.get('audio_rms_window_s')
    if window is None or not np.isclose(float(window), .020, rtol=0, atol=1e-9):
        raise ValueError('Expected 20ms RMS CSV from preprocessing/preprocessing.py')


def raw_features(df):
    # Quaternion represents attitude; retain RPY in CSV but avoid duplication.
    # Tick/time/IDs can identify the experiment and are never model inputs.
    low = [c for c in df if c.startswith("ls_") and not c.startswith(("ls_tick", "ls_rpy_"))]
    speed = [c for c in ("sp_vel_x", "sp_vel_y", "sp_vel_z") if c in df]
    for field in ("q", "dq", "tau"):
        if len([c for c in low if c.startswith(f"ls_{field}_")]) != 12:
            raise ValueError(f"Need all 12 ls_{field} columns: rebuild preprocessing/preprocessing.py")
    return low, speed


def build_windows(df, columns, lags_ms=DEFAULT_LAGS_MS, sample_ms=DEFAULT_SAMPLE_MS, horizon_ms=DEFAULT_HORIZON_MS, with_targets=True, include_standing=False, contact_config=None):
    """Past-only robot windows; targets/history cannot cross a missing segment."""
    derived = [c for c in columns if c in contact_feature_names()]
    if derived:
        df = add_contact_features(df, contact_config, requested=derived)
        contact_config = DEFAULT_CONFIG if contact_config is None else contact_config
    xs, ys, metas, coverage = [], [], [], []
    names = [f"{c}__lag{lag:g}ms" for lag in lags_ms for c in columns]
    for trial, g in df.groupby("trial", sort=False):
        g = g.sort_values("time_abs").reset_index(drop=True)
        command_fields = [c for c in ('command_vx', 'command_vy', 'command_vyaw') if c in g]
        if (not include_standing and command_fields
                and np.all(np.abs(g[command_fields].to_numpy(float)) < 1e-9)):
            coverage.append(dict(trial=trial, n_source=len(g), n_windows=0, reason="standing_command"))
            continue
        t = g.time_abs.to_numpy(float)
        if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f"{trial}: timestamps must be finite and increasing")
        dt = float(np.median(np.diff(t)))
        steps = np.rint(np.asarray(lags_ms)/1000/dt).astype(int)
        horizon = int(round(horizon_ms/1000/dt))
        stride = max(1, int(round(sample_ms/1000/dt)))
        if not np.allclose(steps*dt*1000, lags_ms, atol=.1) or abs(horizon*dt*1000-horizon_ms) > .1:
            raise ValueError(f"{trial}: lags/horizon do not align to {dt}s grid")
        breaks = np.r_[False, np.diff(t) > dt*1.5]
        if "segment" in g:
            breaks |= np.r_[False, np.diff(g.segment.to_numpy()) != 0]
        segments = np.cumsum(breaks)
        data = g[columns].to_numpy(float)
        count = 0
        for seg in np.unique(segments):
            pos = np.flatnonzero(segments == seg)
            anchors = np.arange(pos[0]+max(steps), pos[-1]+1-(horizon if with_targets else 0), stride)
            if not len(anchors): continue
            x = np.concatenate([data[anchors-step] for step in steps], axis=1)
            y = g[TARGET_COLUMN].to_numpy(float)[anchors+horizon] if with_targets else np.full(len(anchors), np.nan)
            good = np.isfinite(x).all(axis=1)
            if contact_config is not None:
                good &= (t[anchors-max(steps)]-t[pos[0]])*1000 >= contact_config["extra_history_ms"]-1e-6
            if with_targets: good &= np.isfinite(y)
            anchors, x, y = anchors[good], x[good], y[good]
            if not len(anchors): continue
            xs.append(x.astype(np.float32)); ys.append(y); count += len(y)
            metas.append(pd.DataFrame(dict(trial=trial, segment=int(seg), anchor_time=t[anchors],
                                          target_time=t[anchors]+horizon_ms/1000,
                                          time_from_move_start=g.time_from_move_start.to_numpy(float)[anchors]+horizon_ms/1000,
                                          command_vx=g.command_vx.to_numpy(float)[anchors] if "command_vx" in g else np.nan,
                                          command_vy=g.command_vy.to_numpy(float)[anchors] if "command_vy" in g else np.nan,
                                          command_vyaw=g.command_vyaw.to_numpy(float)[anchors] if "command_vyaw" in g else np.nan,
                                          body_vx=g.sp_vel_x.to_numpy(float)[anchors] if "sp_vel_x" in g else np.nan)))
        coverage.append(dict(trial=trial, n_source=len(g), n_windows=count, reason="included" if count else "insufficient_valid_history"))
    if not xs: raise ValueError("No valid whole-sequence windows")
    return np.concatenate(xs), np.concatenate(ys), pd.concat(metas, ignore_index=True), names, pd.DataFrame(coverage)


# Executable feature-definition stage. Statistical ranking is descriptive;
# automatic ranking uses training trials only and must be repeated inside validation folds.
def model_feature_ranking(x, y, groups, names, trees=96, seed=42):
    """Fit only supplied training trials; aggregate all lags of each state."""
    from sklearn.ensemble import ExtraTreesRegressor
    counts=pd.Series(groups).value_counts()
    sample_weights=np.asarray([1/counts[g] for g in groups])
    sample_weights/=sample_weights.mean()
    model = ExtraTreesRegressor(n_estimators=trees, max_depth=12, min_samples_leaf=20,
        max_features=.8, random_state=seed, n_jobs=1)
    model.fit(x, y, sample_weight=sample_weights)
    table = pd.DataFrame({'feature':[n.split('__lag')[0] for n in names],
                          'model_importance':model.feature_importances_})
    return table.groupby('feature',as_index=False).model_importance.sum().sort_values('model_importance',ascending=False)


def select_input_features(candidates, requested=None):
    selected = list(candidates if requested is None else requested)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select a nonempty, unique LowState feature list")
    invalid = set(selected)-set(candidates)
    if invalid:
        raise ValueError(f"Invalid LowState features: {sorted(invalid)}")
    return selected


def feature_correlations(x, y, meta, names):
    """Training-trial Spearman associations to the actual future noise target."""
    rows = []
    for trial, positions in meta.groupby('trial', sort=False).groups.items():
        positions = np.asarray(list(positions))
        values = pd.DataFrame(x[positions], columns=names)
        target = pd.Series(y[positions]).rank()
        variable = values.columns[values.nunique(dropna=True)>1]
        correlations = (values[variable].rank().corrwith(target).reindex(values.columns)
                        if target.nunique()>1 else pd.Series(np.nan,index=values.columns))
        for name, rho in correlations.items():
            feature, lag = name.rsplit('__lag', 1)
            rows.append(dict(trial=trial, predictor=name, feature=feature,
                             lag_ms=float(lag.removesuffix('ms')), spearman_rho=rho,
                             n_windows=len(positions)))
    details = pd.DataFrame(rows)
    ranked = details.assign(abs_rho=details.spearman_rho.abs()).groupby('feature', as_index=False).agg(
        mean_rho=('spearman_rho', 'mean'), mean_abs_rho=('abs_rho', 'mean'),
        n_trial_lag_pairs=('spearman_rho', 'count'))
    return details, ranked.sort_values('mean_abs_rho', ascending=False)



def select_ranked_features(ranked, candidates, top_k):
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    ranking = ranked[ranked.feature.isin(candidates)].dropna(subset=['mean_abs_rho'])
    top = set()
    for feature in ranking.sort_values(['mean_abs_rho', 'feature'], ascending=[False, True]).feature:
        dependency = feature_dependency(feature)
        required = {feature} | ({dependency} if dependency in candidates else set())
        if len(top | required) <= top_k:
            top |= required
        if len(top) == top_k:
            break
    selected = [feature for feature in candidates if feature in top]
    if not selected:
        raise ValueError('No finite training-only feature/noise correlations')
    return selected


def select_from_windows(x, y, meta, names, candidates, top_k):
    _, ranked = feature_correlations(x, y, meta.reset_index(drop=True), names)
    return select_ranked_features(ranked, candidates, top_k)


def load_feature_definition(path):
    import json
    config = json.loads(path.read_text())
    low, speed = config['selected_lowstate_features'], config['speed_features']
    if not low or not speed or len(low+speed) != len(set(low+speed)):
        raise ValueError('Invalid selected feature schema')
    if set(config['training_trials']) & set(config['test_trials']):
        raise ValueError('Feature analysis contains reserved test trials')
    return config


def main():
    import argparse
    import json
    from pathlib import Path
    from noise_prediction_model.experiments.feature_selection.model_evaluation import ROOT, FEATURE_DIR, plt
    parser = argparse.ArgumentParser(description='Define training inputs and analyze training-only feature/noise associations')
    parser.add_argument('--input', type=Path, default=ROOT/'preprocessed/1007/train/all_trials.csv')
    parser.add_argument('--output', type=Path, default=FEATURE_DIR)
    parser.add_argument('--test-trials', nargs='+')
    parser.add_argument('--features', nargs='+', help='Optional explicit LowState columns, overriding automatic top-K selection')
    parser.add_argument('--top-k', type=int, default=20, help='Top raw LowState features by training-trial mean absolute Spearman rho')
    parser.add_argument('--sample-ms', type=float, default=DEFAULT_SAMPLE_MS)
    parser.add_argument('--lags-ms', type=float, nargs='+', default=list(DEFAULT_LAGS_MS))
    parser.add_argument('--horizon-ms', type=float, default=DEFAULT_HORIZON_MS)
    parser.add_argument('--include-standing', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--feature-mode', choices=['raw', 'contact'], default='contact')
    parser.add_argument('--match-contact-history', action='store_true', help='Use identical warmup for a raw-feature comparison')
    args = parser.parse_args()
    if args.top_k <= 0 or args.sample_ms <= 0 or args.horizon_ms <= 10 or not args.lags_ms or min(args.lags_ms) < 0 or 0 not in args.lags_ms:
        parser.error('sample > 0, horizon > 10ms, nonnegative lags including zero required')
    args.lags_ms = sorted(set(args.lags_ms))
    metadata = json.loads((args.input.parent/'metadata.json').read_text())
    validate_preprocessing_metadata(metadata)
    args.test_trials=args.test_trials or metadata.get('test_trials',['1.8_6','1.8_7'])
    df = pd.read_csv(args.input)
    df = df[~df.trial.isin(args.test_trials)].copy()
    raw_candidates, speed = raw_features(df)
    derived_candidates = contact_feature_names() if args.feature_mode == "contact" else []
    candidates = raw_candidates + derived_candidates
    contact_config = dict(DEFAULT_CONFIG) if derived_candidates or args.match_contact_history else None
    if len(speed) != 3:
        raise ValueError('Measured body velocity x/y/z required for the baseline')
    x, y, meta, names, coverage = build_windows(df, candidates+speed, args.lags_ms, args.sample_ms,
                                               args.horizon_ms, include_standing=args.include_standing, contact_config=contact_config)
    details, ranked = feature_correlations(x, y, meta, names)
    selected = (select_input_features(candidates, args.features) if args.features else
                select_ranked_features(ranked, candidates, args.top_k))
    args.output.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output/'feature_noise_correlations.csv', index=False)
    ranked['selected_input'] = ranked.feature.isin(selected+speed)
    ranked.to_csv(args.output/'feature_correlation_summary.csv', index=False)
    coverage.to_csv(args.output/'analysis_coverage.csv', index=False)
    # A zero-filled cached speed also encodes contact phase. Report valid-only
    # associations separately; never restrict the training rows to these events.
    valid_rows = []
    for name in names:
        base, lag = name.rsplit('__lag', 1)
        if not base.startswith('contact_pre_vz_body_'): continue
        mask_name = feature_dependency(base)+'__lag'+lag
        j, mask_j = names.index(name), names.index(mask_name)
        for trial, positions in meta.groupby('trial', sort=False).groups.items():
            positions = np.asarray(list(positions))
            positions = positions[x[positions, mask_j] == 1]
            rho = (pd.Series(x[positions, j]).corr(pd.Series(y[positions]), method='spearman')
                   if len(positions)>2 and np.unique(x[positions,j]).size>1 and np.unique(y[positions]).size>1 else np.nan)
            valid_rows.append(dict(trial=trial, predictor=name, feature=base, n_valid=len(positions), spearman_rho_valid_only=rho))
    pd.DataFrame(valid_rows, columns=['trial','predictor','feature','n_valid','spearman_rho_valid_only']).to_csv(args.output/'precontact_velocity_valid_correlations.csv', index=False)
    selected_names = [f'{c}__lag{lag:g}ms' for lag in args.lags_ms for c in selected+speed]
    stat = args.input.stat()
    config = dict(schema_version=1, feature_mode=args.feature_mode, contact_feature_config=contact_config,
                  raw_lowstate_candidates=raw_candidates, derived_contact_candidates=derived_candidates, input=str(args.input.resolve()),
                  input_size=stat.st_size, input_mtime_ns=stat.st_mtime_ns,
                  selected_lowstate_features=selected, speed_features=speed, feature_names=selected_names,
                  candidate_lowstate_features=candidates, lags_ms=args.lags_ms, sample_ms=args.sample_ms,
                  horizon_ms=args.horizon_ms, include_standing=args.include_standing,
                  training_trials=sorted(meta.trial.unique()), test_trials=sorted(set(args.test_trials)),
                  selection_method='explicit_columns' if args.features else 'training_trial_mean_abs_spearman_top_k',
                  top_k=args.top_k, selection_uses_noise_ranking=not bool(args.features), preprocessing_metadata=metadata)
    pd.DataFrame([dict(feature=c, family=next((k for k,v in FAMILIES.items() if c.startswith(v)), 'other'),
                       role=('contact_derived' if c in derived_candidates else 'lowstate') if c in selected else 'speed_baseline',
                       n_lags=len(args.lags_ms)) for c in selected+speed]).to_csv(args.output/'selected_features.csv', index=False)
    if not args.no_plots:
        top = ranked[ranked.feature.isin(candidates)].head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top.feature, top.mean_abs_rho)
        ax.set(xlabel='Mean |Spearman rho| across training trials and lags', title='Training-only future-noise associations')
        fig.tight_layout(); fig.savefig(args.output/'feature_noise_correlations.png', dpi=130); plt.close(fig)
    lines = ['# 학습 input feature 정의 및 분석', '',
             f"학습 분석 {len(config['training_trials'])}개 시퀀스 / {len(y):,}개 창. 제외 테스트: {', '.join(config['test_trials'])}.",
             f'선정 상태·파생 feature {len(selected)}개 + 실제 운동 상태 {len(speed)}개, lag 포함 input {len(selected_names)}차원.', '',
             f'후보 feature는 시퀀스별·lag별 절대 Spearman 상관을 평균한 순위의 상위 {args.top_k}개를 선정한다. 운동 상태는 별도 비교 기준으로 유지한다.',
             '`--features`로 명시한 목록은 그대로 train.py의 입력이 된다.',
             f"목표: 20ms RMS noise in dBFS. 과거/현재 상태와 {args.horizon_ms:g}ms 이후 목표의 시퀀스 내 Spearman 관계다.",
             '상관 순위는 입력 선정 지표이며 모델의 조건부 예측 중요도나 인과 영향력은 아니다. p-value를 사용하지 않는다.',
             'train.py는 외부 및 내부 검증의 학습 구간에서 feature를 다시 선정해 검증 정답 누출을 막는다.',
             '상관을 보고 수동으로 고른 feature의 교차검증 점수에는 선택 편향이 있을 수 있다. 최종 판단은 선정에 사용하지 않은 테스트에서 한다.',
             '같은 기록의 재분할은 새 날짜/바닥/마이크 조건으로의 독립 검증이 아니다.', '',
             '|Feature|평균 절대 상관|선정|', '|---|---:|---|']
    if contact_config:
        lines[2:2] = [
            f'전체 보행 후보: LowState {len(raw_candidates)}개 + 접촉 파생 {len(derived_candidates)}개. 접촉 이벤트로 행을 제한하지 않는다.',
            '힘 증가율: 과거 6ms의 양의 기울기, 그 기울기의 과거 20ms 최댓값. 힘 적분: 과거 20ms 사다리꼴 적분(단위 미검증이므로 충격량 대리지표).',
            '접촉 추정: on 10 / off 5, 6ms 확인. 확인 시점부터만 접촉 상태·접촉 경과시간·접촉 전 발 z속도를 사용한다. 초기 접촉과 미확인 값은 유효성 마스크로 구분한다.',
            '발 속도는 몸체 좌표계 상대속도이며 지면 상대 착지 속도가 아니다. 접촉 경과시간과 속도의 유효성 마스크도 top-K 예산 안에서 함께 선정한다.',
            '접촉 전 속도의 0 채움과 보행 위상 효과를 구분하기 위해 precontact_velocity_valid_correlations.csv에 유효 구간만의 보조 상관을 저장한다.',
            '파생 feature의 26ms 준비 구간을 최대 lag 이전에 추가 확보한다. 원본 모델과 비교할 때도 동일한 준비 구간을 적용한다.',
            '접촉 임계값은 오디오·test 결과로 조정하지 않았다. 시간 동기화 불확실성과 센서 잡음은 짧은 충격의 상관을 약화할 수 있다.', '']
    lines.extend(f'|{r.feature}|{r.mean_abs_rho:.4f}|{r.selected_input}|' for r in ranked.head(20).itertuples())
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    # Publish the manifest last, after its associated analysis has been saved.
    (args.output/'selected_features.json').write_text(json.dumps(config, indent=2))
    print(f'Saved feature definition ({len(selected_names)} inputs) and training-only analysis -> {args.output}')


if __name__ == '__main__':
    main()
