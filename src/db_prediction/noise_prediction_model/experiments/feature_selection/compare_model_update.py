#!/usr/bin/env python3
"""Compare frozen old/new predictors on identical updated held-out windows."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from noise_prediction_model.experiments.feature_selection.model_evaluation import ROOT, metrics
from noise_prediction_model.experiments.feature_selection.noise_feature_extraction import build_windows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motion_type(record):
    vectors = np.asarray(record['recorded_command_vectors'], dtype=float)
    if not vectors.size:
        vectors = np.array([[float(record['meta'].get(k, 0.)) for k in ('vx', 'vy', 'vyaw')]])
    active = np.any(np.abs(vectors) > 1e-9, axis=0)
    return {(True, False, False): 'forward_translation',
            (False, True, False): 'lateral_translation',
            (False, False, True): 'in_place_rotation',
            (False, False, False): 'standing'}.get(tuple(active), 'mixed_commands')


def artifact_predictions(artifact, frame):
    cfg = artifact['config']
    x, y, meta, names, coverage = build_windows(
        frame, artifact['columns'], cfg['lags_ms'], cfg['sample_ms'],
        cfg['horizon_ms'], include_standing=cfg['include_standing'],
        contact_config=cfg.get('contact_feature_config'))
    if names != artifact['feature_names']:
        raise ValueError('Feature schema differs from fitted artifact')
    selected = x[:, artifact['indices']]
    prediction = artifact['model'].predict(selected)
    rows = meta[['trial', 'anchor_time', 'target_time']].copy()
    rows['anchor_tick'] = np.rint(rows.anchor_time * 500).astype(np.int64)
    rows['actual_dbfs'] = y
    rows['predicted_dbfs'] = prediction
    return rows, selected, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=ROOT / 'preprocessed/1007/test/all_trials.csv')
    args = parser.parse_args()
    old_path = args.run_dir / 'previous_models/noise_predictor.joblib'
    new_path = ROOT / 'noise_prediction_model/models/feature_selection/noise_predictor.joblib'
    old, new = joblib.load(old_path), joblib.load(new_path)
    manifest = json.loads((args.run_dir / 'raw_manifest.json').read_text())
    test_records = [r for r in manifest if r['split'] == 'test']
    requested = {r['trial'] for r in test_records}
    for artifact in (old, new):
        if requested & set(artifact['config']['training_trials']):
            raise ValueError('A comparison trial was used in training')
    for key in ('lags_ms', 'sample_ms', 'horizon_ms', 'include_standing'):
        if old['config'][key] != new['config'][key]:
            raise ValueError(f'Target/window contract changed: {key}')
    frame = pd.read_csv(args.input)
    if set(frame.trial) != requested:
        raise ValueError('Held-out source differs from raw manifest')
    old_rows, _, old_coverage = artifact_predictions(old, frame)
    new_rows, x_new, new_coverage = artifact_predictions(new, frame)
    paired = old_rows.merge(new_rows, on=['trial', 'anchor_tick'],
                            suffixes=('_old', '_new'), validate='one_to_one')
    if len(paired) != len(old_rows) or len(paired) != len(new_rows):
        raise ValueError('Old/new prediction windows are not identical')
    np.testing.assert_allclose(paired.actual_dbfs_old, paired.actual_dbfs_new, atol=1e-8)
    np.testing.assert_allclose(paired.target_time_old, paired.target_time_new, rtol=0, atol=1e-7)
    saved = pd.read_csv(ROOT / 'noise_prediction_model/results/feature_selection/test/test_predictions.csv')
    saved['anchor_tick'] = np.rint(saved.anchor_time * 500).astype(np.int64)
    checked = paired.merge(saved[['trial', 'anchor_tick', new['config']['selected_model']]],
                            on=['trial', 'anchor_tick'], validate='one_to_one')
    if len(checked) != len(paired):
        raise ValueError('Saved test evaluation windows differ')
    np.testing.assert_allclose(checked.predicted_dbfs_new,
                               checked[new['config']['selected_model']], atol=1e-8)
    added = {r['trial'] for r in test_records if r['change'] == 'added'}
    motion = {r['trial']: motion_type(r) for r in test_records}
    paired['test_subset'] = np.where(paired.trial.isin(added), 'added_test', 'existing_test')
    paired['motion'] = paired.trial.map(motion)
    paired.to_csv(args.run_dir / 'paired_test_predictions.csv', index=False)
    trial_rows = []
    for trial, group in paired.groupby('trial', sort=True):
        for version in ('old', 'new'):
            trial_rows.append(dict(trial=trial, test_subset=group.test_subset.iloc[0],
                motion=motion[trial],
                version=version, n_samples=len(group),
                **metrics(group.actual_dbfs_new.to_numpy(), group[f'predicted_dbfs_{version}'].to_numpy())))
    trials = pd.DataFrame(trial_rows)
    trials.to_csv(args.run_dir / 'model_comparison_by_trial.csv', index=False)
    summary = []
    subsets = ['all_test', 'existing_test', 'added_test', *sorted(set(motion.values()))]
    for subset in subsets:
        part = (paired if subset == 'all_test' else paired[paired.test_subset.eq(subset)]
                if subset in ('existing_test', 'added_test') else paired[paired.motion.eq(subset)])
        if part.empty:
            continue
        for version in ('old', 'new'):
            t = trials[trials.version.eq(version) & trials.trial.isin(part.trial)]
            summary.append(dict(subset=subset, version=version, n_trials=t.trial.nunique(),
                n_samples=len(part), mean_trial_rmse_db=float(t.rmse_db.mean()),
                mean_trial_mae_db=float(t.mae_db.mean()), mean_trial_r2=float(t.r2.mean()),
                **{'pooled_' + k: v for k, v in metrics(part.actual_dbfs_new.to_numpy(),
                    part[f'predicted_dbfs_{version}'].to_numpy()).items()}))
    summary = pd.DataFrame(summary)
    summary.to_csv(args.run_dir / 'model_comparison_summary.csv', index=False)
    old_coverage.to_csv(args.run_dir / 'previous_model_current_test_coverage.csv', index=False)
    new_coverage.to_csv(args.run_dir / 'updated_model_current_test_coverage.csv', index=False)
    # Warm, one-row model prediction only; sensor and feature processing excluded.
    for i in range(50):
        new['model'].predict(x_new[i % len(x_new): i % len(x_new) + 1])
    elapsed = []
    for i in range(500):
        row = x_new[i % len(x_new): i % len(x_new) + 1]
        start = time.perf_counter_ns()
        new['model'].predict(row)
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    latency = dict(n_calls=500, measured_on='workspace host CPU, one thread, warm predict only',
        median_ms=float(np.median(elapsed)), p95_ms=float(np.percentile(elapsed, 95)),
        max_ms=float(np.max(elapsed)), model_inputs=int(x_new.shape[1]))
    (args.run_dir / 'inference_latency.json').write_text(json.dumps(latency, indent=2) + '\n')
    table = trials.pivot(index='trial', columns='version', values='rmse_db')
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    positions = np.arange(len(table))
    ax.bar(positions - .2, table.old, .4, label='Previous frozen model')
    ax.bar(positions + .2, table.new, .4, label='Updated model')
    ax.set(xticks=positions, xticklabels=table.index, ylabel='Held-out RMSE (dB)',
           title='Same updated test windows | selected model before vs after data update')
    ax.tick_params(axis='x', rotation=35)
    ax.legend(); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(args.run_dir / 'model_comparison.png', dpi=160); plt.close(fig)
    unchanged = all(sha256(r['path']) == r['sha256'] for r in manifest)
    if not unchanged:
        raise AssertionError('Raw data changed during update')
    verification = dict(raw_files=len(manifest), raw_files_unchanged=True,
        train_test_disjoint=True, identical_old_new_test_windows=True,
        new_predictions_match_saved_evaluation=True, n_paired_windows=len(paired),
        previous_model_sha256=sha256(old_path), updated_model_sha256=sha256(new_path),
        test_input_sha256=sha256(args.input), selected_old=old['config']['selected_model'],
        selected_new=new['config']['selected_model'], comparison_did_not_fit_models=True)
    (args.run_dir / 'comparison_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    lines = ['# 데이터 업데이트 전후 모델 비교', '',
        '두 저장 모델을 같은 업데이트된 test 창에 적용했다. 이 스크립트는 학습하거나 모델을 선택하지 않는다. '
        '새 모델의 종류·입력 선정·파라미터는 train 내부 검증으로 결정했다. test는 최종 평가에만 사용했다.', '',
        f"이전 대표 모델: `{old['config']['selected_model']}`, 새 대표 모델: `{new['config']['selected_model']}`.",
        f"학습 trial: {len(old['config']['training_trials'])} → {len(new['config']['training_trials'])}. "
        '기본 설정에 따라 명령 속도 0의 정지 기록은 모델 창에서 제외되며 coverage에 기록된다.', '',
        '|Test 범위|모델|Trial|동일 평가 창|평균 trial RMSE (dB)|전체 RMSE (dB)|전체 R²|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'|{r.subset}|{r.version}|{r.n_trials}|{r.n_samples}|{r.mean_trial_rmse_db:.4f}|{r.pooled_rmse_db:.4f}|{r.pooled_r2:.4f}|')
    lines += ['', '## Trial별 결과', '', '|Trial|이전 RMSE|새 RMSE|차이: 새−이전|', '|---|---:|---:|---:|']
    for trial, row in table.iterrows():
        lines.append(f'|{trial}|{row.old:.4f}|{row.new:.4f}|{row.new-row.old:+.4f}|')
    lines += ['', '평균 trial RMSE는 기록마다 같은 가중치를 준 값이다. 전체 RMSE/R²는 모든 창을 함께 계산한다. '
        '기존 6개 test와 추가 4개 test를 분리해서 보아 데이터 구성 변화와 모델 변화의 영향을 구분한다. '
        '같은 test 기록을 반복 확인한 결과이므로 새로운 날짜·바닥·마이크 조건에서의 독립적인 배포 성능을 보증하지 않는다.', '',
        '목표는 t+20ms를 중심으로 한 20ms 녹음 RMS dBFS다. 모터음과 발 충격음이 섞인 녹음 전체를 예측한다. '
        '착지 분석의 centered 변화율·착지 후 최대값을 새 실시간 입력으로 추가하지 않았다.', '',
        f"현 PC의 단일 행 추론 중앙값 {latency['median_ms']:.3f}ms, 95백분위 {latency['p95_ms']:.3f}ms. "
        '모델 계산만 측정했으며 로봇에서의 센서·특징 처리 포함 지연은 아니다.', '',
        '- model_comparison.png: trial별 전후 RMSE.',
        '- model_comparison_summary.csv / model_comparison_by_trial.csv: 집계·개별 지표.',
        '- paired_test_predictions.csv: 동일 창의 정답과 두 예측.',
        '- comparison_verification.json: 원본 해시·평가 창·저장된 test 예측 일치 검증.',
        '- previous_models/: 업데이트 전 모델 보관.', '']
    (args.run_dir / 'model_comparison_report.md').write_text('\n'.join(lines), encoding='utf-8')
    model_lines = ['# 현재 저장 모델의 업데이트·평가', '',
        f"대표 모델: `{new['config']['selected_model']}`. 모델 종류: `{type(new['model']).__name__}`.",
        f"학습 {len(new['config']['training_trials'])}개 trial / {new['config']['n_windows']:,}개 창. "
        f"대표 모델 입력 {x_new.shape[1]}차원. 입력 lag {new['config']['lags_ms']}ms, "
        f"주기 {new['config']['sample_ms']}ms, horizon {new['config']['horizon_ms']}ms.", '',
        'Train/test 폴더 구분을 유지했다. 명령 3축이 모두 0인 기록은 기존 기본 정책으로 모델 창에서 제외했다. '
        '측면 이동·제자리 회전은 포함했다. 명령값은 입력이 아니며 측정 선속도와 LowState·과거 접촉 파생값을 사용한다.', '',
        '|Test 범위|Trial|이전 평균 trial RMSE|새 평균 trial RMSE|새 전체 R²|',
        '|---|---:|---:|---:|---:|']
    for subset in summary.subset.unique():
        old_score = summary[summary.subset.eq(subset) & summary.version.eq('old')].iloc[0]
        new_score = summary[summary.subset.eq(subset) & summary.version.eq('new')].iloc[0]
        model_lines.append(f'|{subset}|{new_score.n_trials}|{old_score.mean_trial_rmse_db:.4f}|{new_score.mean_trial_rmse_db:.4f}|{new_score.pooled_r2:.4f}|')
    model_lines += ['', f'[같은 test 창에서의 상세 전후 비교](../../results/feature_selection/{args.run_dir.name}/model_comparison_report.md)', '',
        '[현재 test 평가](../../results/feature_selection/test/report.md) · '
        '[현재 시퀀스 교차검증](../../results/feature_selection/cross_validation/report.md)', '',
        '착지 전후의 centered 변화율·착지 후 최대값은 회고 분석용이다. 이번 학습에서는 기존 과거 정보만 쓰는 '
        '접촉 특징 정의를 유지했다. 소음은 분리된 발 충격음이 아닌 전체 녹음 RMS dBFS다.', '']
    (new_path.parent / 'evaluation_summary.md').write_text('\n'.join(model_lines), encoding='utf-8')
    print(summary.to_string(index=False))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
