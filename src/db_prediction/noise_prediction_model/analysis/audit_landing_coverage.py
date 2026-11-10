#!/usr/bin/env python3
"""Account for every raw trial and audit fixed-threshold landing coverage."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'noise_prediction_model/results/analysis/landing_dynamics'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Update run containing raw_manifest.json with added/unchanged flags')
    args = parser.parse_args()
    output = BASE / 'coverage_audit'
    output.mkdir(exist_ok=True)
    metrics = pd.read_csv(BASE / 'landing_event_metrics.csv')
    analyzed = metrics.groupby('trial').size()
    rows, leg_rows = [], []
    for split in ('train', 'test'):
        folder = ROOT / f'preprocessed/1007/{split}'
        metadata = json.loads((folder / 'metadata.json').read_text())
        events = pd.read_csv(folder / 'contact_event_features.csv')
        coverage = pd.read_csv(ROOT / ('noise_prediction_model/results/feature_selection/cross_validation/coverage.csv'
            if split == 'train' else 'noise_prediction_model/results/feature_selection/test/test_coverage.csv'))
        for name in metadata[split + '_trials']:
            frame = pd.read_csv(folder / f'{name}.csv')
            local = events[events.trial.eq(name)]
            row = dict(split=split, trial=name, n_source_rows=len(frame),
                n_detected_events=len(local), n_feature_valid=int(local.feature_valid.sum()),
                n_landing_analysis_events=int(analyzed.get(name, 0)) if split == 'train' else 0,
                analysis_status=('included' if name in analyzed else 'no_force_threshold_contact_on_events'
                    if len(local) == 0 else 'no_complete_expanded_windows')
                    if split == 'train' else 'held_out_from_train_landing_analysis',
                n_model_windows=int(coverage.set_index('trial').loc[name, 'n_windows']),
                model_window_status=coverage.set_index('trial').loc[name, 'reason'])
            rows.append(row)
            for leg in ('FR', 'FL', 'RR', 'RL'):
                force = frame['ls_ff_' + leg.lower()].to_numpy()
                q = np.percentile(force, [0, 1, 10, 50, 100])
                leg_rows.append(dict(split=split, trial=name, leg=leg,
                    force_min=q[0], force_q01=q[1], force_q10=q[2], force_median=q[3], force_max=q[4],
                    fraction_force_le_off5=float(np.mean(force <= 5)),
                    never_reaches_off5=bool(np.min(force) > 5),
                    n_detected_contact_on_events=int(local.leg.eq(leg).sum())))
    trials, legs = pd.DataFrame(rows), pd.DataFrame(leg_rows)
    trials.to_csv(BASE / 'landing_trial_coverage.csv', index=False)
    legs.to_csv(output / 'force_floor_by_trial_leg.csv', index=False)
    train = trials[trials.split.eq('train')]
    missing = train[train.n_detected_events.eq(0)]
    manifest = json.loads((args.run_dir / 'raw_manifest.json').read_text())
    new_names = sorted(r['trial'] for r in manifest if r['split'] == 'train' and r['change'] == 'added')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, layout='constrained')
    for ax, leg in zip(axes.flat, ('FR', 'FL', 'RR', 'RL')):
        local = legs[legs.split.eq('train') & legs.leg.eq(leg)].set_index('trial').loc[new_names]
        x = np.arange(len(local))
        ax.plot(x, local.force_q01, 'o', color='tab:orange', label='1st percentile of force')
        ax.plot(x, local.force_min, 'x', color='tab:blue', label='Minimum force')
        ax.axhline(5, color='black', ls='--', label='Existing OFF threshold = 5')
        for pos, row in enumerate(local.itertuples()):
            ax.annotate(str(row.n_detected_contact_on_events), (pos, row.force_q01), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=8)
        ax.set(title=leg, ylabel='Force (uncalibrated sensor units)', xticks=x, xticklabels=new_names)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Added train recordings | numbers: detected landings per foot\n'
                 'A force signal that never falls to OFF cannot generate repeated ON events', fontsize=12)
    fig.savefig(output / 'added_trials_force_floor.png', dpi=165); plt.close(fig)
    lines = ['# 착지 검출 범위와 힘 신호 하한 검토', '',
        f'Train {len(train)}개를 전부 전처리했다. 착지 검출 결과가 있는 기록은 '
        f'{int(train.n_detected_events.gt(0).sum())}개, 확장 창 분석에 포함된 착지는 {len(metrics):,}개다. '
        '검출 0건인 기록을 누락된 입력과 구분하기 위해 landing_trial_coverage.csv에 모든 기록을 남겼다.', '',
        '## 검출 0건인 train 기록', '', '|Trial|행 수|검출 착지|전체 시계열 모델 창|', '|---|---:|---:|---:|']
    for r in missing.itertuples():
        lines.append(f'|{r.trial}|{r.n_source_rows}|{r.n_detected_events}|{r.n_model_windows}|')
    lines += ['', '현재 검출기는 OFF ≤5 / ON ≥10을 6ms 확인한다. '
        '이 기록들에서는 힘이 OFF 기준까지 내려오지 않아 접촉 상태가 해제되지 않는다. '
        '이는 실제 착지가 없다는 뜻이 아니다. 추가 기록의 힘 신호 하한이 기존 전진 기록과 다르며, '
        '센서 영점 변화인지 잔류 하중·접촉 방식 차이인지는 이 기록만으로 확정하지 못한다.', '',
        '## 추가 train 기록별 검출량', '', '|Trial|검출 착지|확장 창 포함|시계열 모델 창|', '|---|---:|---:|---:|']
    for r in train[train.trial.isin(new_names)].itertuples():
        lines.append(f'|{r.trial}|{r.n_detected_events}|{r.n_landing_analysis_events}|{r.n_model_windows}|')
    lines += ['', '추가 기록의 착지 통계는 임계값을 통과한 일부 발·구간에 치우쳐 있다. '
        '따라서 업데이트된 착지 Spearman이나 분포가 새 좌우 이동·회전을 충분히 대표한다고 해석하면 안 된다. '
        '전체 시계열 예측 모델은 착지로 행을 제한하지 않으므로 이 기록들도 학습했다. '
        '다만 기존 접촉 상태·경과시간 파생값 역시 같은 검출 기준의 영향을 받는다.', '',
        '힘 하한을 공중 상태의 센서 영점으로 단정해 빼거나, 소음 상관·test 성능에 맞춰 임계값을 조정하지 않았다. '
        '검출 방법을 바꾸려면 원본 힘·운동학 파형과 독립적인 발 들림 증거를 먼저 확인하고, '
        'train에서 방법을 정한 후 test에는 동일하게 적용해야 한다.', '',
        '- ../landing_trial_coverage.csv: 모든 train/test 기록의 전처리·이벤트·모델 창 포함 여부.',
        f'- force_floor_by_trial_leg.csv: 모든 {len(trials)}개 기록의 발별 힘 범위 및 OFF 기준 도달 비율.',
        '- added_trials_force_floor.png: 추가 train 기록의 힘 하한·발별 검출량.', '',
        'Test의 힘 범위는 품질 진단으로만 기록했다. 모델·임계값 선정에 사용하지 않았다. '
        '기존 착지 시각·힘·속도·소음·모델을 변경하지 않았다.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    result = dict(n_preprocessed_trials=len(trials), n_train_trials=len(train),
        n_train_trials_with_detected_events=int(train.n_detected_events.gt(0).sum()),
        n_train_trials_with_analyzed_events=int(train.n_landing_analysis_events.gt(0).sum()),
        no_detected_train_trials=missing.trial.tolist(),
        n_added_train_detected_events=int(train[train.trial.isin(new_names)].n_detected_events.sum()),
        n_added_train_analyzed_events=int(train[train.trial.isin(new_names)].n_landing_analysis_events.sum()),
        no_threshold_or_signal_changes=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    notice = ('> 전체 원본 기록의 포함 여부와 새 기록의 OFF 임계값 미도달 문제는 '
              '[검출 범위 검토](coverage_audit/report.md) 및 '
              '[전체 기록 표](landing_trial_coverage.csv)를 함께 확인한다.\n\n')
    for name in ('interpretation.md', 'report.md'):
        path = BASE / name
        text = path.read_text()
        if notice not in text:
            path.write_text(notice + text, encoding='utf-8')
    cfg = json.loads((ROOT / 'noise_prediction_model/models/feature_selection/run.json').read_text())
    selected_contact = sorted(set(cfg['selected_lowstate_features']) & set(cfg.get('derived_contact_candidates', [])))
    selected_text = ', '.join(f'`{name}`' for name in selected_contact) or '없음'
    model_notice = ('\n추가 기록에서 힘 신호가 기존 OFF=5 기준에 도달하지 않아 착지 검출이 드문 문제가 있다. '
                    '전체 시계열 학습에는 포함했다. 접촉 상태·경과시간 후보는 이 기준의 영향을 받지만, '
                    f'현재 대표 모델의 선정된 접촉 파생 입력은 {selected_text}이다. '
                    '[검출 범위와 힘 하한 검토](../../results/analysis/landing_dynamics/coverage_audit/report.md)를 참고한다.\n')
    model_report = ROOT / 'noise_prediction_model/models/feature_selection/evaluation_summary.md'
    content = model_report.read_text()
    if model_notice not in content:
        model_report.write_text(content + model_notice, encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
