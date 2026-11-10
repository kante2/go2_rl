#!/usr/bin/env python3
"""Scatter individual landing Spearman coefficients against measured speed."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {
    'force': ('Contact force vs noise', 'tab:blue'),
    'vz': ('Signed vertical velocity vs noise', 'crimson'),
    'force_rate': ('Force rate (12ms) vs noise', 'tab:orange'),
    'vz_rate': ('Vertical velocity rate (12ms) vs noise', 'tab:purple'),
}


def measured_event_speeds(local, source_path):
    """One measured planar speed per event; preserve signed vx diagnostics."""
    frame = pd.read_csv(source_path, usecols=['time_abs', 'sp_vel_x', 'sp_vel_y'])
    times = frame.time_abs.to_numpy(float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f'{source_path}: invalid source time axis')
    centers = np.searchsorted(times, local.contact_time_abs.to_numpy() - 1e-8)
    offsets = np.arange(-10, 31)  # Same -20..+60ms window as each rho.
    indices = centers[:, None] + offsets[None, :]
    if indices.min() < 0 or indices.max() >= len(times):
        raise ValueError(f'{source_path}: incomplete correlation window')
    if not np.allclose(times[indices] - local.contact_time_abs.to_numpy()[:, None],
                       offsets[None, :] * .002, atol=1e-7, rtol=0):
        raise ValueError(f'{source_path}: correlation window crosses a gap')
    values = frame.sp_vel_x.to_numpy(float)[indices]
    lateral = frame.sp_vel_y.to_numpy(float)[indices]
    if not np.isfinite(values).all() or not np.isfinite(lateral).all():
        raise ValueError(f'{source_path}: nonfinite measured speed')
    q = np.percentile(values, [25, 50, 75], axis=1)
    planar = np.hypot(values, lateral)
    planar_q = np.percentile(planar, [25, 50, 75], axis=1)
    return pd.DataFrame(dict(event_id=local.event_id.to_numpy(),
                             measured_vx_q25_mps=q[0], measured_vx_median_mps=q[1],
                             measured_vx_q75_mps=q[2], measured_vx_at_contact_mps=values[:, 10],
                             measured_vy_median_mps=np.median(lateral, axis=1),
                             measured_planar_speed_q25_mps=planar_q[0],
                             measured_planar_speed_median_mps=planar_q[1],
                             measured_planar_speed_q75_mps=planar_q[2],
                             measured_planar_speed_at_contact_mps=planar[:, 10],
                             measured_speed_window_n_samples=values.shape[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
                        default=ROOT / 'noise_prediction_model/results/analysis/landing_dynamics')
    parser.add_argument('--preprocessed-dir', type=Path, default=ROOT / 'preprocessed/1007/train')
    args = parser.parse_args()
    output = args.input_dir / 'trials'
    names = {p.name.removesuffix('_landing_profiles.png') for p in output.glob('*_landing_profiles.png')}
    if not names:
        raise ValueError('No trial waveform figures found')
    events = pd.read_csv(args.input_dir / 'landing_event_metrics.csv')
    if not events.split.eq('train').all() or events.event_id.duplicated().any():
        raise ValueError('Expected unique train-only events')
    events = events[events.trial.isin(names)].copy()
    if set(events.trial) != names:
        raise ValueError('Trial plots and event table do not match')
    speeds = []
    for trial, local in events.groupby('trial', sort=False):
        speeds.append(measured_event_speeds(local, args.preprocessed_dir / f'{trial}.csv'))
    columns = ['event_id', 'trial', 'leg', 'command_vx', 'contact_time_abs', 'synchronization_status',
               *[f'{signal}_waveform_spearman_rho' for signal in FIELDS]]
    columns += [c for c in ('command_vy', 'command_vyaw') if c in events]
    points = events[columns].merge(pd.concat(speeds, ignore_index=True), on='event_id', validate='one_to_one')
    points.to_csv(output / 'spearman_summary.csv', index=False)
    # Retain a current auxiliary trial table without using it in the scatter.
    by_trial = []
    for trial, group in points.groupby('trial', sort=True):
        row = dict(trial=trial, n_events=len(group), synchronization_status=group.synchronization_status.iloc[0],
                   measured_planar_speed_median_mps=float(group.measured_planar_speed_median_mps.median()))
        for signal in FIELDS:
            values = group[f'{signal}_waveform_spearman_rho']
            row[f'{signal}_spearman_rho'] = values.median()
            row[f'{signal}_n_defined'] = int(values.notna().sum())
        by_trial.append(row)
    pd.DataFrame(by_trial).to_csv(output / 'spearman_by_trial.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, layout='constrained')
    x = points.measured_planar_speed_median_mps.to_numpy()
    for ax, (signal, (title, color)) in zip(axes.flat, FIELDS.items()):
        values = points[f'{signal}_waveform_spearman_rho'].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(values)
        ax.scatter(x[finite], values[finite], color=color, s=9, alpha=.2, edgecolors='none', rasterized=True)
        ax.axhline(0, color='.35', lw=.8)
        ax.set(ylim=(-1, 1), xlim=(x.min() - .04, x.max() + .04), title=f'{title} | n={finite.sum():,} landings',
               xlabel='Measured planar body speed (SportMode |v_xy|, m/s)',
               ylabel='Per-landing Spearman rho (lag = 0ms)')
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Individual landing events | each dot is one landing\n'
                 'x: median measured planar speed within that event; y: waveform Spearman in [-20,+60]ms\n'
                 'No speed bins, trial aggregation, fitted lines or distribution bands', fontsize=12)
    fig.savefig(output / 'spearman_summary.png', dpi=175)
    plt.close(fig)
    (output / 'spearman_summary.md').write_text(
        '# 개별 착지 이벤트의 Spearman 산점도\n\n'
        'spearman_summary.png와 CSV는 착지 이벤트 하나를 점 하나/행 하나로 표시한다. '
        '속도별 묶음, trial별 중앙값 집계, 전체 평균, 추세선, 음영을 사용하지 않는다.\n\n'
        '- x: 해당 이벤트 −20~+60ms의 41개 sqrt(sp_vel_x² + sp_vel_y²) 표본 중앙값. '
        '다른 이벤트와 합치지 않고 각 이벤트 구간 안에서 측정 속도의 대표값만 계산한다. '
        '좌우 이동 기록 추가에 맞춰 전진 성분만 사용하던 축을 평면 속도 크기로 바꿨다. '
        'CSV에는 평면 속도의 25/75분위·접촉 0ms 값과 기존 부호 있는 sp_vel_x 지표도 저장한다. '
        '제자리 회전의 각속도는 이 선속도 축으로 표현되지 않는다. '
        'SportMode 몸체 선속도의 측정/추정값이며 발 속도 또는 독립 외부 측정 진실값은 아니다.\n'
        '- y: 같은 이벤트 −20~+60ms에서 신호와 centered 5ms RMS 소음을 각각 순위로 바꾸고 '
        '같은 시각의 순위끼리 Pearson을 구한 Spearman이다. 기존 이벤트별 값을 그대로 읽는다. '
        '속도나 소음의 단일 순간값끼리 상관을 구한 값은 아니다.\n'
        '- 네 패널: 접촉력 / 부호 있는 수직 속도 / 12ms 힘 변화율 / 12ms 속도 변화율. '
        '수직 속도는 양수 상승·음수 하강이며 착지 직전 하강 속도 크기 한 값과 다르다.\n\n'
        f'{len(points):,}개 착지 / {points.trial.nunique()}개 train trial. 각 점은 반투명하게 표시해 겹치는 분포를 볼 수 있게 했다. '
        '상관이 정의되지 않는 이벤트는 해당 패널에서 생략하며 CSV에는 빈 값으로 남긴다. '
        'trial별 이벤트 수가 달라 점 밀도에는 표본 수가 반영되며, 공유 소음·겹치는 창 때문에 점들이 독립 표본인 것은 아니다. '
        '명령 속도는 추적용 컬럼일 뿐 x축이나 분류에 쓰지 않는다. '
        'spearman_by_trial.csv는 최신 trial별 보조 집계이며 현재 산점도의 원자료는 spearman_summary.csv다.\n\n'
        '접촉 0ms는 기존 힘 임계값 검출 시작이며 실제 최초 접촉이나 발 정지 완료의 정답이 아니다. '
        '해당 한계는 ../contact_timing_audit/report.md에 정리했다.\n\n'
        '재실행: `.venv/bin/python noise_prediction_model/analysis/summarize_landing_spearman.py`\n', encoding='utf-8')
    print(f'Saved {len(points):,} individual landings / {points.trial.nunique()} trials -> {output / "spearman_summary.png"}')
    print(points[[f'{signal}_waveform_spearman_rho' for signal in FIELDS]].notna().sum().to_string())


if __name__ == '__main__':
    main()
