#!/usr/bin/env python3
"""Actual noise distributions vs landing force, velocity and finite-time rates."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {
    'force': 'Foot contact force (sensor units)',
    'vz': 'Signed vertical foot velocity (+up / -down, m/s)',
    'force_rate': 'Force change rate over 12ms (sensor units/s)',
    'vz_rate': 'Vertical velocity change rate over 12ms (m/s², estimated)',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(input_dir):
    samples_path = input_dir / 'landing_aligned_samples.csv'
    events_path = input_dir / 'landing_event_metrics.csv'
    hashes = {str(p.resolve()): digest(p) for p in (samples_path, events_path)}
    samples = pd.read_csv(samples_path)
    events = pd.read_csv(events_path)
    if events.empty or not events.split.eq('train').all() or events.event_id.duplicated().any():
        raise ValueError('Expected unique nonempty train-only events')
    if samples.duplicated(['event_id', 'relative_time_ms']).any():
        raise ValueError('Duplicate event/time sample rows')
    samples = samples[samples.relative_time_ms.between(-20, 60)].copy()
    counts = samples.groupby('event_id').size()
    if not counts.eq(41).all() or set(counts.index) != set(events.event_id):
        raise ValueError('Expected 41 time points per event in [-20,+60]ms')
    for _, group in samples.groupby('event_id', sort=False):
        if not np.array_equal(group.relative_time_ms.to_numpy(), np.arange(-20, 62, 2)):
            raise ValueError('Unexpected event-relative sampling')
    samples = samples.merge(events[['event_id', 'synchronization_status']], on='event_id', validate='many_to_one')
    if not np.isfinite(samples[[*FIELDS, 'noise_5ms_dbfs', 'noise_change_db']].to_numpy()).all():
        raise ValueError('Nonfinite plotted values')
    return samples, events, hashes


def scatter_figure(samples, output, quality_only=False, limits=None):
    if quality_only:
        samples = samples[samples.synchronization_status.eq('CONSISTENT_CANDIDATE')]
    if samples.empty:
        return
    # Deterministic random drawing order prevents the last foot/event dominating
    # overlaps. Every point is retained and neither x nor y is jittered.
    order = np.random.default_rng(42).permutation(len(samples))
    data = samples.iloc[order]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True, layout='constrained')
    norm = Normalize(-20, 60)
    for ax, (field, label) in zip(axes.flat, FIELDS.items()):
        picture = ax.scatter(data[field], data.noise_5ms_dbfs,
                             c=data.relative_time_ms, cmap='viridis', norm=norm,
                             s=2, alpha=.12, edgecolors='none', rasterized=True)
        ax.set(xlabel=label, ylabel='Measured noise (5ms RMS, dBFS)', title=label)
        ax.grid(alpha=.15); ax.set_axisbelow(True)
        if limits is not None:
            ax.set_xlim(limits[field])
            ax.set_ylim(limits['noise'])
    # Use an opaque colorbar; scatter alpha affects the marks only.
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axes,
                       shrink=.85, pad=.02)
    bar.set_label('Time from force-threshold onset (ms)')
    tag = 'Synchronization-candidate events' if quality_only else 'All valid landing events'
    fig.suptitle(f'{tag}: force / velocity vs actual noise\n'
                 f'One dot = one foot-event time sample; {len(samples):,} samples / '
                 f'{samples.event_id.nunique():,} events / {samples.trial.nunique()} trials\n'
                 'Same timestamps in [-20,+60]ms; all points retained; no event maxima or trial averaging', fontsize=12)
    name = 'force_velocity_noise_scatter_sync_candidate.png' if quality_only else 'force_velocity_noise_scatter.png'
    fig.savefig(output / name, dpi=170); plt.close(fig)


def density_figure(samples, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True, layout='constrained')
    rows = []
    noise = samples.noise_5ms_dbfs.to_numpy()
    for ax, (field, label) in zip(axes.flat, FIELDS.items()):
        counts, xedges, yedges = np.histogram2d(samples[field].to_numpy(), noise, bins=80)
        if int(counts.sum()) != len(samples):
            raise AssertionError('Histogram lost source points')
        masked = np.ma.masked_equal(counts.T, 0)
        picture = ax.pcolormesh(xedges, yedges, masked, cmap='magma',
                                norm=LogNorm(vmin=1, vmax=max(2, counts.max())), shading='flat', rasterized=True)
        ax.set(xlabel=label, ylabel='Measured noise (5ms RMS, dBFS)', title=label)
        fig.colorbar(picture, ax=ax, shrink=.85, label='Number of samples (log scale)')
        for i, j in np.argwhere(counts > 0):
            rows.append(dict(feature=field, x_left=xedges[i], x_right=xedges[i+1],
                             noise_left_dbfs=yedges[j], noise_right_dbfs=yedges[j+1], n_samples=int(counts[i, j])))
    pd.DataFrame(rows).to_csv(output / 'noise_distribution_histogram.csv', index=False)
    fig.suptitle('Noise distribution density: where the time-sample pairs concentrate\n'
                 'y = actual noise level; color = sample count, not mean noise; all ranges included', fontsize=12)
    fig.savefig(output / 'force_velocity_noise_density.png', dpi=170); plt.close(fig)


def event_change_figure(events, output):
    xfields = {'force_post_minus_pre': 'Force change: post[+10,+30] minus pre[-20,0) (sensor units)',
               'vz_post_minus_pre': 'Vertical velocity change: post minus pre (m/s)'}
    targets = {'noise_post40_rms_dbfs': 'Post-contact noise RMS [0,+40)ms (dBFS)',
               'noise_post40_minus_pre_db': 'Noise RMS change: post[0,+40) minus pre[-50,-20) (dB)'}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for row, (field, label) in enumerate(xfields.items()):
        for col, (target, ylabel) in enumerate(targets.items()):
            axes[row, col].scatter(events[field], events[target], s=7, color='tab:blue' if row == 0 else 'crimson',
                                    alpha=.18, edgecolors='none', rasterized=True)
            axes[row, col].set(xlabel=label, ylabel=ylabel)
            axes[row, col].axvline(0, color='.5', lw=.6)
            if col == 1:
                axes[row, col].axhline(0, color='.5', lw=.6)
            axes[row, col].grid(alpha=.15)
    fig.suptitle(f'Before/after changes per landing | one dot = one event, n={len(events):,}\n'
                 'Within-event window means define force/velocity changes; no pooling across events', fontsize=12)
    fig.savefig(output / 'landing_changes_noise_scatter.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
                        default=ROOT / 'noise_prediction_model/results/analysis/landing_dynamics')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.input_dir / 'noise_distributions'
    output.mkdir(parents=True, exist_ok=True)
    samples, events, hashes = load_data(args.input_dir)
    columns = ['event_id', 'trial', 'leg', 'relative_time_ms', 'time_abs', *FIELDS,
               'noise_5ms_dbfs', 'noise_change_db', 'synchronization_status']
    samples[columns].to_csv(output / 'noise_distribution_samples.csv', index=False)
    event_columns = ['event_id', 'trial', 'leg', 'synchronization_status', 'force_pre20_mean',
                     'force_post10_30_mean', 'force_post_minus_pre', 'vz_pre20_mean',
                     'vz_post10_30_mean', 'vz_post_minus_pre', 'noise_baseline_rms_dbfs',
                     'noise_post40_rms_dbfs', 'noise_post40_minus_pre_db']
    events[event_columns].to_csv(output / 'noise_distribution_event_changes.csv', index=False)
    limits = {field: (float(samples[field].min()), float(samples[field].max())) for field in FIELDS}
    limits['noise'] = (float(samples.noise_5ms_dbfs.min()), float(samples.noise_5ms_dbfs.max()))
    scatter_figure(samples, output, limits=limits)
    scatter_figure(samples, output, quality_only=True, limits=limits)
    density_figure(samples, output)
    event_change_figure(events, output)
    if any(digest(path) != expected for path, expected in hashes.items()):
        raise AssertionError('Input changed during plotting')
    quality = samples.synchronization_status.eq('CONSISTENT_CANDIDATE')
    metadata = dict(n_events=len(events), n_trials=int(events.trial.nunique()), n_samples=len(samples),
                    n_sync_candidate_events=int(samples.loc[quality, 'event_id'].nunique()),
                    n_sync_candidate_samples=int(quality.sum()), input_sha256=hashes, input_unchanged=True,
                    script_sha256=digest(__file__), relative_time_ms=[-20, 60], dt_ms=2,
                    noise='Existing centered 5ms RMS from raw PCM, DC removed, dBFS; no added shift',
                    derivatives='Existing centered finite differences across +/-6ms, total span 12ms',
                    histogram='80x80 bins, count only, entire finite data range; empty bins blank',
                    event_change='Existing within-event pre[-20,0) and post[10,30] mean differences; not peak values',
                    no_event_or_trial_averaging_in_sample_plots=True, no_subsampling=True)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# 접촉력·발 속도에 따른 실제 소음 분포\n\n'
        '## 주 그림\n\n'
        '`force_velocity_noise_scatter.png`: x는 접촉력·부호 있는 수직 속도·힘 변화율·속도 변화율, '
        'y는 같은 시각의 실제 녹음 소음 5ms RMS dBFS다. Spearman이 아니다. '
        '색은 접촉력 임계값 교차로부터의 상대 시각이다. 보라색은 이전, 노랑에 가까울수록 이후 시점이다. '
        '점 하나는 발 이벤트 하나의 특정 시각이며, −20~+60ms의 41개 점을 착지마다 모두 표시한다. '
        '평균선·최대값 추출·trial별 합산·점 생략·값 범위 자르기는 하지 않았다.\n\n'
        '`force_velocity_noise_density.png`: 같은 x/y 자료의 2차원 히스토그램이다. '
        '밝을수록 해당 힘/속도와 소음 조합의 관측 수가 많다. 색은 소음 평균이 아니며 y축이 소음이다. '
        '정규화 확률밀도나 독립 사건 수가 아니라, 발 이벤트에 속한 시간 표본의 개수다.\n\n'
        f'`force_velocity_noise_scatter_sync_candidate.png`: 동기화 진단 후보 {samples.loc[quality, "trial"].nunique()}개 trial의 같은 산점도. '
        '전체와 같은 축 범위를 사용한다. 후보 통과가 완벽한 시간 동기화를 보장하지는 않는다.\n\n'
        '## 착지별 전후 변화\n\n'
        '`landing_changes_noise_scatter.png`: 점 하나는 착지 하나다. x는 그 착지의 힘 또는 수직 속도의 '
        '[+10,+30]ms 평균에서 [−20,0)ms 평균을 뺀 값이다. 왼쪽 y는 [0,+40)ms 원본 PCM RMS, '
        '오른쪽 y는 이 값에서 [−50,−20)ms RMS를 뺀 dB 변화다. '
        '서로 다른 이벤트를 평균내지는 않지만, 변화량 정의를 위해 한 이벤트 안의 두 구간 평균을 사용한다.\n\n'
        '## 해석 범위\n\n'
        '수직 속도는 양수 상승·음수 하강인 발 기준점의 추정 속도다. '
        '속도 변화율의 양수는 하강하는 발의 감속일 수도 있으며 항상 상승 동작을 의미하지 않는다. '
        '12ms 변화율은 짧은 구간의 변화이지 무한히 짧은 순간 미분값은 아니다. '
        '힘 단위는 보정 전 sensor units이며, 소음은 dB SPL이 아닌 dBFS다. '
        'dBFS가 클수록(음수의 절댓값이 작을수록) 녹음 RMS가 크다.\n\n'
        '0ms는 기존 힘 임계값 검출 시작이다. 정확한 최초 접촉·정지 완료 시점은 아니며, '
        '검출 시점을 옮기거나 접촉 후 속도를 강제로 0으로 만들지 않았다. '
        '같은 시각끼리 비교하며 별도 시간차 보정은 하지 않았다.\n\n'
        '전체 trial·발·접촉 단계를 모은 기술적 분포다. 명령/측정 몸체 속도나 접촉 단계 등을 통제한 '
        '조건부 효과 또는 인과관계는 아니다. 다른 발의 근접 착지는 같은 마이크 소리를 공유하므로 '
        '소음을 발 하나에 배정하지 못하며, 겹치는 이벤트와 오디오 창 때문에 표본들은 독립적이지 않다. '
        '점 밀도에는 trial별 이벤트 수와 각 접촉 단계의 지속시간도 반영된다.\n\n'
        '## 데이터\n\n'
        f'- {len(events):,}개 착지 / {events.trial.nunique()}개 train trial / {len(samples):,}개 시간 표본.\n'
        '- noise_distribution_samples.csv: 주 산점도의 모든 점과 ID·상대 시각·동기화 상태.\n'
        '- noise_distribution_event_changes.csv: 착지별 전후 변화 그림 원자료.\n'
        '- noise_distribution_histogram.csv: 비어 있지 않은 80×80 히스토그램 구간의 경계와 표본 수.\n'
        '- metadata.json: 입력 해시, 표본 수 및 정의. 기존 입력 파일은 변경하지 않았다.\n\n'
        '재실행: `.venv/bin/python noise_prediction_model/analysis/plot_landing_noise_distributions.py`\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print(f'Saved noise distributions -> {output}')


if __name__ == '__main__':
    main()
