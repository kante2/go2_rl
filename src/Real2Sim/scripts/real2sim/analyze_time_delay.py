#!/usr/bin/env python3
"""Inspect a recorded real/sim pair's apparent lag; do not change replay alignment."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import minimum_filter1d

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'real2sim'
from .config import PROJECT, JOINTS, DEFAULT_DATASET, result_path
from .preprocessing import load, sha, prepare_pair, event, filtered, interp


def sweep(t, real, sim, valid, window, lags):
    # Keep the same samples for all candidate shifts, including gap exclusions.
    margin = int(np.ceil(max(abs(lags)) / np.median(np.diff(t))))
    safe = minimum_filter1d(valid.astype(int), size=2 * margin + 1, mode='constant') > 0
    mask = safe & (t >= window[0]) & (t <= window[1])
    if mask.sum() < 100:
        raise ValueError('At least 100 shared valid samples required per window')
    x = real[mask] - real[mask].mean(axis=0)
    scores = []
    for lag in lags:
        y = interp(t, sim, t[mask] + lag)
        y -= y.mean(axis=0)
        denominator = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
        scores.append(np.divide((x * y).sum(axis=0), denominator,
            out=np.full(real.shape[1], np.nan), where=denominator > 1e-8))
    return np.asarray(scores), mask


def estimate(curve, lags):
    if not np.isfinite(curve).any():
        return dict(lag_ms=None, correlation_before=None, correlation_best=None, boundary=False)
    best = int(np.nanargmax(curve))
    return dict(lag_ms=round(float(lags[best] * 1000), 3),
        correlation_before=float(curve[np.argmin(abs(lags))]),
        correlation_best=float(curve[best]), boundary=best in (0, len(lags) - 1))


def analyze(trial, joint, calibration, validation, output, result_dir=None):
    directory = Path(result_dir) if result_dir is not None else result_path()
    summary = next(r for r in json.loads((directory / 'metrics/mapping_summary.json').read_text()) if r['trial'] == trial)
    provenance = summary['provenance']
    for domain in ('real', 'sim'):
        if sha(Path(provenance[domain + 'Path'])) != provenance[domain + 'Sha256']:
            raise ValueError('Recording changed since mapping export; rebuild first')
    raw_logs = {domain: load(Path(provenance[domain + 'Path']))[0] for domain in ('real', 'sim')}
    # Always diagnose the original PC-clock data, even after replay applies a lag.
    data, _ = prepare_pair(raw_logs['real'], raw_logs['sim'], 15., .005)
    for domain in ('real', 'sim'):
        for quantity in ('q', 'dq', 'ddq', 'tau'):
            data[f'{domain}_{quantity}_before'] = data[f'{domain}_{quantity}']
    t = data['t']
    for domain in ('real', 'sim'):
        raw = raw_logs[domain]
        if domain == 'real':
            origin = event(raw, 'move_start')
        native = raw['sim_time'] if domain == 'sim' else raw['lc_t'] - origin
        uniform, signal = filtered(native, raw['lc_q'], .005 if domain == 'sim' else .002, 15.)
        data[domain + '_target_before'] = interp(raw['lc_t'] - origin, interp(uniform, signal, native), t)
    lags = np.arange(-250, 251, dtype=float) / 1000
    focus = JOINTS.index(joint)
    results, curves, masks = {}, {}, {}
    for quantity in ('q', 'dq', 'ddq', 'tau', 'target'):
        results[quantity], curves[quantity] = {}, {}
        for label, window in (('calibration', calibration), ('validation', validation)):
            curve, mask = sweep(t, data[f'real_{quantity}_before'], data[f'sim_{quantity}_before'],
                                data['steady'], window, lags)
            curves[quantity][label], masks[label] = curve, mask
            results[quantity][label] = {name: estimate(curve[:, j], lags) for j, name in enumerate(JOINTS)}
    # A single shift is learned from the early q window of the chosen joint.
    # Apply that exact shift to all quantities for an independent late check.
    shift = results['q']['calibration'][joint]['lag_ms'] / 1000
    validation_metrics = {}
    mask = masks['validation']
    for quantity in ('q', 'dq', 'ddq', 'tau', 'target'):
        real, sim = data[f'real_{quantity}_before'], data[f'sim_{quantity}_before']
        shifted = interp(t, sim, t[mask] + shift)
        validation_metrics[quantity] = {name: dict(
            rmse_before=float(np.sqrt(np.mean((sim[mask, j] - real[mask, j]) ** 2))),
            rmse_with_q_calibration_shift=float(np.sqrt(np.mean((shifted[:, j] - real[mask, j]) ** 2))),
            correlation_with_q_calibration_shift=float(np.corrcoef(real[mask, j], shifted[:, j])[0, 1]))
            for j, name in enumerate(JOINTS)}
    output.mkdir(parents=True, exist_ok=True)
    report = dict(trial=trial, joint=joint, provenance=provenance,
        calibration_window_s=list(calibration), validation_window_s=list(validation),
        diagnostic_shift_s=shift, replay_alignment_changed=False,
        current_replay_shift_s=summary['synchronization']['applied_lag_s'],
        sign='positive lag: sim waveform occurs later; aligned sim(t) = original sim(t + lag)',
        interpretation='Waveform phase offset includes policy phase and dynamics; not an identified transport latency.',
        method='Pearson correlation, 1ms grid within +/-250ms; no amplitude fit; fixed valid samples; late window held out',
        estimates=results, validation_metrics=validation_metrics)
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    with (output / 'lag_estimates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['quantity', 'window', 'joint', 'lag_ms', 'correlation_before', 'correlation_best', 'boundary'])
        writer.writeheader()
        for quantity, windows in results.items():
            for window, joints in windows.items():
                for name, values in joints.items():
                    writer.writerow(dict(quantity=quantity, window=window, joint=name, **values))
    plot(data, report, output)
    for quantity in results:
        print(quantity, {label: results[quantity][label][joint] for label in results[quantity]},
              'held-out', validation_metrics[quantity][joint], flush=True)
    print(f'Diagnostics saved: {output}; replay alignment unchanged', flush=True)


def plot(data, report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    focus = JOINTS.index(report['joint'])
    t, shift = data['t'], report['diagnostic_shift_s']
    mask = (t >= report['calibration_window_s'][0]) & (t <= report['validation_window_s'][1])
    fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex=True)
    for row, (quantity, unit) in enumerate((('q', 'rad'), ('dq', 'rad/s'), ('ddq', 'rad/s2'), ('tau', 'N m'))):
        for column, lag in enumerate((0., shift)):
            ax = axes[row, column]
            ax.plot(t[mask], data[f'real_{quantity}_before'][mask, focus], label='Real', lw=1)
            ax.plot(t[mask], np.interp(t[mask] + lag, t, data[f'sim_{quantity}_before'][:, focus]), label='Existing sim', lw=1)
            ax.axvspan(*report['validation_window_s'], color='green', alpha=.06)
            ax.set_ylabel(f'{quantity} [{unit}]'); ax.grid(alpha=.2)
            ax.set_title('Clock baseline: no time shift' if column == 0 else f'Diagnostic only: same +{shift * 1000:.0f} ms shift for all channels')
    for ax in axes[-1]: ax.set_xlabel('Time from real recording start [s]')
    axes[0, 0].legend()
    fig.suptitle(f"{report['trial']} / {report['joint']}\nShift fitted from early q only; green area is the held-out late interval. Replay data unchanged.")
    fig.tight_layout(); fig.savefig(output / 'delay_comparison.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    x = np.arange(12)
    for ax, quantity in zip(axes.flat, ('q', 'dq', 'ddq', 'tau')):
        for label in ('calibration', 'validation'):
            ax.plot(x, [report['estimates'][quantity][label][j]['lag_ms'] for j in JOINTS], 'o-', label=label)
        ax.axhline(0, color='gray', lw=.7)
        ax.set_xticks(x, [j.removesuffix('_joint') for j in JOINTS], rotation=65, ha='right', fontsize=8)
        ax.set_ylabel('Apparent lag [ms]'); ax.set_title(quantity); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Per-channel apparent lags; different optima do not identify one hardware latency')
    fig.tight_layout(); fig.savefig(output / 'lag_by_joint.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--trial', default='straight_0.7_0.5_v2')
    parser.add_argument('--joint', choices=JOINTS, default='FR_thigh_joint')
    parser.add_argument('--calibration', type=float, nargs=2, default=(2.3, 4.4))
    parser.add_argument('--validation', type=float, nargs=2, default=(4.8, 7.1))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not args.calibration[0] < args.calibration[1] < args.validation[0] < args.validation[1]:
        parser.error('Early calibration and late validation windows must not overlap')
    directory = args.result_dir or result_path(args.dataset)
    analyze(args.trial, args.joint, args.calibration, args.validation,
            args.output_dir or directory / 'diagnostics' / args.trial, directory)
