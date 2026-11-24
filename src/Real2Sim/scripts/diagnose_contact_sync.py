#!/usr/bin/env python3
"""Check positive foot-load waveforms and touchdown residuals in saved results."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from real2sim.config import DEFAULT_DATASET, LEGS, ROOT, result_path
from real2sim.analyze_time_delay import sweep, estimate

sys.path.insert(0, str(ROOT / 'src/db_preprocessing/scripts'))
from contact_detection import detect_contact

LAGS = np.arange(-50, 51) * .005


def contact_events(t, force, valid, calibration):
    thresholds, events, states = [], [], []
    selected = valid & (t >= calibration[0]) & (t <= calibration[1])
    for foot in range(4):
        if selected.sum() < 20:
            thresholds.append(None); events.append([]); states.append(np.full(len(t), np.nan)); continue
        low, high = np.percentile(force[selected, foot], [10, 90])
        if high - low < 1.:
            thresholds.append(None); events.append([]); states.append(np.full(len(t), np.nan)); continue
        on, off = float(low + .15 * (high - low)), float(low + .07 * (high - low))
        values = force[:, foot].astype(float).copy()
        values[~valid] = np.nan
        state, _, detected = detect_contact(t, values, np.zeros(len(t)),
            on=on, off=off, confirm_ms=6., max_gap_s=.010)
        onsets, last_off = [], None
        for item in detected:
            stamp = item['onset_time_abs']
            if item['event'] == 'contact_off': last_off = stamp
            elif last_off is not None and stamp - last_off >= .040:
                onsets.append(float(stamp))
        thresholds.append(dict(on=on, off=off))
        events.append(onsets)
        states.append(np.where(state >= 0, state, np.nan))
    return thresholds, events, np.asarray(states).T


def match_onsets(real, sim, window, maximum=.150):
    selected = [stamp for stamp in real if window[0] <= stamp <= window[1]]
    used, pairs = set(), []
    for stamp in selected:
        candidates = [(abs(other - stamp), index, other) for index, other in enumerate(sim)
                      if index not in used and abs(other - stamp) <= maximum]
        if candidates:
            _, index, other = min(candidates)
            used.add(index)
            pairs.append(dict(real_s=stamp, sim_s=other, residual_ms=float((other - stamp) * 1000)))
    residuals = np.asarray([pair['residual_ms'] for pair in pairs])
    return dict(real_events=len(selected), matched=len(pairs),
        median_ms=float(np.median(residuals)) if len(residuals) else None,
        abs_p90_ms=float(np.percentile(abs(residuals), 90)) if len(residuals) else None, pairs=pairs)


def analyze_pair(data, synchronization):
    t, valid, steady = data['t'], data['valid'], data['steady']
    real, sim = data['real_foot_sensor_raw'], data['sim_foot_fz_N_before']
    if np.any(sim < -1e-6):
        raise ValueError('Saved sim load is negative; rebuild with the corrected S3 force convention first')
    windows = {label: synchronization[label]['window_s'] for label in ('calibration', 'validation')}
    real_thresholds, real_events, real_states = contact_events(t, real, valid, windows['calibration'])
    sim_thresholds, sim_events, sim_states = contact_events(t, sim, valid, windows['calibration'])
    feet = {}
    for foot, name in enumerate(LEGS):
        a, b = real[steady, foot], sim[steady, foot]
        correlation = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-8 else None
        feet[name] = dict(correlation=correlation, real_threshold=real_thresholds[foot],
            sim_threshold=sim_thresholds[foot], windows={})
    common = {}
    for label, window in windows.items():
        try:
            curves, _ = sweep(t, real, sim, steady, window, LAGS)
        except ValueError:
            curves = np.full((len(LAGS), 4), np.nan)
        active = np.isfinite(curves).all(axis=0)
        common[label] = estimate(curves[:, active].mean(axis=1), LAGS) if active.any() else None
        for foot, name in enumerate(LEGS):
            feet[name]['windows'][label] = dict(
                waveform=estimate(curves[:, foot], LAGS),
                touchdown=match_onsets(real_events[foot], sim_events[foot], window))
    return dict(applied_joint_shift_ms=synchronization['applied_lag_s'] * 1000,
        joint_alignment_validated=synchronization['passed'], windows_s=windows,
        common_residual_waveform_lag=common, feet=feet,
        interpretation='Residual after the existing joint shift; threshold-defined onset, not hardware latency; no force-only shift applied'), (real_states, sim_states)


def plot_pair(data, result, name, output, states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = data['t']
    mask = (t >= result['windows_s']['calibration'][0]) & (t <= result['windows_s']['validation'][1])
    fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex=True)
    for foot, leg in enumerate(LEGS):
        ax = axes[foot, 0]; right = ax.twinx()
        ax.plot(t[mask], data['real_foot_sensor_raw'][mask, foot], color='C0', lw=.9, label='Real counts')
        right.plot(t[mask], data['sim_foot_fz_N_before'][mask, foot], color='C1', lw=.9, label='Sim positive load')
        ax.set_ylabel(f'{leg}: counts', color='C0'); right.set_ylabel('N', color='C1')
        correlation = result['feet'][leg]['correlation']
        ax.set_title('Positive load convention | r=unavailable' if correlation is None else f'Positive load convention | r={correlation:.3f}')
        contact = axes[foot, 1]
        contact.step(t[mask], states[0][mask, foot], where='post', color='C0', label='Real')
        contact.step(t[mask], states[1][mask, foot] + 1.5, where='post', color='C1', label='Sim')
        contact.set_yticks([0, 1, 1.5, 2.5], ['Real off', 'Real on', 'Sim off', 'Sim on'])
        residual = result['feet'][leg]['windows']['validation']['touchdown']['median_ms']
        contact.set_title('Late touchdown residual: unavailable' if residual is None else f'Late touchdown residual: {residual:+.0f} ms (sim minus real)')
        for axis in (ax, contact):
            axis.axvspan(*result['windows_s']['validation'], color='green', alpha=.06)
            axis.grid(alpha=.2)
    for ax in axes[-1]: ax.set_xlabel('Real recording time [s]')
    fig.suptitle(f"{name}: applied joint shift {result['applied_joint_shift_ms']:+.0f} ms\nPositive S3 load = -recorded z. Green: held-out interval. No force-specific shift applied.")
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def diagnose(directory, plot_trial='straight_0.7_0.5_v2', output=None):
    directory = Path(directory)
    output = Path(output) if output else directory / 'diagnostics/contact_sync'
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((directory / 'metrics/mapping_summary.json').read_text())
    results, rows = [], []
    for trial in summary:
        name = trial['trial']
        with np.load(directory / 'processed' / f'{name}_mapping.npz', allow_pickle=False) as z:
            data = {key: z[key] for key in ('t', 'valid', 'steady', 'real_foot_sensor_raw', 'sim_foot_fz_N_before')}
        result, states = analyze_pair(data, trial['synchronization'])
        result.update(trial=name, provenance=trial['provenance'])
        results.append(result)
        for foot, record in result['feet'].items():
            for window, values in record['windows'].items():
                rows.append(dict(trial=name, foot=foot, window=window,
                    joint_shift_ms=result['applied_joint_shift_ms'], joint_validated=result['joint_alignment_validated'],
                    correlation=record['correlation'], additional_waveform_lag_ms=values['waveform']['lag_ms'],
                    waveform_search_boundary=values['waveform']['boundary'],
                    touchdown_median_ms=values['touchdown']['median_ms'],
                    touchdown_abs_p90_ms=values['touchdown']['abs_p90_ms'],
                    touchdown_matched=values['touchdown']['matched'], touchdown_real_events=values['touchdown']['real_events']))
        if name == plot_trial: plot_pair(data, result, name, output / f'{name}.png', states)
        print(name, 'q/dq shift', result['applied_joint_shift_ms'], 'correlations',
            {foot: record['correlation'] for foot, record in result['feet'].items()}, flush=True)
    (output / 'summary.json').write_text(json.dumps(dict(
        method='Positive load convention; same-foot hysteresis onsets (15%/7% of early p10-p90 range), 6ms confirmation, >=40ms swing; 150ms maximum one-to-one event match',
        replay_alignment_changed=False, results=results), indent=2, ensure_ascii=False, allow_nan=False))
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('Contact diagnostics saved:', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--plot-trial', default='straight_0.7_0.5_v2')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    diagnose(args.result_dir or result_path(args.dataset), args.plot_trial, args.output_dir)
