#!/usr/bin/env python3
"""Landing-aligned waveforms and descriptive temporal noise associations.

Train only. Preserve the force-detected event times and the existing audio
clock. No peak matching, per-event shifting, significance tests or model fit.
"""

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
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'preprocessing'))
from go2_kinematics import add_foot_kinematics, MODEL_PATH
from contact_event_extraction import continuous_segments

LEGS = ('FL', 'FR', 'RL', 'RR')
COLORS = dict(zip(LEGS, ('tab:blue', 'tab:orange', 'tab:green', 'tab:purple')))
DT = .002
REL_MS = np.arange(-50, 102, 2)
HALF_STEPS = 3
SIGNALS = {
    'noise_change_db': ('5ms RMS change\nfrom pre-window (dB)', 'black'),
    'force': ('Contact force\n(sensor units)', 'tab:blue'),
    'vz': ('Vertical foot velocity\n(+up / -down, m/s)', 'crimson'),
    'force_rate': ('Force rate over 12ms\n(sensor units/s)', 'tab:orange'),
    'vz_rate': ('Velocity rate over 12ms\n(m/s², estimated)', 'tab:purple'),
}
DRIVERS = ('force', 'vz', 'force_rate', 'vz_rate')
SUBSETS = ('all_valid', 'sync_candidate', 'isolated_extended', 'isolated_extended_sync')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pcm_rms(audio, slope, t0, starts, stops):
    """DC-removed PCM RMS for exact half-open clock intervals [start,stop)."""
    starts, stops = np.asarray(starts), np.asarray(stops)
    left = np.ceil((starts - t0) / slope - 1).astype(np.int64)
    right = np.ceil((stops - t0) / slope - 1).astype(np.int64)
    valid = (left >= 0) & (right <= len(audio)) & (right > left)
    left = np.clip(left, 0, len(audio)); right = np.clip(right, 0, len(audio))
    audio = np.asarray(audio, float)
    sums = np.r_[0., np.cumsum(audio)]
    squares = np.r_[0., np.cumsum(audio * audio)]
    count = np.maximum(right - left, 1)
    mean = (sums[right] - sums[left]) / count
    rms = np.sqrt(np.maximum((squares[right] - squares[left]) / count - mean * mean, 0))
    rms = np.where(valid, rms, np.nan)
    return 20 * np.log10(np.maximum(rms, 1e-12) / 32768.), valid


def centered_rate(values, segments, dt=DT, half_steps=HALF_STEPS):
    """Finite change over +/-6ms; never crosses a missing-data segment."""
    values = np.asarray(values, float)
    result = np.full(values.shape, np.nan)
    if len(values) <= 2 * half_steps:
        return result
    middle = np.arange(half_steps, len(values) - half_steps)
    good = (segments[middle - half_steps] == segments[middle + half_steps])
    use = middle[good]
    result[use] = (values[use + half_steps] - values[use - half_steps]) / (2 * half_steps * dt)
    return result


def row_correlation(x, y):
    """Pearson across each row; inputs here are finite complete windows."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    return np.divide(numerator, denominator, out=np.full(len(x), np.nan), where=denominator > 1e-12)


def lag_correlations(x, noise, relative_ms=REL_MS, lags_ms=np.arange(-30, 32, 2)):
    """Event-wise Spearman x(t) vs noise(t+lag), fixed driver support [-20,60]."""
    core = np.flatnonzero((relative_ms >= -20) & (relative_ms <= 60))
    xrank = rankdata(x[:, core], axis=1)
    result = []
    for lag in lags_ms:
        shifted = core + int(round(lag / 2))
        if shifted.min() < 0 or shifted.max() >= len(relative_ms):
            raise ValueError('lag extends outside exported windows')
        yrank = rankdata(noise[:, shifted], axis=1)
        result.append(row_correlation(xrank, yrank))
    return np.column_stack(result)


def subset_masks(events):
    quality = events.synchronization_status.eq('CONSISTENT_CANDIDATE').to_numpy()
    isolated = events.other_landings_in_extended_window.eq(0).to_numpy()
    return dict(all_valid=np.ones(len(events), bool), sync_candidate=quality,
                isolated_extended=isolated, isolated_extended_sync=quality & isolated)


def source_ages(source, times):
    index = np.searchsorted(source, times, side='right') - 1
    age = (times - source[np.clip(index, 0, len(source) - 1)]) * 1000
    age[(index < 0) | (times > source[-1])] = np.nan
    return age


def extract_trial(frame, events, raw_path):
    """Return complete 76-sample windows plus an explicit exclusion audit."""
    frames = events.body_velocity_frame.unique()
    if len(frames) != 1:
        raise ValueError('mixed body velocity conventions')
    frame = add_foot_kinematics(frame, velocity_frame=frames[0])
    t = frame.time_abs.to_numpy(float)
    segments, dt = continuous_segments(frame)
    if not np.isclose(dt, DT, atol=1e-8):
        raise ValueError('expected 500Hz input')
    # All kept segments must themselves follow the 2ms grid.
    same = segments[1:] == segments[:-1]
    if not np.allclose(np.diff(t)[same], DT, atol=1e-8):
        raise ValueError('nonuniform segment')
    delays = frame.audio_delay_s.unique()
    if len(delays) != 1:
        raise ValueError('mixed audio delays')
    with np.load(raw_path, allow_pickle=False) as raw:
        audio = raw['audio']; slope = float(raw['audio_slope'])
        t0 = float(raw['audio_t0']) - float(delays[0])
        ls_age = source_ages(raw['ls_t'], t)
        sp_age = source_ages(raw['sp_t'], t)
    envelope, audio_good = pcm_rms(audio, slope, t0, t - .0025, t + .0025)
    onsets = events.contact_time_abs.to_numpy(float)
    baselines, baseline_good = pcm_rms(audio, slope, t0, onsets - .050, onsets - .020)
    post, post_good = pcm_rms(audio, slope, t0, onsets, onsets + .040)
    source_good = np.isfinite(ls_age) & np.isfinite(sp_age) & (ls_age <= 20 + 1e-6) & (sp_age <= 20 + 1e-6)
    source_good &= (ls_age >= -1e-6) & (sp_age >= -1e-6)
    values = {}
    for leg in LEGS:
        force = frame[f'ls_ff_{leg.lower()}'].to_numpy(float)
        vz = frame[f'foot_velocity_world_{leg.lower()}_z_mps'].to_numpy(float)
        values[leg] = dict(force=force, vz=vz, force_rate=centered_rate(force, segments),
                           vz_rate=centered_rate(vz, segments))
    arrays = {name: [] for name in (*SIGNALS, 'noise_5ms_dbfs', 'noise_20ms_dbfs')}
    accepted, audit, neighbors = [], [], {}
    offsets = (REL_MS / 2).astype(int)
    for event_number, event in enumerate(events.to_dict('records')):
        center = int(np.searchsorted(t, event['contact_time_abs'] - 1e-8))
        indices = center + offsets
        margin = np.arange(indices[0] - HALF_STEPS, indices[-1] + HALF_STEPS + 1)
        reason = ''
        if not event['feature_valid']:
            reason = 'original_feature_invalid'
        elif center >= len(t) or abs(t[center] - event['contact_time_abs']) > 1e-7:
            reason = 'event_not_on_source_grid'
        elif margin[0] < 0 or margin[-1] >= len(t):
            reason = 'expanded_window_or_derivative_at_boundary'
        elif np.any(segments[margin] != segments[center]):
            reason = 'expanded_window_or_derivative_crosses_gap'
        elif not source_good[margin].all():
            reason = 'robot_source_age_invalid'
        elif not audio_good[indices].all() or not baseline_good[event_number] or not post_good[event_number]:
            reason = 'audio_window_incomplete'
        else:
            for name, value in values[event['leg']].items():
                if not np.isfinite(value[indices]).all():
                    reason = 'nonfinite_waveform'
        audit.append(dict(event_id=event['event_id'], trial=event['trial'], leg=event['leg'],
                          included=not reason, exclusion_reason=reason,
                          synchronization_status=event['synchronization_status']))
        if reason:
            continue
        actual = (t[indices] - event['contact_time_abs']) * 1000
        if not np.allclose(actual, REL_MS, atol=1e-5):
            raise AssertionError('event extraction shifted time')
        neighbor = events[(events.contact_time_abs >= event['contact_time_abs'] - .050 - 1e-9) &
                          (events.contact_time_abs <= event['contact_time_abs'] + .100 + 1e-9) &
                          (events.event_id != event['event_id'])]
        neighbors[event['event_id']] = [(float((r.contact_time_abs - event['contact_time_abs']) * 1000), r.leg)
                                       for r in neighbor.itertuples()]
        event = dict(event, noise_baseline_rms_dbfs=baselines[event_number],
                     noise_post40_rms_dbfs=post[event_number],
                     noise_post40_minus_pre_db=post[event_number] - baselines[event_number],
                     other_landings_in_extended_window=len(neighbor),
                     other_legs_in_extended_window=','.join(sorted(neighbor.leg.unique())),
                     max_ls_age_ms=float(np.max(ls_age[margin])), max_sp_age_ms=float(np.max(sp_age[margin])))
        for name in ('force', 'vz', 'force_rate', 'vz_rate'):
            arrays[name].append(values[event['leg']][name][indices])
        arrays['noise_5ms_dbfs'].append(envelope[indices])
        arrays['noise_20ms_dbfs'].append(frame.audio_dbfs_20ms.to_numpy(float)[indices])
        arrays['noise_change_db'].append(envelope[indices] - baselines[event_number])
        accepted.append(event)
    result = {k: np.asarray(v, float).reshape(-1, len(REL_MS)) for k, v in arrays.items()}
    return pd.DataFrame(accepted), result, pd.DataFrame(audit), neighbors


def event_metrics(events, arrays):
    events = events.copy()
    before = (REL_MS >= -20) & (REL_MS < 0)
    after = (REL_MS >= 10) & (REL_MS <= 30)
    search = (REL_MS >= -20) & (REL_MS <= 60)
    search_indices = np.flatnonzero(search)
    for name in ('force', 'vz'):
        events[f'{name}_pre20_mean'] = arrays[name][:, before].mean(axis=1)
        events[f'{name}_post10_30_mean'] = arrays[name][:, after].mean(axis=1)
        events[f'{name}_post_minus_pre'] = events[f'{name}_post10_30_mean'] - events[f'{name}_pre20_mean']
    for name in ('noise_5ms_dbfs', 'force_rate', 'vz_rate'):
        peak = np.argmax(arrays[name][:, search], axis=1)
        events[f'{name}_peak_time_ms'] = REL_MS[search_indices[peak]]
        events[f'{name}_peak_at_search_edge'] = (peak == 0) | (peak == len(search_indices) - 1)
    for name in DRIVERS:
        x, y = arrays[name][:, search], arrays['noise_5ms_dbfs'][:, search]
        events[f'{name}_waveform_pearson_r'] = row_correlation(x, y)
        events[f'{name}_waveform_spearman_rho'] = row_correlation(rankdata(x, axis=1), rankdata(y, axis=1))
    return events


def group_indices(events, mask):
    # Equal trial weighting is applied after within-trial event aggregation.
    return [np.flatnonzero(mask & events.trial.eq(trial).to_numpy())
            for trial in events.loc[mask, 'trial'].unique()]


def profile_table(events, arrays):
    rows = []
    masks = subset_masks(events)
    for subset, mask in masks.items():
        for leg in ('ALL', *LEGS):
            use = mask if leg == 'ALL' else mask & events.leg.eq(leg).to_numpy()
            groups = group_indices(events, use)
            if not groups:
                continue
            for name in SIGNALS:
                event_profile = arrays[name][use]
                balanced = np.array([np.median(arrays[name][group], axis=0) for group in groups])
                q = np.percentile(balanced, [25, 50, 75], axis=0)
                eq = np.percentile(event_profile, [25, 50, 75], axis=0)
                for i, time in enumerate(REL_MS):
                    rows.append(dict(subset=subset, leg=leg, signal=name, relative_time_ms=int(time),
                                     n_events=int(use.sum()), n_trials=len(groups),
                                     trial_balanced_q25=q[0, i], trial_balanced_median=q[1, i], trial_balanced_q75=q[2, i],
                                     event_q25=eq[0, i], event_median=eq[1, i], event_q75=eq[2, i]))
    return pd.DataFrame(rows)


def lag_tables(events, arrays):
    lags = np.arange(-30, 32, 2)
    rows, trial_rows = [], []
    masks = subset_masks(events)
    for name in DRIVERS:
        coefficients = lag_correlations(arrays[name], arrays['noise_5ms_dbfs'], lags_ms=lags)
        for subset, mask in masks.items():
            for leg in ('ALL', *LEGS):
                use = mask if leg == 'ALL' else mask & events.leg.eq(leg).to_numpy()
                groups = group_indices(events, use)
                if not groups:
                    continue
                medians = []
                for group in groups:
                    c = coefficients[group]
                    with np.errstate(invalid='ignore'):
                        median = np.nanmedian(c, axis=0)
                    medians.append(median)
                    for j, lag in enumerate(lags):
                        trial_rows.append(dict(subset=subset, leg=leg, trial=events.trial.iloc[group[0]],
                                               signal=name, lag_ms=int(lag), n_events=len(group),
                                               n_defined=int(np.isfinite(c[:, j]).sum()), median_event_rho=median[j]))
                medians = np.asarray(medians)
                for j, lag in enumerate(lags):
                    local = medians[:, j]; local = local[np.isfinite(local)]
                    rows.append(dict(subset=subset, leg=leg, signal=name, lag_ms=int(lag),
                                     n_events=int(use.sum()), n_trials=len(groups), n_defined_trials=len(local),
                                     mean_trial_median_rho=float(np.mean(local)) if len(local) else np.nan,
                                     median_trial_median_rho=float(np.median(local)) if len(local) else np.nan,
                                     trial_q25=float(np.percentile(local, 25)) if len(local) else np.nan,
                                     trial_q75=float(np.percentile(local, 75)) if len(local) else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(trial_rows)


def time_resolved_table(events, arrays):
    """Across-event amplitude relations at each fixed landing-relative time."""
    rows = []
    for subset in ('all_valid', 'sync_candidate'):
        mask = subset_masks(events)[subset]
        for leg in ('ALL', *LEGS):
            use = mask if leg == 'ALL' else mask & events.leg.eq(leg).to_numpy()
            if use.sum() < 3:
                continue
            groups = group_indices(events, use)
            for name in DRIVERS:
                x, y = arrays[name][use], arrays['noise_5ms_dbfs'][use]
                pooled = row_correlation(rankdata(x, axis=0).T, rankdata(y, axis=0).T)
                xr, yr = [], []
                for group in groups:
                    a = rankdata(arrays[name][group], axis=0); b = rankdata(arrays['noise_5ms_dbfs'][group], axis=0)
                    xr.append(a - a.mean(axis=0)); yr.append(b - b.mean(axis=0))
                within = row_correlation(np.concatenate(xr).T, np.concatenate(yr).T)
                for j, time in enumerate(REL_MS):
                    rows.append(dict(subset=subset, leg=leg, signal=name, relative_time_ms=int(time),
                                     n_events=int(use.sum()), n_trials=len(groups),
                                     pooled_spearman_rho=pooled[j], within_trial_rank_rho=within[j]))
    return pd.DataFrame(rows)


def decorate(ax):
    ax.axvline(0, color='.25', ls='--', lw=1)
    ax.axhline(0, color='.7', lw=.6)
    ax.set_xlim(-50, 100)
    ax.grid(alpha=.15)


def draw_profiles(events, arrays, output, trial=None):
    if trial is None:
        columns = ('all_valid', 'sync_candidate')
        masks = subset_masks(events)
        titles = ('All valid landings', 'Synchronization candidates')
    else:
        columns = LEGS
        masks = {leg: events.trial.eq(trial).to_numpy() & events.leg.eq(leg).to_numpy() for leg in LEGS}
        titles = LEGS
    fig, axes = plt.subplots(5, len(columns), figsize=(7 * len(columns) if trial is None else 17, 13),
                             sharex=True, squeeze=False, constrained_layout=True)
    for col, key in enumerate(columns):
        mask = masks[key]; groups = group_indices(events, mask)
        axes[0, col].set_title(f'{titles[col]} | n={mask.sum():,}; trials={len(groups)}')
        for row, (name, (label, color)) in enumerate(SIGNALS.items()):
            ax = axes[row, col]
            if mask.any():
                if trial is None:
                    data = np.array([np.median(arrays[name][g], axis=0) for g in groups])
                else:
                    data = arrays[name][mask]
                    # Time-ordered evenly spaced examples; no audio selection.
                    for i in np.linspace(0, len(data) - 1, min(24, len(data))).astype(int):
                        ax.plot(REL_MS, data[i], color=color, alpha=.07, lw=.5)
                q = np.percentile(data, [25, 50, 75], axis=0)
                ax.fill_between(REL_MS, q[0], q[2], color=color, alpha=.18)
                ax.plot(REL_MS, q[1], color=color, lw=1.7)
            decorate(ax)
            if col == 0:
                ax.set_ylabel(label)
            if row == 4:
                ax.set_xlabel('Time relative to force-detected landing (ms)')
    title = ('Across-trial landing dynamics | median of trial medians; band = trial IQR'
             if trial is None else f'{trial}: landing dynamics by foot | median and event IQR; faint lines = time-selected events')
    fig.suptitle(title + '\nNo audio-based time shifting; noise baseline is [-50,-20)ms', fontsize=13)
    fig.savefig(output, dpi=155); plt.close(fig)


def draw_examples(events, arrays, neighbors, output, trial='0.3_1'):
    mask = events.trial.eq(trial)
    if not mask.any():
        return
    midpoint = (events.loc[mask, 'contact_time_from_move_start'].min() + events.loc[mask, 'contact_time_from_move_start'].max()) / 2
    fig, axes = plt.subplots(5, 4, figsize=(18, 13), sharex=True, constrained_layout=True)
    chosen = []
    for col, leg in enumerate(LEGS):
        candidates = events.index[mask & events.leg.eq(leg)]
        if not len(candidates):
            continue
        i = min(candidates, key=lambda i: abs(events.contact_time_from_move_start.iloc[i] - midpoint))
        event = events.iloc[i]; chosen.append(event.event_id)
        axes[0, col].set_title(f'{leg}: t={event.contact_time_from_move_start:.3f}s\nOther landings in window: {event.other_landings_in_extended_window}')
        for row, (name, (label, color)) in enumerate(SIGNALS.items()):
            ax = axes[row, col]
            if name == 'noise_change_db':
                ax.plot(REL_MS, arrays['noise_20ms_dbfs'][i] - event.noise_baseline_rms_dbfs,
                        color='.65', lw=1.1, label='20ms RMS')
            ax.plot(REL_MS, arrays[name][i], color=color, lw=1.5,
                    label='5ms RMS' if name == 'noise_change_db' else name)
            ax.axvline(event.noise_5ms_dbfs_peak_time_ms, color='black', ls=':', lw=1, alpha=.65)
            for when, other_leg in neighbors[event.event_id]:
                ax.axvline(when, color=COLORS[other_leg], alpha=.45, lw=1)
                if row == 0:
                    ax.text(when, .02, other_leg, transform=ax.get_xaxis_transform(), color=COLORS[other_leg], fontsize=8)
            decorate(ax)
            if col == 0:
                ax.set_ylabel(label)
            if row == 4:
                ax.set_xlabel('Time from landing (ms)')
        axes[0, col].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'{trial}: one central-time landing per foot, selected without audio\n'
                 'Dashed = own landing; dotted = local 5ms noise maximum [-20,+60]ms; colored lines = other landings', fontsize=12)
    fig.savefig(output, dpi=165); plt.close(fig)
    return chosen


def draw_heatmap(events, arrays, output, trial='0.3_1'):
    if not events.trial.eq(trial).any():
        return
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    for row, leg in enumerate(LEGS):
        indices = np.flatnonzero(events.trial.eq(trial) & events.leg.eq(leg))
        indices = indices[np.argsort(events.contact_time_abs.iloc[indices].to_numpy())]
        for col, name in enumerate(('force', 'vz_rate', 'noise_change_db')):
            ax = axes[row, col]
            if not len(indices):
                continue
            data = arrays[name][indices]
            if name == 'vz_rate':
                limit = np.percentile(abs(data), 98)
                vmin, vmax, cmap = -limit, limit, 'RdBu_r'
            else:
                vmin, vmax = np.percentile(data, [2, 98]); cmap = 'viridis' if name == 'force' else 'magma'
            picture = ax.imshow(data, aspect='auto', origin='lower', extent=(-51, 101, -.5, len(indices)-.5),
                                vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
            ax.axvline(0, color='white', ls='--', lw=.9)
            ax.set(xlim=(-50, 100), ylabel=f'{leg}: event order (n={len(indices)})')
            if row == 0:
                ax.set_title(SIGNALS[name][0].replace('\n', ' '))
            if row == 3:
                ax.set_xlabel('Time from landing (ms)')
            fig.colorbar(picture, ax=ax, shrink=.85)
    fig.suptitle(f'{trial}: every included landing in chronological order | colors clipped to panel 2–98 percentiles', fontsize=12)
    fig.savefig(output, dpi=150); plt.close(fig)


def draw_associations(lags, time_table, output):
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True, constrained_layout=True)
    for col, subset in enumerate(('all_valid', 'sync_candidate')):
        for row, name in enumerate(DRIVERS):
            ax = axes[row, col]
            data = lags[(lags.subset == subset) & lags.leg.eq('ALL') & lags.signal.eq(name)]
            if len(data):
                ax.fill_between(data.lag_ms, data.trial_q25, data.trial_q75, color=SIGNALS[name][1], alpha=.18)
                ax.plot(data.lag_ms, data.mean_trial_median_rho, color=SIGNALS[name][1], lw=1.8)
            ax.axhline(0, color='.5', lw=.6); ax.axvline(0, color='.5', ls='--', lw=.8)
            ax.set(ylim=(-1, 1), ylabel=f'{name}\nWithin-event Spearman rho')
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(subset)
            if row == 3:
                ax.set_xlabel('Lag (ms): positive = noise sampled after driver')
    fig.suptitle('Temporal co-variation within landings: driver(t) vs 5ms noise(t + lag)\n'
                 'Mean of trial median correlations; band = trial IQR; fixed driver interval [-20,+60]ms', fontsize=12)
    fig.savefig(output / 'landing_waveform_lag_correlations.png', dpi=155); plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True, constrained_layout=True)
    for col, subset in enumerate(('all_valid', 'sync_candidate')):
        for row, name in enumerate(DRIVERS):
            ax = axes[row, col]
            data = time_table[(time_table.subset == subset) & time_table.leg.eq('ALL') & time_table.signal.eq(name)]
            ax.plot(data.relative_time_ms, data.pooled_spearman_rho, color='.6', label='Pooled events')
            ax.plot(data.relative_time_ms, data.within_trial_rank_rho, color=SIGNALS[name][1], label='Within-trial ranks')
            decorate(ax); ax.set(ylim=(-1, 1), ylabel=f'{name}\nAcross-event rank correlation')
            if row == 0:
                ax.set_title(subset); ax.legend(fontsize=8)
            if row == 3:
                ax.set_xlabel('Time from landing (ms)')
    fig.suptitle('At each fixed time: do larger driver values accompany louder noise across landings?\n'
                 'This differs from temporal co-variation inside one landing', fontsize=12)
    fig.savefig(output / 'landing_time_resolved_correlations.png', dpi=155); plt.close(fig)


def write_samples(events, arrays, path):
    n = len(REL_MS)
    samples = pd.DataFrame(dict(event_id=np.repeat(events.event_id.to_numpy(), n),
                                trial=np.repeat(events.trial.to_numpy(), n), leg=np.repeat(events.leg.to_numpy(), n),
                                relative_time_ms=np.tile(REL_MS, len(events)),
                                time_abs=(events.contact_time_abs.to_numpy()[:, None] + REL_MS[None, :] / 1000).ravel()))
    for name, values in arrays.items():
        samples[name] = values.ravel()
    samples.to_csv(path, index=False, float_format='%.12g')


def before_after_table(events):
    rows = []
    metrics = ('force_post_minus_pre', 'vz_post_minus_pre', 'noise_post40_minus_pre_db',
               'noise_5ms_dbfs_peak_time_ms', 'force_rate_peak_time_ms', 'vz_rate_peak_time_ms')
    for subset, mask in subset_masks(events).items():
        for leg in ('ALL', *LEGS):
            use = mask if leg == 'ALL' else mask & events.leg.eq(leg).to_numpy()
            local = events[use]
            if local.empty:
                continue
            for metric in metrics:
                values = local[metric].to_numpy(float)
                by_trial = local.groupby('trial')[metric].median().to_numpy(float)
                q = np.percentile(values, [25, 50, 75])
                rows.append(dict(subset=subset, leg=leg, metric=metric, n_events=len(local),
                                 n_trials=local.trial.nunique(), event_q25=q[0], event_median=q[1], event_q75=q[2],
                                 mean_trial_median=float(by_trial.mean()), median_trial_median=float(np.median(by_trial)),
                                 fraction_positive=float(np.mean(values > 0))))
    return pd.DataFrame(rows)


def write_report(events, audit, before_after, lags, output):
    masks = subset_masks(events)
    descriptions = {
        'force_post_minus_pre': '접촉력 변화 (sensor units)',
        'vz_post_minus_pre': '수직 속도 변화 (m/s, 양수=상향 방향 변화)',
        'noise_post40_minus_pre_db': '후 40ms RMS − 전 구간 RMS (dB)',
        'noise_5ms_dbfs_peak_time_ms': '소음 5ms RMS 최대 시점 (ms)',
        'force_rate_peak_time_ms': '힘 증가율 최대 시점 (ms)',
        'vz_rate_peak_time_ms': '수직 속도 증가율 최대 시점 (ms)',
    }
    lines = [
        '# 착지 전후의 힘·속도·소음 시계열 비교', '',
        '## 분석 질문', '',
        '1. 착지 전후에 접촉력이 증가하고 하강 중인 발이 감속할 때, 소음도 반복적으로 증가하는가?',
        '2. 한 번의 착지 안에서 시간에 따라 함께 변하는 관계와, 착지마다 힘·속도 값이 클수록 더 시끄러운 관계는 같은가?', '',
        '## 데이터와 정렬', '',
        f'- Train {events.trial.nunique()}개 trial. 기존 착지 {len(audit):,}건 중 확장 창까지 사용할 수 있는 {len(events):,}건.',
        '- 착지 하나당 −50~+100ms, 2ms 간격 76행. 창과 변화율 계산 범위가 공백·segment 경계를 넘으면 제외한다.',
        '- 기존 force on=10 / off=5, 6ms 확인 후 최초 교차로 소급한 착지 시각을 그대로 사용한다. '
        '0ms는 실제 최초 발바닥 접촉의 추정값이다. 검출 자체에 힘을 사용하므로 힘의 정렬된 증가만으로 소음과의 관계를 증명할 수 없다.',
        '- 소음이나 속도 peak로 이벤트를 선택·이동하지 않는다. 오디오 지연도 추가로 추정하지 않는다. Test는 사용하지 않는다.',
        '- 기존 feature_valid에 더해 확장 창 및 변화율 양끝의 LowState/Sport 원본 표본 age ≤20ms와 유효 오디오를 확인한다.', '',
        '|Subset|이벤트|Trial|', '|---|---:|---:|',
    ]
    for subset, mask in masks.items():
        lines.append(f'|{subset}|{mask.sum():,}|{events.loc[mask,"trial"].nunique()}|')
    lines += ['', '제외 사유:', '']
    for reason, count in audit.loc[~audit.included, 'exclusion_reason'].value_counts().items():
        lines.append(f'- `{reason}`: {count}건')
    lines += [
        '', '## 신호 계산', '',
        '- F(t): 기존 LowState 발 접촉력. 보정되지 않은 sensor units이며 N이 아니다.',
        '- v_z(t): 기존 Go2 URDF Jacobian, q/dq·gyro·자세 및 Sport 몸체 선속도로 계산한 지면 좌표계 발 수직 속도. '
        '양수 상승 / 음수 하강을 그대로 유지한다. foot frame 원점의 강체 추정값으로 발바닥 접촉점 실측이 아니다.',
        '- 힘 변화율: `[F(t+6ms) − F(t−6ms)] / 0.012s`.',
        '- 속도 변화율: `[v_z(t+6ms) − v_z(t−6ms)] / 0.012s`. '
        '하강 속도가 0으로 바뀌면 양수다. 추정 속도의 변화율이며 별도 가속도 센서 측정값이 아니다.',
        '- 두 변화율은 12ms 구간 변화량이며 무한히 짧은 순간 변화율이 아니다. 미래 6ms를 사용하는 회고적 분석이다.',
        '- 소음: 원본 int16 PCM에서 centered 5ms 창마다 DC 제거 RMS를 계산해 dBFS로 표시한다. '
        '기존 centered 20ms RMS도 samples에 보존하고 개별 사례 그림에 겹쳐 표시한다.',
        '- 소음 기준값: 해당 착지 −50~−20ms 미만 원본 PCM의 DC 제거 RMS. '
        '5ms 파형에서 이 dBFS를 빼서 전 구간 대비 진폭 변화를 표시한다. 기준 구간이 순수 배경음이라는 가정은 하지 않는다.',
        '- 착지 전후 요약: 힘·속도는 [−20,0)ms 평균과 [+10,+30]ms 평균의 차이. '
        '소음은 [0,+40)ms 원본 PCM RMS와 [−50,−20)ms RMS의 dB 차이.', '',
        '## 착지 전후 수치', '',
        '아래 값은 trial마다 이벤트 중앙값을 구한 뒤 trial들 사이의 중앙값이다. '
        '이벤트 수가 많은 trial이 전체 결과를 지배하지 않도록 했으며 신뢰구간은 아니다.', '',
        '|지표|전체 유효|동기화 후보|', '|---|---:|---:|',
    ]
    for metric, label in descriptions.items():
        values = []
        for subset in ('all_valid', 'sync_candidate'):
            row = before_after[(before_after.subset == subset) & before_after.leg.eq('ALL') & before_after.metric.eq(metric)]
            values.append(f'{row.median_trial_median.iloc[0]:+.3f}' if len(row) else '없음')
        lines.append(f'|{label}|{values[0]}|{values[1]}|')
    lines += [
        '', '## 두 종류의 상관관계를 구분', '',
        '**착지 내부 파형 상관:** 각 이벤트에서 고정 구간 [−20,+60]ms의 driver(t)와 noise(t+lag)의 Spearman을 계산한다. '
        'trial 안의 이벤트 상관 중앙값을 구한 다음 trial에 같은 가중치를 주어 평균한다. '
        '양의 lag는 뒤 시각의 소음을 비교한다는 뜻이다. 상대적 시간 이동을 탐색하는 통계이며 원본 시계를 수정하지 않는다.', '',
        '|Driver|전체: lag=0 rho|동기화 후보: lag=0 rho|동기화 후보: 탐색 최대 rho / lag|',
        '|---|---:|---:|---:|',
    ]
    for driver in DRIVERS:
        values = []
        for subset in ('all_valid', 'sync_candidate'):
            local = lags[(lags.subset == subset) & lags.leg.eq('ALL') & lags.signal.eq(driver)]
            zero = local[local.lag_ms.eq(0)]
            values.append(f'{zero.mean_trial_median_rho.iloc[0]:+.3f}' if len(zero) else '없음')
        local = lags[(lags.subset == 'sync_candidate') & lags.leg.eq('ALL') & lags.signal.eq(driver)]
        local = local[np.isfinite(local.mean_trial_median_rho)]
        if len(local):
            best = local.loc[local.mean_trial_median_rho.idxmax()]
            best_text = f'{best.mean_trial_median_rho:+.3f} / {best.lag_ms:+.0f}ms'
        else:
            best_text = '없음'
        lines.append(f'|{driver}|{values[0]}|{values[1]}|{best_text}|')
    lines += [
        '', '**같은 상대 시각에서 착지 간 상관:** 예를 들어 모든 착지의 +10ms만 모아 힘/변화율과 소음의 상관을 계산한다. '
        '전체 Spearman과 trial 내부 순위에서 평균 순위를 제거한 상관을 각각 저장한다. '
        '이 그림은 단지 공통된 착지 파형이 있다는 이유로 생기는 상관과 이벤트 간 진폭 관계를 구분하기 위한 것이다.', '',
        '두 상관은 표본 축과 질문이 다르다. 이전 최대값 산점도 상관과 숫자 크기를 직접 비교해 개선율로 해석하지 않는다. '
        '착지 내부 관계가 뚜렷하더라도 이벤트 간 최대값 관계는 약할 수 있다.', '',
        '## 그림 읽는 순서', '',
        '1. `0.3_1_landing_examples.png`: 각 발에서 기록 중앙 시각에 가장 가까운 착지 하나를 소음과 무관하게 선택. '
        '0ms 점선, local 소음 최대 시점 점선, 다른 발 착지의 색 선을 함께 표시한다.',
        '2. `0.3_1_landing_heatmap.png`: 해당 trial의 모든 포함 이벤트를 시간순으로 표시. '
        '반복되는 시간 패턴과 이벤트별 차이를 볼 수 있다. 색 범위만 패널별 2~98분위로 제한하며 원본 숫자는 자르지 않았다.',
        '3. `trials/0.3_1_landing_profiles.png` 및 다른 trial 그림: 발별 중앙값·이벤트 IQR·최대 24개 시간순 파형.',
        '4. `landing_dynamics_overview.png`: trial별 이벤트 중앙값의 중앙값 및 trial IQR. '
        '전체와 동기화 후보는 trial 구성도 다르므로 차이를 동기화 효과만으로 보지 않는다.',
        '5. `landing_waveform_lag_correlations.png`: 한 착지 안의 시간적 동반 변화. 음영은 trial IQR.',
        '6. `landing_time_resolved_correlations.png`: 같은 상대 시각에서 착지 간 진폭 관계.', '',
        '## 해석 범위', '',
        '- 같은 마이크가 모든 발과 모터의 소리를 측정한다. 두 발의 근접 착지를 각각 정렬하면 같은 소리가 중복된다. '
        '`other_landings_in_extended_window`는 새 [−50,+100]ms 안의 다른 착지 수이며 기존 짧은 창의 isolated 플래그와 다르다.',
        '- 반복 이벤트와 겹치는 창은 독립 표본이 아니다. 독립 표본 p-value를 보고하지 않는다. '
        '시간 샘플도 RMS와 변화율 계산 창 때문에 서로 연관된다.',
        '- centered 5ms RMS는 ±2.5ms, 변화율은 ±6ms를 사용하므로 작은 선후관계를 흐릴 수 있다. '
        '기존 동기화 진단의 후보 통과도 하드웨어 수준 동기화 보증이 아니다.',
        '- local 최대 시점은 [−20,+60]ms 안에서만 검색했다. 경계에 걸리는 경우를 metrics CSV의 *_peak_at_search_edge에 표시한다. '
        'lag 탐색 최댓값은 탐색 편향이 있으며 실제 물리 지연이나 인과 효과로 해석하지 않는다.',
        '- 착지 후 접촉력·감속은 소음과 동시에 나타나는 반응일 수 있다. 이 분석은 시간적 연관성을 설명하며 '
        '소음의 가장 중요한 원인 또는 착지 전 예측 가능성을 입증하지 않는다.', '',
        '## 저장 파일과 재현', '',
        '- `landing_aligned_samples.csv`: 각 착지의 76개 시각과 모든 파형. 시간축·값을 유지한다.',
        '- `landing_event_metrics.csv`: 사용 이벤트, 원래 feature, 전후 변화·파형 상관·최대 시점·다른 발 정보.',
        '- `landing_event_coverage.csv`: 원래 이벤트 전부의 포함 여부와 제외 사유.',
        '- `landing_profiles.csv`, `landing_before_after_summary.csv`: 전체/발별/품질·고립 subset의 파형 및 전후 요약.',
        '- `landing_lag_correlations.csv`, `landing_lag_correlations_by_trial.csv`: 전체/발별·trial별 lag 곡선.',
        '- `landing_time_resolved_correlations.csv`: 고정 상대 시각의 착지 간 상관.',
        '- `landing_dynamics_metadata.json`: 입력 SHA256, 계산 정의, 선택 사례 ID 및 입력 무변경 검증.', '',
        '```bash', '.venv/bin/python noise_prediction_model/analysis/landing_event_dynamics.py',
        '.venv/bin/python -m unittest discover -s noise_prediction_model/analysis -p test_landing_event_dynamics.py -v', '```', '',
        '기존 전처리 CSV·접촉 이벤트 CSV·그림과 예측 모델은 덮어쓰지 않는다. '
        '원본 NPZ·개별 CSV·이벤트 CSV·URDF의 실행 전후 해시가 같아야 실행이 완료된다.', '',
    ]
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    interpretation = [
        '# 현재 데이터의 착지 전후 분석 해설', '',
        f'{events.trial.nunique()}개 train 기록의 {len(audit):,}개 착지 중 '
        f'{len(events):,}개를 분석했다. 착지마다 −50~+100ms의 76개 시각을 사용했다. '
        '새로 추가된 이동·회전 기록도 동일한 검출·계산 조건을 적용했다. Test 기록은 이 분석에 사용하지 않았다.', '',
        f'착지 후 [0,+40)ms 소음 RMS가 이전 [−50,−20)ms RMS보다 큰 이벤트는 '
        f'{100*events.noise_post40_minus_pre_db.gt(0).mean():.1f}%다. '
        '공유 마이크와 반복 착지 때문에 이를 독립 사건 확률이나 해당 발만의 충격음으로 해석하지 않는다.', '',
        '|전후 변화: trial별 중앙값의 중앙값|전체|동기화 후보|', '|---|---:|---:|',
    ]
    for metric in ('force_post_minus_pre', 'vz_post_minus_pre', 'noise_post40_minus_pre_db'):
        values = []
        for subset in ('all_valid', 'sync_candidate'):
            row = before_after[before_after.subset.eq(subset) & before_after.leg.eq('ALL') & before_after.metric.eq(metric)]
            values.append(f'{row.median_trial_median.iloc[0]:+.3f}' if len(row) else '없음')
        interpretation.append(f'|{descriptions[metric]}|{values[0]}|{values[1]}|')
    interpretation += ['', '|착지 내부 Spearman: lag=0, trial에 동일 가중치|전체|동기화 후보|', '|---|---:|---:|']
    for driver in DRIVERS:
        values = []
        for subset in ('all_valid', 'sync_candidate'):
            row = lags[lags.subset.eq(subset) & lags.leg.eq('ALL') & lags.signal.eq(driver) & lags.lag_ms.eq(0)]
            values.append(f'{row.mean_trial_median_rho.iloc[0]:+.3f}' if len(row) else '없음')
        interpretation.append(f'|{driver}|{values[0]}|{values[1]}|')
    isolated = int(events.other_landings_in_extended_window.eq(0).sum())
    interpretation += [
        '', 'Spearman은 각 착지 −20~+60ms의 41개 시각에서 driver(t)와 소음(t)의 순위를 비교한다. '
        '이벤트별 상관의 trial 중앙값을 구한 뒤 trial 사이에서 평균했다. '
        '착지마다 최대 힘이 클수록 소음도 큰지 묻는 이벤트 간 상관과는 다른 지표다. '
        '순간 변화율과 소음 레벨은 파형 폭이 달라 같은 시각 상관이 낮을 수 있으며, '
        '이 표만으로 가장 중요한 소음 원인을 결정하지 않는다.', '',
        f'확장 창 안에 다른 발 착지가 없는 이벤트는 {isolated:,}개이고, '
        f'나머지 {len(events)-isolated:,}개는 다른 착지를 포함한다. '
        '모터음·다른 발 소리·잔향이 섞인 녹음으로 발 하나의 소음 기여를 분리하지 못한다.', '',
        '`trials/spearman_summary.png`는 이벤트 하나당 점 하나이며, '
        'x는 해당 이벤트의 측정 평면 선속도 sqrt(vx²+vy²), y는 이벤트별 Spearman이다. '
        '측면 이동을 전진 속도 0으로 몰아 표시하지 않도록 평면 속도를 사용한다. '
        '제자리 회전 각속도는 이 선속도 축에서 표현되지 않는다. '
        '`noise_distributions/`의 y축은 같은 시각의 절대 5ms RMS dBFS다.', '',
        '발별 profiles 그림의 첫 행은 착지 이전 기준 RMS 대비 ΔdB이며 절대 dBFS가 아니다. '
        '검은 0ms 선은 force on=10이 6ms 지속되기 시작한 첫 교차다. '
        '실제 최초 접촉이나 정지 완료의 정답이 아니므로 그때 수직 속도가 0일 필요는 없다. '
        '12ms centered 변화율은 미래 6ms를 사용하므로 착지 전 실시간 예측 입력으로 그대로 넣지 않는다.', '',
        '힘 단위는 보정되지 않은 sensor units이고 소음은 dB SPL이 아닌 dBFS다. '
        '동기화 후보도 하드웨어 수준의 정확한 시각 일치를 보증하지 않는다. '
        '계산 정의와 제외 사유는 report.md, 원본 대조 검증은 verification.json, '
        '힘·속도 시각 검토는 force_velocity_sync_review/report.md에서 확인한다.', '',
    ]
    (output / 'interpretation.md').write_text('\n'.join(interpretation), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, default=ROOT / 'preprocessed/1007/train/contact_event_features.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'noise_prediction_model/results/analysis/landing_dynamics')
    parser.add_argument('--trials', nargs='+', help='Optional train trial subset')
    parser.add_argument('--no-trial-plots', action='store_true')
    args = parser.parse_args()
    source = pd.read_csv(args.events)
    if source.empty or not source.split.eq('train').all() or source.event_id.duplicated().any():
        raise ValueError('Expected unique train-only contact events')
    if not source.leg.isin(LEGS).all():
        raise ValueError('Unknown leg')
    if args.trials:
        missing = set(args.trials) - set(source.trial)
        if missing:
            raise ValueError(f'Unknown trials: {missing}')
        source = source[source.trial.isin(args.trials)].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'trials').mkdir(exist_ok=True)
    hashes = {str(args.events.resolve()): digest(args.events), str(MODEL_PATH): digest(MODEL_PATH)}
    all_events, all_arrays, all_audits, neighbors = [], [], [], {}
    for trial, group in source.groupby('trial', sort=True):
        csv = args.events.parent / f'{trial}.csv'
        raw = ROOT / f'data/1007/train/{trial}.npz'
        for path in (csv, raw):
            hashes[str(path.resolve())] = digest(path)
        events, arrays, audit, local_neighbors = extract_trial(pd.read_csv(csv), group.reset_index(drop=True), raw)
        if len(events):
            all_events.append(events); all_arrays.append(arrays)
        all_audits.append(audit); neighbors.update(local_neighbors)
        print(f'{trial}: {len(events)}/{len(group)} complete expanded windows', flush=True)
    if not all_events:
        raise ValueError('No complete expanded event windows')
    events = pd.concat(all_events, ignore_index=True)
    arrays = {name: np.concatenate([a[name] for a in all_arrays]) for name in all_arrays[0]}
    events = event_metrics(events, arrays)
    audit = pd.concat(all_audits, ignore_index=True)
    events.to_csv(args.output / 'landing_event_metrics.csv', index=False)
    audit.to_csv(args.output / 'landing_event_coverage.csv', index=False)
    write_samples(events, arrays, args.output / 'landing_aligned_samples.csv')
    profiles = profile_table(events, arrays)
    profiles.to_csv(args.output / 'landing_profiles.csv', index=False)
    print('Computing within-event lag associations...', flush=True)
    lags, local_lags = lag_tables(events, arrays)
    lags.to_csv(args.output / 'landing_lag_correlations.csv', index=False)
    local_lags.to_csv(args.output / 'landing_lag_correlations_by_trial.csv', index=False)
    times = time_resolved_table(events, arrays)
    times.to_csv(args.output / 'landing_time_resolved_correlations.csv', index=False)
    before_after = before_after_table(events)
    before_after.to_csv(args.output / 'landing_before_after_summary.csv', index=False)
    write_report(events, audit, before_after, lags, args.output)
    draw_profiles(events, arrays, args.output / 'landing_dynamics_overview.png')
    selected = draw_examples(events, arrays, neighbors, args.output / '0.3_1_landing_examples.png')
    draw_heatmap(events, arrays, args.output / '0.3_1_landing_heatmap.png')
    draw_associations(lags, times, args.output)
    if not args.no_trial_plots:
        for trial in events.trial.unique():
            print(f'Plotting {trial}...', flush=True)
            draw_profiles(events, arrays, args.output / f'trials/{trial}_landing_profiles.png', trial=trial)
    changed = [path for path, expected in hashes.items() if digest(path) != expected]
    if changed:
        raise AssertionError(f'Source changed during analysis: {changed}')
    metadata = dict(
        source_events=str(args.events.resolve()), input_sha256=hashes, all_sources_unchanged=True,
        script_sha256=digest(__file__), n_source_events=len(source), n_included_events=len(events),
        n_trials=int(events.trial.nunique()), relative_time_ms=REL_MS.tolist(), dt_ms=2,
        derivative='centered finite change [t-6,t+6]ms / 0.012s; retrospective, not a causal real-time feature',
        noise='DC-removed raw PCM, centered 5ms RMS; stored audio clock minus existing delay; no additional shift',
        baseline='exact PCM RMS [-50,-20)ms; may contain other noise, not assumed quiet',
        waveform_correlation='per-event Spearman on fixed driver [-20,+60]ms vs noise(t+lag); trial medians then equal-weight trial mean',
        lag_sign='positive: driver leads sampled noise; exploratory descriptive curve, not physical latency estimate',
        sample_spearman='at fixed relative time across events; pooled and demeaned within-trial ranks',
        subsets=SUBSETS, selected_example_event_ids=selected, no_audio_event_selection=True,
        trial_balanced_profile='median of trial medians; IQR across trial medians, not confidence interval',
    )
    (args.output / 'landing_dynamics_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(f'Saved {len(events)} landings / {len(events)*len(REL_MS)} samples -> {args.output}', flush=True)


if __name__ == '__main__':
    main()
