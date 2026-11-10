#!/usr/bin/env python3
"""Audit force/velocity timing and its effect on noise associations.

Read-only with respect to raw data, preprocessing, previous analyses and models.
Both splits are audited. Any proposed shift is estimated from OTHER train
trials' force/velocity timing, never from audio, and is diagnostic only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/low_noise_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr

import noise_prediction_model.analysis.landing_event_dynamics as ld
from noise_prediction_model.analysis.review_landing_signal_alignment import rate, plateau_peak

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'noise_prediction_model/results/analysis/landing_dynamics'
OUT = BASE / 'alignment_impact_review'
sys.path.insert(0, str(ROOT / 'preprocessing'))
sys.path.insert(0, str(ROOT / 'comparison'))
from go2_kinematics import Go2Kinematics, LEGS as SDK_LEGS
from force_estimation import polynomial_filter

REL = ld.REL_MS
CORE = np.flatnonzero((REL >= -20) & (REL <= 60))
DRIVERS = ld.DRIVERS
COLORS = dict(zip(ld.LEGS, ('tab:blue', 'tab:orange', 'tab:green', 'tab:purple')))


def peak_time(values, lo, hi, minimum=False):
    """First and last tied extrema midpoint, with explicit ambiguity flags."""
    ids = np.flatnonzero((REL >= lo) & (REL <= hi))
    x = values[:, ids]
    extreme = np.min(x, axis=1) if minimum else np.max(x, axis=1)
    eq = np.isclose(x, extreme[:, None], rtol=1e-9, atol=1e-9)
    first = np.min(np.where(eq, REL[ids], np.inf), axis=1)
    last = np.max(np.where(eq, REL[ids], -np.inf), axis=1)
    return (first + last) / 2, last - first, eq[:, 0] | eq[:, -1]


def waveform_rho(x, noise, shifts_ms=0):
    """rho[x(t+s), noise(t)] on the SAME 41 time points for every shift."""
    shifts = np.broadcast_to(shifts_ms, len(x))
    steps = np.rint(shifts / 2).astype(int)
    if not np.allclose(steps * 2, shifts):
        raise ValueError('Shifts must be on the 2ms grid')
    ids = CORE[None, :] + steps[:, None]
    if ids.min() < 0 or ids.max() >= len(REL):
        raise ValueError('Shift outside the exported support')
    xx = np.take_along_axis(x, ids, axis=1)
    return ld.row_correlation(rankdata(xx, axis=1), rankdata(noise[:, CORE], axis=1))


def heldout_shifts(events):
    """Equal-trial median timing by leg; exclude evaluated train trial.

    delta = t(dv/dt peak)-t(dF/dt peak). To align velocity to force,
    sample velocity(t+delta). Positive sample shift advances a trace.
    """
    valid = ~events.rate_peak_at_edge & (events.rate_peak_tie_width_ms <= 6)
    calibration = events[valid & events.split.eq('train')].groupby(
        ['trial', 'leg'], as_index=False).rate_delta_ms.median()
    rows, shifts = [], np.zeros(len(events), float)
    for (split, trial, leg), idx in events.groupby(['split', 'trial', 'leg']).groups.items():
        source = calibration[calibration.leg.eq(leg)]
        if split == 'train':
            source = source[~source.trial.eq(trial)]
        value = float(np.median(source.rate_delta_ms)) if len(source) else 0.
        shift = float(np.clip(np.rint(value / 2) * 2, -20, 20))
        shifts[np.asarray(idx)] = shift
        rows.append(dict(split=split, trial=trial, leg=leg, n_calibration_trials=len(source),
                         timing_delta_ms=value, velocity_sample_shift_ms=shift,
                         force_sample_shift_ms=-shift, calibration_uses_audio=False))
    return shifts, pd.DataFrame(rows)


def trial_diagnostics(frame, accepted, arrays, raw_path, model):
    """Source mapping, velocity decomposition and alternative estimators."""
    t = frame.time_abs.to_numpy(float)
    rotation = Rotation.from_quat(frame[[f'ls_quaternion_{a}' for a in 'wxyz']].to_numpy()[:, [1, 2, 3, 0]])
    body = rotation.apply(frame[[f'sp_vel_{a}' for a in 'xyz']].to_numpy())[:, 2]
    gyro = frame[[f'ls_gyro_{a}' for a in 'xyz']].to_numpy()
    with np.load(raw_path, allow_pickle=False) as raw:
        lt, st = raw['ls_t'], raw['sp_t']
        li = np.searchsorted(lt, t, side='right') - 1
        si = np.searchsorted(st, t, side='right') - 1
        if min(li.min(), si.min()) < 0:
            raise AssertionError('Missing native source')
        for key in ('q', 'dq', 'gyro', 'quaternion', 'ff'):
            suffix = ([f'{leg}_{j}' for leg in SDK_LEGS for j in ('hip', 'thigh', 'calf')]
                      if key in ('q', 'dq') else (list('xyz') if key == 'gyro' else
                      (list('wxyz') if key == 'quaternion' else list(SDK_LEGS))))
            actual = frame[[f'ls_{key}_{s}' for s in suffix]].to_numpy()
            np.testing.assert_allclose(actual, raw[f'ls_{key}'][li], atol=2e-10, rtol=0)
        np.testing.assert_allclose(frame[[f'sp_vel_{a}' for a in 'xyz']], raw['sp_vel'][si], atol=2e-10, rtol=0)
        # Future Sport sample is allowed ONLY for the interpolation diagnostic.
        sport_linear = np.column_stack([np.interp(t, st, raw['sp_vel'][:, j]) for j in range(3)])
        body_linear = rotation.apply(sport_linear)[:, 2]
        # Independent calculation from the same encoders, NOT ground truth.
        _, qdot_fit = polynomial_filter(lt, raw['ls_q'], half_samples=3, max_gap_s=.010)
        qdot_fit = qdot_fit[li]
    ls_age, sp_age = (t-lt[li])*1000, (t-st[si])*1000
    clock = dict(split=raw_path.parent.name, trial=raw_path.stem, n_csv_rows=len(frame),
                 mapping_verified=True, ls_age_median_ms=np.median(ls_age), ls_age_max_ms=max(ls_age),
                 sp_age_median_ms=np.median(sp_age), sp_age_max_ms=max(sp_age),
                 ls_sp_source_difference_median_ms=np.median(abs(lt[li]-st[si])*1000),
                 native_ls_dt_median_ms=np.median(np.diff(lt))*1000,
                 native_sp_dt_median_ms=np.median(np.diff(st))*1000)
    extras = {name: np.full_like(arrays['vz'], np.nan) for name in
              ('body_vz', 'rotation_vz', 'joint_vz', 'vz_sport_linear', 'vz_qdot_fit')}
    centers = np.searchsorted(t, accepted.contact_time_abs.to_numpy()-1e-8) if len(accepted) else []
    if len(accepted):
        np.testing.assert_allclose(t[centers], accepted.contact_time_abs, atol=1e-7, rtol=0)
    indices = np.asarray(centers, int)[:, None] + (REL/2).astype(int)[None, :]
    for leg_index, leg in enumerate(SDK_LEGS):
        q = frame[[f'ls_q_{leg}_{j}' for j in ('hip', 'thigh', 'calf')]].to_numpy()
        dq = frame[[f'ls_dq_{leg}_{j}' for j in ('hip', 'thigh', 'calf')]].to_numpy()
        p, jac = model.position_jacobian(leg, q)
        joint = rotation.apply(np.einsum('nij,nj->ni', jac, dq))[:, 2]
        angular = rotation.apply(np.cross(gyro, p))[:, 2]
        good_qdot = np.isfinite(qdot_fit[:, 3*leg_index:3*leg_index+3]).all(axis=1)
        joint_qdot = np.full(len(t), np.nan)
        relative = np.einsum('nij,nj->ni', jac[good_qdot], qdot_fit[good_qdot, 3*leg_index:3*leg_index+3])
        joint_qdot[good_qdot] = rotation[good_qdot].apply(relative)[:, 2]
        use = accepted.leg.eq(leg.upper()).to_numpy() if len(accepted) else np.zeros(0, bool)
        for name, values in dict(body_vz=body, rotation_vz=angular, joint_vz=joint,
                                 vz_sport_linear=body_linear+angular+joint,
                                 vz_qdot_fit=body+angular+joint_qdot).items():
            extras[name][use] = values[indices[use]]
    if len(accepted):
        np.testing.assert_allclose(extras['body_vz']+extras['rotation_vz']+extras['joint_vz'],
                                   arrays['vz'], atol=1e-12)
    return extras, clock


def timing_metrics(events, a):
    events = events.copy()
    ftime, fwidth, fedge = peak_time(a['force'], 0, 40)
    vtime, vwidth, vedge = peak_time(a['vz'], -20, 20, minimum=True)
    events['force_peak_ms'] = ftime
    events['vz_minimum_ms'] = vtime
    events['force_peak_minus_vz_minimum_ms'] = ftime-vtime
    events['magnitude_peak_at_edge'] = fedge | vedge
    events['magnitude_peak_tie_width_ms'] = np.maximum(fwidth, vwidth)
    fr = rate(a['force'], 3); vr = rate(a['vz'], 3)
    ft, first, last, _, fe, _ = plateau_peak(fr)
    vt, vf, vl, _, ve, _ = plateau_peak(vr)
    events['force_rate_peak_ms'] = ft
    events['vz_rate_peak_ms'] = vt
    events['rate_delta_ms'] = vt-ft
    events['rate_peak_at_edge'] = fe | ve
    events['rate_peak_tie_width_ms'] = np.maximum(last-first, vl-vf)
    near = (REL >= -20) & (REL <= 20)
    late = (REL >= 60) & (REL <= 100)
    events['vz_at_on_mps'] = a['vz'][:, REL == 0].ravel()
    events['late_vz_median_mps'] = np.median(a['vz'][:, late], axis=1)
    for key in ('body_vz', 'rotation_vz', 'joint_vz'):
        events[f'{key}_at_on_mps'] = a[key][:, REL == 0].ravel()
        events[f'{key}_prepost_change_mps'] = a[key][:, REL == 12].ravel()-a[key][:, REL == -12].ravel()
    events['vz_at_minus2_mps'] = a['vz'][:, REL == -2].ravel()
    before = (REL >= -20) & (REL < 0)
    events['downward_minus2_mps'] = np.maximum(-events.vz_at_minus2_mps, 0)
    events['downward_pre20_max_mps'] = np.maximum(-np.min(a['vz'][:, before], axis=1), 0)
    events['downward_pre20_mean_mps'] = np.maximum(-np.mean(a['vz'][:, before], axis=1), 0)
    events['force_peak_0_40'] = np.max(a['force'][:, (REL >= 0) & (REL <= 40)], axis=1)
    events['force_rate_peak_value'] = np.max(fr[:, near], axis=1)
    events['vz_rate_peak_value'] = np.max(vr[:, near], axis=1)
    events['measured_planar_speed_mps'] = np.nan  # Filled from existing measured body data below.
    variant_rows = []
    for span in (4, 8, 12, 20):
        ft = plateau_peak(rate(a['force'], span//4))[0]
        vt = plateau_peak(rate(a['vz'], span//4))[0]
        variant_rows.append(pd.DataFrame(dict(event_id=events.event_id, split=events.split,
            trial=events.trial, leg=events.leg, variant=f'difference_{span}ms', rate_delta_ms=vt-ft)))
    for key in ('vz_sport_linear', 'vz_qdot_fit'):
        values = a[key]
        valid = np.isfinite(values).all(axis=1)
        delta = np.full(len(events), np.nan)
        delta[valid] = plateau_peak(rate(values[valid], 3))[0]-events.force_rate_peak_ms.to_numpy()[valid]
        events[f'{key}_rate_delta_ms'] = delta
        events[f'{key}_mae_mps'] = np.nanmean(abs(values-a['vz']), axis=1)
        variant_rows.append(pd.DataFrame(dict(event_id=events.event_id, split=events.split,
            trial=events.trial, leg=events.leg, variant=key, rate_delta_ms=delta)))
    return events, pd.concat(variant_rows, ignore_index=True)


def summarize_timing(e, keys):
    return e.groupby(keys, as_index=False).agg(n_events=('event_id', 'size'),
        rate_delta_median_ms=('rate_delta_ms', 'median'),
        rate_delta_q25_ms=('rate_delta_ms', lambda x:x.quantile(.25)),
        rate_delta_q75_ms=('rate_delta_ms', lambda x:x.quantile(.75)),
        fraction_abs_delta_gt4ms=('rate_delta_ms', lambda x:x.abs().gt(4).mean()),
        fraction_abs_delta_gt6ms=('rate_delta_ms', lambda x:x.abs().gt(6).mean()),
        fraction_velocity_earlier_gt2ms=('rate_delta_ms', lambda x:x.lt(-2).mean()),
        fraction_velocity_later_gt2ms=('rate_delta_ms', lambda x:x.gt(2).mean()),
        magnitude_gap_median_ms=('force_peak_minus_vz_minimum_ms', 'median'),
        magnitude_gap_q25_ms=('force_peak_minus_vz_minimum_ms', lambda x:x.quantile(.25)),
        magnitude_gap_q75_ms=('force_peak_minus_vz_minimum_ms', lambda x:x.quantile(.75)),
        n_rate_edge=('rate_peak_at_edge', 'sum'),
        late_vz_median_mps=('late_vz_median_mps', 'median'))


def correlations(events, a):
    fixed, paired, amplitude, strata = [], [], [], []
    shifts, shift_table = heldout_shifts(events)
    for audio_key in ('noise_5ms_dbfs', 'noise_20ms_dbfs'):
        for driver in DRIVERS:
            orig = waveform_rho(a[driver], a[audio_key])
            signed_shift = shifts if driver in ('vz', 'vz_rate') else -shifts
            shifted = waveform_rho(a[driver], a[audio_key], signed_shift)
            tmp = events[['event_id', 'split', 'trial', 'leg', 'rate_delta_ms',
                          'synchronization_status', 'other_landings_in_extended_window']].copy()
            tmp['audio'] = audio_key; tmp['driver'] = driver
            tmp['sample_shift_ms'] = signed_shift
            tmp['rho_original'] = orig; tmp['rho_timing_shift'] = shifted
            tmp['delta_rho'] = shifted-orig
            paired.append(tmp)
            for shift in (-20, -10, -6, -4, -2, 0, 2, 4, 6, 10, 20):
                r = waveform_rho(a[driver], a[audio_key], shift)
                ss = events[['split', 'trial']].copy()
                ss['rho'] = r; ss['abs_delta_rho'] = abs(r-orig)
                for (split, trial), g in ss.groupby(['split', 'trial']):
                    fixed.append(dict(split=split, trial=trial, driver=driver, audio=audio_key,
                                      sample_shift_ms=shift, n_events=len(g),
                                      median_rho=g.rho.median(), median_abs_delta_rho=g.abs_delta_rho.median()))
            for (split, trial), ids in events.groupby(['split', 'trial']).groups.items():
                ids = np.asarray(ids)
                for label, mask in [('within4ms', abs(events.rate_delta_ms.to_numpy()[ids]) <= 4),
                                    ('over4ms', abs(events.rate_delta_ms.to_numpy()[ids]) > 4)]:
                    group = ids[mask]
                    if len(group) >= 5:
                        strata.append(dict(split=split, trial=trial, driver=driver, audio=audio_key,
                                           timing_group=label, n_events=len(group), median_rho=np.nanmedian(orig[group])))
    for (split, trial, leg), g in events.groupby(['split', 'trial', 'leg']):
        for feature in ('force_peak_0_40', 'force_rate_peak_value', 'downward_minus2_mps',
                        'downward_pre20_mean_mps', 'downward_pre20_max_mps', 'vz_rate_peak_value'):
            good = np.isfinite(g[feature]) & np.isfinite(g.noise_post40_rms_dbfs)
            x, y = g.loc[good, feature], g.loc[good, 'noise_post40_rms_dbfs']
            rho = spearmanr(x, y).statistic if len(x) >= 10 and x.nunique() > 1 and y.nunique() > 1 else np.nan
            amplitude.append(dict(split=split, trial=trial, leg=leg, feature=feature,
                                  n_events=int(good.sum()), spearman_rho=rho))
    return (pd.concat(paired, ignore_index=True), pd.DataFrame(fixed),
            pd.DataFrame(amplitude), pd.DataFrame(strata), shift_table)


def plot_trial(events, a, trial, split, output):
    ids_trial = np.flatnonzero(events.trial.eq(trial) & events.split.eq(split))
    fig, axes = plt.subplots(5, 4, figsize=(17, 14), layout='constrained')
    for col, leg in enumerate(ld.LEGS):
        ids = ids_trial[events.iloc[ids_trial].leg.eq(leg)]
        color = COLORS[leg]
        axes[0, col].set_title(f'{leg}: {len(ids)} events')
        if not len(ids):
            for ax in axes[:, col]:ax.text(.5, .5, 'No accepted events', transform=ax.transAxes, ha='center')
            continue
        settings = [('noise_5ms_dbfs', '5ms RMS (dBFS)', 'black'),
                    ('force', 'Foot force (sensor units)', color),
                    ('vz', 'Foot vertical velocity (m/s)', 'crimson')]
        for row, (key, ylabel, c) in enumerate(settings):
            q = np.percentile(a[key][ids], [25, 50, 75], axis=0)
            for wave in a[key][ids][::max(1, len(ids)//25)]:
                axes[row, col].plot(REL, wave, color=c, alpha=.08, lw=.5)
            axes[row, col].fill_between(REL, q[0], q[2], color=c, alpha=.2)
            axes[row, col].plot(REL, q[1], color=c, lw=1.5)
            if col == 0:axes[row, col].set_ylabel(ylabel)
        for key, c in [('force_rate', 'tab:orange'), ('vz_rate', 'tab:purple')]:
            waves = a[key][ids]
            scale = np.max(abs(waves[:, (REL >= -12) & (REL <= 20)]), axis=1)
            q = np.percentile(waves / np.maximum(scale[:, None], 1e-12), [25, 50, 75], axis=0)
            axes[3, col].plot(REL, q[1], color=c, label=key)
            axes[3, col].fill_between(REL, q[0], q[2], color=c, alpha=.12)
        axes[3, col].legend(fontsize=7)
        if col == 0:axes[3, col].set_ylabel('12ms rates / own absolute maximum')
        for row in range(4):
            axes[row, col].axvline(0, color='black', ls='--', lw=.8)
            axes[row, col].axhline(0, color='.7', lw=.5)
            axes[row, col].set_xlim(-30, 80); axes[row, col].grid(alpha=.15)
        axes[3, col].set_xlabel('Time from force threshold onset (ms)')
        local = events.iloc[ids]
        x = local.contact_time_from_move_start
        axes[4, col].scatter(x, local.force_peak_minus_vz_minimum_ms, s=15, color='.55', label='Force peak - velocity minimum')
        axes[4, col].scatter(x, local.rate_delta_ms, s=17, color='tab:purple', label='Velocity-rate - force-rate peak')
        axes[4, col].axhspan(-4, 4, color='green', alpha=.08)
        axes[4, col].axhline(0, color='black', lw=.6)
        axes[4, col].set(xlabel='Time from move start (s)', ylabel='Peak-time difference (ms)' if col == 0 else '')
        axes[4, col].grid(alpha=.15)
        axes[4, col].text(.03, .97, f'Rate median: {local.rate_delta_ms.median():+.1f}ms\n'
                         f'|rate delta|>4ms: {local.rate_delta_ms.abs().gt(4).mean():.0%}\n'
                         f'Magnitude gap: {local.force_peak_minus_vz_minimum_ms.median():+.1f}ms',
                         transform=axes[4, col].transAxes, va='top', fontsize=8)
    axes[4, 0].legend(fontsize=6, loc='lower right')
    fig.suptitle(f'{split}/{trial}: magnitude peaks and abrupt changes answer different timing questions\n'
                 'Upper rows: individual traces, median and event IQR. Bottom: every event in recording order.\n'
                 '0ms = original force threshold crossing; no source data shifted.', fontsize=12)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{trial}_timing_and_repetition.png', dpi=140)
    plt.close(fig)


def overview_plots(e, a, by_leg, paired, fixed):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for split, color in [('train', 'tab:blue'), ('test', 'tab:orange')]:
        g = e[e.split.eq(split)]
        axes[0, 0].hist(g.force_peak_minus_vz_minimum_ms, bins=np.arange(-22, 64, 2),
                        alpha=.5, color=color, label=f'{split}: n={len(g)}')
        axes[0, 1].hist(g.rate_delta_ms, bins=np.arange(-34, 36, 2), alpha=.5, color=color, label=split)
    axes[0, 0].set(xlabel='Force magnitude peak - v_z minimum (ms)', ylabel='Events', title='Different physical quantities')
    axes[0, 1].set(xlabel='dv_z/dt peak - dF/dt peak (ms)', ylabel='Events', title='Abrupt-change timing, matching 12ms operators')
    order = list(e.drop_duplicates(['split', 'trial']).itertuples())
    matrix = np.full((len(order), 4), np.nan)
    for row, r in enumerate(order):
        for col, leg in enumerate(ld.LEGS):
            g = by_leg[by_leg.split.eq(r.split) & by_leg.trial.eq(r.trial) & by_leg.leg.eq(leg)]
            if len(g):matrix[row, col] = g.fraction_abs_delta_gt4ms.iloc[0]*100
    im = axes[1, 0].imshow(matrix, aspect='auto', cmap='viridis', vmin=0, vmax=100)
    axes[1, 0].set(xticks=range(4), xticklabels=ld.LEGS, yticks=range(len(order)),
                   yticklabels=[f'{r.split}/{r.trial}' for r in order], title='Repetition: fraction of events with |rate delta| >4ms')
    axes[1, 0].tick_params(axis='y', labelsize=6)
    fig.colorbar(im, ax=axes[1, 0], label='% of events')
    p = paired[paired.audio.eq('noise_5ms_dbfs')].groupby(['split', 'trial', 'driver'], as_index=False)[
        ['rho_original', 'rho_timing_shift']].median()
    for driver, c in zip(DRIVERS, ('tab:blue', 'crimson', 'tab:orange', 'tab:purple')):
        g = p[p.driver.eq(driver)]
        axes[1, 1].scatter(g.rho_original, g.rho_timing_shift, s=18, color=c, alpha=.65, label=driver)
    axes[1, 1].plot([-1, 1], [-1, 1], color='.5', ls='--')
    axes[1, 1].set(xlabel='Original within-event rho (trial median)',
                   ylabel='rho after OTHER-train timing-based shift', title='No audio used to choose shifts', xlim=(-1,1), ylim=(-1,1))
    for ax in (axes[0, 0], axes[0, 1]):ax.axvline(0, color='black', lw=.8);ax.legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.savefig(OUT/'alignment_overview.png', dpi=160);plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained', sharex=True)
    for ax, driver in zip(axes.flat, DRIVERS):
        for audio, ls in [('noise_5ms_dbfs', '-'), ('noise_20ms_dbfs', '--')]:
            g = fixed[fixed.driver.eq(driver) & fixed.audio.eq(audio) & fixed.split.eq('train')]
            med = g.groupby('sample_shift_ms').median_rho.median()
            ax.plot(med.index, med, marker='o', ls=ls, label=audio)
        ax.axvline(0, color='black', lw=.8);ax.set(title=driver, xlabel='Signal sample shift s: rho[x(t+s), noise(t)] (ms)',
                                                ylabel='Median across trial median rhos');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Fixed shift sensitivity: exploratory, no shift selected by noise correlation')
    fig.savefig(OUT/'noise_correlation_shift_sensitivity.png', dpi=165);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    sources = [ROOT/'preprocessed/1007'/s/'contact_event_features.csv' for s in ('train', 'test')]
    protected = sources + sorted((ROOT/'data/1007').glob('*/*.npz')) + [
        p for p in (ROOT/'preprocessed/1007').glob('*/*.csv') if p.stem not in
        ('all_trials', 'quality', 'contact_event_features', 'contact_event_samples')]
    protected += list((ROOT/'noise_prediction_model/models/feature_selection').glob('*'))
    protected += [BASE/'landing_event_metrics.csv', BASE/'landing_aligned_samples.csv']
    hashes = {str(p.relative_to(ROOT)):ld.digest(p) for p in protected if p.is_file()}
    all_events = pd.concat([pd.read_csv(p) for p in sources], ignore_index=True)
    model = Go2Kinematics()
    chunks, wave_chunks, coverage, audits, clocks = [], {}, [], [], []
    for raw in sorted((ROOT/'data/1007').glob('*/*.npz')):
        split, trial = raw.parent.name, raw.stem
        frame = pd.read_csv(ROOT/f'preprocessed/1007/{split}/{trial}.csv', float_precision='round_trip')
        selected = all_events[all_events.split.eq(split) & all_events.trial.eq(trial)]
        if len(selected):
            accepted, arrays, audit, _ = ld.extract_trial(frame, selected, raw)
            audits.append(audit.assign(split=split))
        else:
            accepted = selected.copy()
            arrays = {k:np.empty((0,len(REL))) for k in (*ld.SIGNALS, 'noise_5ms_dbfs', 'noise_20ms_dbfs')}
        extra, clock = trial_diagnostics(frame, accepted, arrays, raw, model)
        clocks.append(clock); arrays.update(extra)
        # Measured planar speed in the same core interval; never use command speed.
        if len(accepted):
            t = frame.time_abs.to_numpy()
            centers = np.searchsorted(t, accepted.contact_time_abs.to_numpy()-1e-8)
            ii = centers[:,None]+(REL[CORE]/2).astype(int)[None,:]
            speed = np.hypot(frame.sp_vel_x.to_numpy(), frame.sp_vel_y.to_numpy())
            accepted = accepted.copy();accepted['audit_measured_speed_mps'] = np.median(speed[ii], axis=1)
            chunks.append(accepted)
            for key, values in arrays.items():wave_chunks.setdefault(key, []).append(values)
        coverage.append(dict(split=split, trial=trial, n_detected_events=len(selected),
                             n_included_events=len(accepted), n_excluded_events=len(selected)-len(accepted)))
        print(f'Extracted {split}/{trial}: {len(accepted)}/{len(selected)}; raw mapping verified', flush=True)
    events = pd.concat(chunks, ignore_index=True)
    a = {k:np.concatenate(v) for k,v in wave_chunks.items()}
    events, variants = timing_metrics(events, a)
    events['measured_planar_speed_mps'] = events.audit_measured_speed_mps
    paired, fixed, amplitude, strata, shifts = correlations(events, a)
    by_leg = summarize_timing(events, ['split', 'trial', 'leg'])
    by_trial = summarize_timing(events, ['split', 'trial'])
    pd.DataFrame(coverage).to_csv(OUT/'trial_coverage.csv', index=False)
    pd.concat(audits).to_csv(OUT/'event_inclusion_audit.csv', index=False)
    pd.DataFrame(clocks).to_csv(OUT/'source_clock_checks.csv', index=False)
    events.to_csv(OUT/'event_timing_and_features.csv', index=False)
    by_leg.to_csv(OUT/'timing_by_trial_leg.csv', index=False)
    by_trial.to_csv(OUT/'timing_by_trial.csv', index=False)
    variants.to_csv(OUT/'timing_estimator_sensitivity.csv', index=False)
    paired.to_csv(OUT/'noise_correlations_paired_by_event.csv', index=False)
    fixed.to_csv(OUT/'fixed_shift_correlations_by_trial.csv', index=False)
    amplitude.to_csv(OUT/'amplitude_noise_correlations_by_trial_leg.csv', index=False)
    strata.to_csv(OUT/'noise_correlations_by_timing_group.csv', index=False)
    shifts.to_csv(OUT/'other_train_timing_shifts.csv', index=False)
    np.savez_compressed(OUT/'audited_waveforms.npz', event_id=events.event_id.to_numpy(str), relative_time_ms=REL, **a)
    paired_trial = paired.groupby(['split', 'trial', 'driver', 'audio'], as_index=False).agg(
        n_events=('event_id', 'size'),rho_original=('rho_original', 'median'),
        rho_timing_shift=('rho_timing_shift', 'median'), median_paired_delta_rho=('delta_rho', 'median'),
        median_abs_paired_delta_rho=('delta_rho', lambda x:x.abs().median()),
        fraction_abs_paired_delta_gt0p05=('delta_rho', lambda x:x.abs().gt(.05).mean()))
    paired_trial.to_csv(OUT/'noise_correlations_paired_by_trial.csv', index=False)
    for (split,trial), _ in events.groupby(['split','trial'], sort=False):
        plot_trial(events,a,trial,split,OUT/'trials'/split)
        print(f'Plotted {split}/{trial}', flush=True)
    overview_plots(events,a,by_leg,paired,fixed)
    # Exact consistency with the previously published train event identities and
    # waveforms. Their derived event summaries are intentionally not overwritten.
    prior = pd.read_csv(BASE/'landing_event_metrics.csv')
    old = prior.set_index('event_id')
    train = events[events.split.eq('train')].set_index('event_id')
    if set(old.index) != set(train.index):raise AssertionError('Train event coverage changed')
    for field in ('force_post_peak_sensor_units','pre_velocity_world_z_mps','noise_post40_rms_dbfs'):
        np.testing.assert_allclose(train.loc[old.index,field],old[field],atol=1e-7,rtol=0)
    for path, value in hashes.items():
        if ld.digest(ROOT/path) != value:raise AssertionError(f'Protected input changed: {path}')
    summary = dict(n_raw_trials=len(coverage),n_event_trials=int(events.groupby(['split','trial']).ngroups),
                   n_events=len(events),train_events=int(events.split.eq('train').sum()),
                   test_events=int(events.split.eq('test').sum()), n_protected_files=len(hashes),
                   raw_source_mapping_verified_for_all_csv_rows=True,
                   previous_train_event_identities_and_values_verified=True,
                   physical_contact_truth_available=False, original_signals_shifted=False,
                   estimator_shift_uses_audio=False,script_sha256=ld.digest(__file__), input_sha256=hashes)
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},indent=2))
    print('Saved',OUT,flush=True)


if __name__ == '__main__':main()
