"""Real/sim footfall phase alignment inspired by db_preprocessing synchronization.

Reuses its force hysteresis detector. Fit one constant offset using start-window
events; hold the end window out. No per-joint shift, clock warp or amplitude fit.
This aligns separate gait sequences, NOT hardware capture clocks.
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / 'src/db_preprocessing/scripts'
sys.path.insert(0, str(REFERENCE))
from contact_detection import detect_contact

LEGS = ('FR', 'FL', 'RR', 'RL')


def contacts(t, signal, valid, start, stop):
    calibration = valid & (t >= start) & (t <= stop)
    if calibration.sum() < 10:
        raise ValueError('Too few calibration samples')
    events, thresholds = [], []
    for foot, leg in enumerate(LEGS):
        low, high = np.percentile(signal[calibration, foot], [10, 90])
        span = high - low
        thresholds.append(dict(foot=leg, low=float(low), high=float(high),
                               on=float(low + .15 * span), off=float(low + .07 * span)))
        if span < 1e-6:
            continue
        values = np.asarray(signal[:, foot], dtype=float).copy()
        values[~valid] = np.nan
        _, _, detected = detect_contact(t, values, np.zeros(len(t)),
            on=max(0., low + .15 * span), off=max(0., low + .07 * span),
            confirm_ms=6., max_gap_s=.010)
        last_off = None
        for item in detected:
            stamp = item['onset_time_abs']
            if item['event'] == 'contact_off':
                last_off = stamp
            elif last_off is not None and stamp - last_off >= .060:
                events.append(dict(foot=leg, time_s=float(stamp),
                                   confirmed_s=float(item['confirmed_time_abs'])))
    return sorted(events, key=lambda e: e['time_s']), thresholds


def match_window(real, sim, window, max_lag=.280, radius=.025, tolerance=.025):
    selected = [e for e in real if window[0] <= e['time_s'] <= window[1]]
    grid = np.arange(-max_lag, max_lag + .0005, .001)
    candidates = []
    votes = np.zeros(len(grid))
    for item in selected:
        possible = [dict(sim_time_s=s['time_s'], lag_s=s['time_s'] - item['time_s'])
                    for s in sim if s['foot'] == item['foot']
                    and abs(s['time_s'] - item['time_s']) <= max_lag]
        candidates.append(dict(item, candidates=possible))
        if possible:
            lags = np.array([c['lag_s'] for c in possible])
            votes += np.any(abs(grid[:, None] - lags) <= radius + 1e-9, axis=1)
    if not votes.any():
        return dict(window_s=list(window), events=len(selected), matched=0, usable=False,
                    median_s=None, mad_s=None, ambiguous=False), []
    best = np.flatnonzero(votes == votes.max())
    mode = float(np.median(grid[best]))
    competing = abs(grid - mode) > 2 * tolerance
    ambiguous = bool(np.any(votes[competing] >= .9 * votes.max()))
    matches = []
    owners = set()
    for item in candidates:
        near = [c for c in item['candidates'] if abs(c['lag_s'] - mode) <= radius + 1e-9]
        if not near:
            continue
        chosen = min(near, key=lambda c: abs(c['lag_s'] - mode))
        owner = (item['foot'], chosen['sim_time_s'])
        if owner in owners:
            continue
        owners.add(owner)
        matches.append(dict(foot=item['foot'], real_time_s=item['time_s'], **chosen))
    lags = np.array([e['lag_s'] for e in matches])
    median = float(np.median(lags)) if len(lags) else None
    mad = float(np.median(abs(lags - median))) if len(lags) else None
    usable = (len(lags) >= 8 and len(lags) >= .6 * len(selected)
              and mad <= .015 + 1e-9 and not ambiguous)
    return dict(window_s=list(window), events=len(selected), matched=len(matches),
                usable=bool(usable), median_s=median, mad_s=mad, ambiguous=ambiguous), matches


def estimate_alignment(arrays, move_start, move_stop):
    # Same start-calibration/end-validation principle as time_synchronization.py.
    calibration = (move_start + 2., move_start + 7.)
    validation = (move_stop - 6., move_stop - 1.)
    if calibration[1] >= validation[0]:
        raise ValueError('Calibration and held-out end windows must not overlap')
    real, real_thresholds = contacts(arrays['t'], arrays['real_foot_sensor_raw'], arrays['valid'], *calibration)
    sim, sim_thresholds = contacts(arrays['t'], arrays['sim_foot_fz_N'],
                                   np.ones(len(arrays['t']), bool), *calibration)
    cal, start_matches = match_window(real, sim, calibration)
    end, end_matches = match_window(real, sim, validation)
    # Keep a clearly marked provisional offset when event spread fails the
    # precision check. Do not claim hardware sync or validation in that case.
    has_candidate = cal['matched'] >= 8 and not cal['ambiguous'] and cal['median_s'] is not None
    lag = cal['median_s'] if has_candidate else 0.
    residuals = np.array([item['lag_s'] - lag for item in end_matches])
    residual = float(np.median(residuals)) if len(residuals) else None
    p90 = float(np.percentile(abs(residuals), 90)) if len(residuals) else None
    passed = bool(cal['usable'] and end['usable'] and abs(residual) <= .025 + 1e-9 and p90 <= .025 + 1e-9)
    summary = dict(
        status='CONSISTENT_PHASE_ALIGNMENT' if passed else 'REVIEW_REQUIRED',
        calibration=cal, validation=end, calibration_accepted=cal['usable'],
        provisional=bool(has_candidate and not passed),
        applied_lag_s=float(lag), held_out_residual_median_s=residual,
        held_out_abs_residual_p90_s=p90, passed=passed,
        real_thresholds=real_thresholds, sim_thresholds=sim_thresholds,
        reference_scripts=['db_preprocessing/scripts/time_synchronization.py',
                           'db_preprocessing/scripts/contact_detection.py'],
        method='same-foot onset consensus; start-only constant phase fit; end held out; no time warp',
        sign='lag = sim event time - real event time; aligned sim(t) = original sim(t + lag)',
        interpretation='Observed gait-event phase offset, NOT sensor/actuator latency.',
        config=dict(max_lag_s=.280, consensus_radius_s=.025, max_mad_s=.015,
                    end_tolerance_s=.025, min_events=8, min_fraction=.6,
                    confirm_ms=6., minimum_swing_s=.060))
    events = [dict(item, window=window, residual_s=float(item['lag_s'] - lag))
              for window, matches in (('calibration', start_matches), ('validation', end_matches))
              for item in matches]
    return summary, events


def apply_alignment(arrays, lag):
    """Translate ALL sim channels equally. Crop overlap; never extrapolate."""
    t = arrays['t']
    query = t + lag
    keep = (query >= t[0]) & (query <= t[-1])
    result = {key: value[keep].copy() for key, value in arrays.items()}
    for key, value in arrays.items():
        if key.startswith('sim_'):
            result[key + '_unaligned'] = value[keep].copy()
            flat = value.reshape(len(t), -1)
            sampled = np.stack([np.interp(query[keep], t, flat[:, i]) for i in range(flat.shape[1])], axis=1)
            result[key] = sampled.reshape((int(keep.sum()),) + value.shape[1:])
    return result


def plot_alignment(before, after, summary, events, path):
    import matplotlib.pyplot as plt
    start = summary['calibration']['window_s'][0] + 1
    stop = start + 2.4
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    for col, data in enumerate((before, after)):
        t = data['t']; selected = (t >= start) & (t <= stop)
        for foot, leg in enumerate(LEGS):
            x, y = data['real_foot_sensor_raw'][:, foot], data['sim_foot_fz_N'][:, foot]
            mask = data['steady']
            x = (x - x[mask].mean()) / x[mask].std()
            y = (y - y[mask].mean()) / y[mask].std()
            axes[0, col].plot(t[selected], x[selected] + foot * 4, color='C0', lw=.9,
                              label='Real sensor' if foot == 0 else None)
            axes[0, col].plot(t[selected], y[selected] + foot * 4, color='C1', lw=.9,
                              label='Simulation' if foot == 0 else None)
        axes[0, col].set_yticks(np.arange(4) * 4, LEGS)
        axes[0, col].set_title('Before: record clock only' if col == 0 else 'After: one common phase shift')
        axes[0, col].set_ylabel('Foot force shape (z-score + offset)')
        axes[0, col].legend(fontsize=8)
        for row, window in enumerate(('calibration', 'validation'), start=1):
            subset = [e for e in events if e['window'] == window]
            for foot, leg in enumerate(LEGS):
                selected_events = [e for e in subset if e['foot'] == leg]
                values = [(e['lag_s'] if col == 0 else e['residual_s']) * 1000 for e in selected_events]
                axes[row, col].scatter([e['real_time_s'] for e in selected_events], values, label=leg, s=15)
            axes[row, col].axhline(0, color='black', lw=.7)
            axes[row, col].set_ylabel(f'{window}: event offset [ms]')
            if row == 2 and col == 1:
                axes[row, col].axhspan(-25, 25, color='green', alpha=.1)
            axes[row, col].legend(ncol=4, fontsize=8)
        for ax in axes[:, col]: ax.grid(alpha=.2); ax.set_xlabel('Real record time [s]')
    fig.suptitle(f'Common phase shift {summary["applied_lag_s"] * 1000:+.1f} ms | '
                 f'{summary["status"]}\nStart footfalls fitted; end footfalls held out. This is NOT hardware clock synchronization.')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
