"""One validated waveform phase shift per trial, shared by every sim channel."""
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

JOINT_INDICES = np.array([1, 2, 4, 5, 7, 8, 10, 11])
LAGS = np.arange(-50, 51) * .005


def contact_window_scores(data, mask, window, lags=LAGS):
    """Independent positive-load evidence; require all four feet to agree."""
    t = data['t']
    indices = np.flatnonzero(mask & (t >= window[0]) & (t <= window[1]))
    invalid = dict(window_s=list(window), usable=False)
    if len(indices) < 150:
        return invalid, None
    real, sim = data['real_foot_sensor_raw'], data['sim_foot_fz_N']
    if np.any(sim < -1e-6):
        return invalid, None
    a, b = real[indices], sim[indices]
    if np.any(a.std(0) < 1.) or np.any(b.std(0) < 1.):
        return invalid, None
    x = a - a.mean(0)
    curves = []
    for step in np.rint(lags / .005).astype(int):
        y = sim[indices + step]
        y = y - y.mean(0)
        scale = np.sqrt((x * x).sum(0) * (y * y).sum(0))
        curves.append(np.divide((x * y).sum(0), scale, out=np.zeros(4), where=scale > 1e-12))
    curves = np.asarray(curves)
    scores = curves.mean(1)
    best = int(np.argmax(scores))
    alternatives = abs(lags - lags[best]) >= .080
    margin = float(scores[best] - scores[alternatives].max())
    per_foot_lag = lags[np.argmax(curves, axis=0)]
    return dict(window_s=list(window), usable=True, n=len(indices), lag_s=float(lags[best]),
        correlation_before=float(scores[np.argmin(abs(lags))]), correlation_best=float(scores[best]),
        boundary=best in (0, len(lags) - 1), ambiguous=margin < .10, peak_margin=margin,
        per_foot_lag_s=per_foot_lag.tolist(), per_foot_correlation=curves[best].tolist()), curves


def contact_supported_fallback(data, mask, windows, joint_records, joint_scores, lags=LAGS):
    """Recover weak joint-shape matches only with strong, held-out contact evidence."""
    records, scores = zip(*(contact_window_scores(data, mask, window, lags) for window in windows))
    if not all(record['usable'] for record in records):
        return None
    calibration, validation = records
    candidate = calibration['lag_s']
    index = int(np.argmin(abs(lags - candidate)))
    if any(record['boundary'] or record['ambiguous'] for record in records):
        return None
    if abs(validation['lag_s'] - candidate) > .020 + 1e-9:
        return None
    for record, curve, joint, qcurve in zip(records, scores, joint_records, joint_scores):
        fixed = curve[index]
        if record['correlation_best'] < .85 or float(fixed.mean()) < .85:
            return None
        if float(fixed.min()) < .75 or np.ptp(record['per_foot_lag_s']) > .020 + 1e-9:
            return None
        # Contacts corroborate a joint timing estimate; they cannot replace a
        # conflicting joint clock, a boundary solution or an ambiguous period.
        if abs(joint['lag_s'] - candidate) > .020 + 1e-9 or float(qcurve[index]) < .50:
            return None
        if float(fixed.mean()) - record['correlation_before'] < .02:
            return None
        record['joint_lag_s'] = joint['lag_s']
        record['joint_correlation_at_contact_lag'] = float(qcurve[index])
        record['per_foot_correlation_at_calibration_lag'] = fixed.tolist()
    validation['correlation_at_calibration_lag'] = float(scores[1][index].mean())
    return calibration, validation


def window_scores(data, mask, window, lags=LAGS):
    t = data['t']
    indices = np.flatnonzero(mask & (t >= window[0]) & (t <= window[1]))
    if len(indices) < 150:
        return dict(window_s=list(window), usable=False, reason='insufficient_motion', n=len(indices)), None
    features, channels = [], []
    for quantity, minimum_std in (('q', .01), ('dq', .15)):
        real = data['real_' + quantity][indices][:, JOINT_INDICES]
        sim = data['sim_' + quantity][indices][:, JOINT_INDICES]
        active = (real.std(0) >= minimum_std) & (sim.std(0) >= minimum_std)
        if active.sum() < 4:
            return dict(window_s=list(window), usable=False, reason='insufficient_motion', n=len(indices)), None
        selected = JOINT_INDICES[active]
        channels += [f'{quantity}:{int(j)}' for j in selected]
        x = real[:, active] - real[:, active].mean(0)
        curve = []
        for step in np.rint(lags / .005).astype(int):
            y = data['sim_' + quantity][indices + step][:, selected]
            y = y - y.mean(0)
            denominator = np.sqrt((x * x).sum(0) * (y * y).sum(0))
            curve.append(np.divide((x * y).sum(0), denominator,
                out=np.zeros(len(selected)), where=denominator > 1e-12))
        features.append(np.asarray(curve))
    # Equal q/dq weighting, equal weighting of active joints within each quantity.
    scores = np.mean([feature.mean(1) for feature in features], axis=0)
    best = int(np.argmax(scores))
    alternatives = abs(lags - lags[best]) >= .080
    margin = float(scores[best] - scores[alternatives].max()) if alternatives.any() else 1.
    result = dict(window_s=list(window), usable=True, n=len(indices),
        lag_s=float(lags[best]), correlation_before=float(scores[np.argmin(abs(lags))]),
        correlation_best=float(scores[best]), boundary=best in (0, len(lags) - 1),
        peak_margin=margin, ambiguous=margin < .02, channels=channels)
    return result, scores


def _estimate_recorded_alignment(data, lags=LAGS, anchor_validation_cycle=False):
    t = data['t']
    if len(t) < 250 or not np.allclose(np.diff(t), .005, atol=1e-8):
        raise ValueError('Alignment requires a common 200 Hz grid')
    # Every candidate uses identical samples, with no interpolation over gaps.
    margin = int(np.rint(max(abs(lags)) / .005))
    safe = minimum_filter1d(data['valid'].astype(int), size=2 * margin + 1, mode='constant') > 0
    motion = []
    for domain in ('real', 'sim'):
        energy = uniform_filter1d(np.mean(data[domain + '_dq'][:, JOINT_INDICES] ** 2, axis=1), size=41)
        motion.append(energy > .15 ** 2)
    mask = safe & data['steady'] & motion[0] & motion[1]
    times = t[mask]
    start, stop = (float(times[0]), float(times[-1])) if len(times) else (float(t[0]), float(t[-1]))
    width = min(5., (stop - start) * .4)
    windows = [(start, start + width), (stop - width, stop)]
    calibration, cal_scores = window_scores(data, mask, windows[0], lags)
    validation, val_scores = window_scores(data, mask, windows[1], lags)
    if anchor_validation_cycle and calibration['usable'] and validation['usable']:
        # Only the early window chooses the gait cycle. A periodic late window
        # may have an equally good peak one full cycle away; validate the early
        # cycle's local residual without silently switching to that other cycle.
        nearby = np.flatnonzero(abs(lags - calibration['lag_s']) <= .080 + 1e-9)
        local_best = int(nearby[np.argmax(val_scores[nearby])])
        validation.update(cycle_anchored=True, local_search_radius_s=.080,
            unrestricted_lag_s=validation['lag_s'],
            unrestricted_correlation_best=validation['correlation_best'],
            lag_s=float(lags[local_best]), correlation_best=float(val_scores[local_best]),
            boundary=local_best in (0, len(lags) - 1))
    reasons = []
    for label, record in (('calibration', calibration), ('validation', validation)):
        if not record['usable']:
            reasons.append(label + ':insufficient_motion')
        else:
            if record['boundary']: reasons.append(label + ':search_boundary')
            if record['ambiguous'] and not record.get('cycle_anchored'):
                reasons.append(label + ':ambiguous_period')
            if record['correlation_best'] < .70: reasons.append(label + ':low_correlation')
    candidate = calibration.get('lag_s', 0.)
    residual = None
    if calibration['usable'] and validation['usable']:
        residual = validation['lag_s'] - candidate
        validation['correlation_at_calibration_lag'] = float(val_scores[np.argmin(abs(lags - candidate))])
        if validation.get('cycle_anchored') and (
                validation['unrestricted_correlation_best'] - validation['correlation_at_calibration_lag'] > .02):
            reasons.append('validation:better_cycle')
        if abs(residual) > .020 + 1e-9: reasons.append('validation:lag_changed')
        if validation['correlation_at_calibration_lag'] < .70: reasons.append('validation:low_fixed_correlation')
        # A meaningful nonzero move must help the independent late window too.
        for label, record in (('calibration', calibration), ('validation', validation)):
            aligned = record.get('correlation_at_calibration_lag', record['correlation_best'])
            if abs(candidate) > .005 and aligned - record['correlation_before'] < .02:
                reasons.append(label + ':no_improvement')
    reference = 'joint_q_dq'
    joint_rejections = list(reasons)
    if reasons and all(reason.split(':')[1] in ('low_correlation', 'low_fixed_correlation') for reason in reasons):
        if all(record['usable'] for record in (calibration, validation)) and all(
                key in data for key in ('real_foot_sensor_raw', 'sim_foot_fz_N')):
            fallback = contact_supported_fallback(data, mask, windows,
                (calibration, validation), (cal_scores, val_scores), lags)
            if fallback is not None:
                calibration, validation = fallback
                candidate = calibration['lag_s']
                residual = validation['lag_s'] - candidate
                reference = 'contact_load_with_joint_consensus'
                reasons = []
    passed = not reasons
    applied = candidate if passed else 0.
    status = 'VALIDATED_CONTACT_ALIGNMENT' if passed and reference == 'contact_load_with_joint_consensus' else (
        'VALIDATED_WAVEFORM_ALIGNMENT' if passed else 'CLOCK_ONLY_REVIEW_REQUIRED')
    return dict(status=status, reference=reference, joint_only_rejection_reasons=joint_rejections,
        passed=passed, applied_lag_s=float(applied), estimated_lag_s=float(candidate),
        calibration=calibration, validation=validation, rejection_reasons=reasons,
        held_out_lag_difference_s=None if residual is None else float(residual),
        held_out_residual_median_s=None, held_out_abs_residual_p90_s=None,
        phase_validation_performed=True, provisional=False,
        method=('Early four-foot load correlation corroborated by joint timing; independent late validation'
                if reference == 'contact_load_with_joint_consensus' else
                'Early q/dq thigh/calf joint correlation; one common shift; independent late validation'),
        sign='lag = sim event time - real event time; aligned sim(t) = original sim(t + lag)',
        interpretation='Waveform phase alignment includes policy phase and dynamics; not measured hardware latency.',
        config=dict(max_lag_s=float(max(abs(lags))), step_s=.005, min_samples=150, min_correlation=.70,
                    max_held_out_lag_difference_s=.020, min_peak_margin=.02,
                    joint_indices=JOINT_INDICES.tolist(), same_shift_all_channels=True,
                    contact_support=dict(min_mean_correlation=.85, min_foot_correlation=.75, min_peak_margin=.10,
                        max_foot_lag_spread_s=.020, min_joint_correlation=.50,
                        max_joint_contact_lag_difference_s=.020)))


def estimate_recorded_alignment(data):
    result = _estimate_recorded_alignment(data)
    allowed = {'search_boundary', 'low_correlation', 'low_fixed_correlation', 'no_improvement'}
    reasons = result['rejection_reasons']
    # Preserve every previously accepted shift. Expand once only when the
    # original search was truncated, never to bypass ambiguity or timing drift.
    if (not result['passed'] and any(reason.endswith(':search_boundary') for reason in reasons)
            and all(reason.split(':', 1)[1] in allowed for reason in reasons)):
        expanded = _estimate_recorded_alignment(data, np.arange(-100, 101) * .005,
            anchor_validation_cycle=True)
        expanded['search_expansion'] = dict(initial_max_lag_s=.25, expanded_max_lag_s=.5,
            initial_rejection_reasons=reasons,
            cycle_selection='Early unique q/dq peak selects cycle; late fixed shift and local residual validate it',
            pure_hardware_delay_identified=False)
        expanded['method'] = ('Early q/dq cycle selection within +/-500ms after +/-250ms boundary; '
            'late fixed-candidate correlation and local residual validation; same shift for all sim channels')
        return expanded
    return result


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
