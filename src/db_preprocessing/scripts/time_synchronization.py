"""Per-sequence audio alignment from natural footfalls; end footfalls are held out.

All matching uses the raw PCM and raw LowState receive timestamps. This aligns
observable contact/sound events, not hardware capture clocks. No time warping.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.signal import find_peaks
from contact_detection import detect_contact

LEGS = ('fr', 'fl', 'rr', 'rl')


@dataclass(frozen=True)
class SyncConfig:
    calibration_seconds: float = 5.0
    end_seconds: float = 5.0
    settle_seconds: float = 2.0
    max_lag_s: float = .100
    group_seconds: float = .030
    confirm_ms: float = 6.0
    min_events: int = 4
    min_inlier_fraction: float = .6
    max_mad_s: float = .006
    tolerance_s: float = .010
    audio_bin_s: float = .001
    audio_reset_ms: float = 5.
    minimum_swing_s: float = .060
    consensus_radius_s: float = .006

    def __post_init__(self):
        positive = (self.calibration_seconds, self.end_seconds, self.max_lag_s,
                    self.group_seconds, self.tolerance_s, self.audio_bin_s, self.consensus_radius_s)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError('synchronization intervals must be positive and finite')
        if self.min_events < 3 or not 0 < self.min_inlier_fraction <= 1:
            raise ValueError('at least three events and a valid inlier fraction required')


def movement_bounds(trial):
    labels = trial.load('ev_label').astype(str)
    times = trial.load('ev_t').astype(float)
    values = []
    for label in ('move_start', 'move_stop'):
        found = times[labels == label]
        if len(found) != 1:
            raise ValueError(f'{trial.name}: exactly one {label} required')
        values.append(float(found[0]))
    if values[1] <= values[0]:
        raise ValueError('move_stop must follow move_start')
    return tuple(values)


def contact_groups(trial, config):
    """Thresholds use start-window force only; no audio/end data is fitted."""
    start, stop = movement_bounds(trial)
    time = np.asarray(trial.load('ls_t'), float)
    force = np.asarray(trial.load('ls_ff'), float)
    if force.shape != (len(time), 4) or np.any(np.diff(time) <= 0):
        raise ValueError('expected strictly increasing ls_t and four force channels')
    calibration = (time >= start) & (time <= min(start+config.calibration_seconds, stop))
    if calibration.sum() < 10:
        raise ValueError('insufficient start-window force samples')
    events, thresholds = [], {}
    for column, leg in enumerate(LEGS):
        values = force[calibration, column]
        low, high = np.nanpercentile(values, [10, 90])
        span = float(high-low)
        off, on = float(low + .07*span), float(low + .15*span)
        thresholds[leg] = dict(baseline=float(low), loaded=float(high), on=on, off=off,
                               usable=bool(span >= 10))
        if span < 10 or not np.isfinite(span):
            continue
        _, _, items = detect_contact(time, force[:, column], np.zeros(len(time)),
                                      on, off, config.confirm_ms, max_gap_s=.020)
        last_off = None
        for item in items:
            t = item['onset_time_abs']
            if item['event'] == 'contact_off':
                last_off = t
                continue
            if last_off is None or t-last_off < config.minimum_swing_s:
                continue
            if item['event'] == 'contact_on' and start <= t <= stop+config.settle_seconds:
                events.append(dict(sensor_time=float(t), leg=leg,
                                   confirmed_time=float(item['confirmed_time_abs'])))
    events.sort(key=lambda item: item['sensor_time'])
    groups = []
    for event in events:
        if not groups or event['sensor_time']-groups[-1]['sensor_time'] > config.group_seconds:
            groups.append(dict(sensor_time=event['sensor_time'], legs=[event['leg']], contacts=[event]))
        else:
            groups[-1]['legs'].append(event['leg'])
            groups[-1]['contacts'].append(event)
    # First threshold onset represents the shared acoustic event of close contacts.
    return groups, thresholds


def audio_onsets(trial, config):
    """First sustained acoustic rise after quiet; later ringing shares its onset."""
    raw = np.asarray(trial.load('audio'), dtype=float)
    slope, t0 = float(trial.load('audio_slope')), float(trial.load('audio_t0'))
    if raw.ndim != 1 or not len(raw) or slope <= 0 or not np.isfinite(raw).all():
        raise ValueError('invalid mono audio/clock')
    width = max(2, int(round(config.audio_bin_s/slope)))
    blocks = raw[:len(raw)//width*width].reshape(-1, width)
    rms = np.std(blocks, axis=1)
    # Three-bin local smoothing is for offline event detection, never model inputs.
    envelope = np.convolve(np.pad(rms, (1, 1), mode='edge'), np.ones(3)/3, mode='valid')
    time = t0+slope*(np.arange(len(rms))*width+(width+1)/2)
    start, stop = movement_bounds(trial)
    noise = envelope[(time >= start-1.5) & (time < start)]
    if len(noise) < 10:
        noise = envelope[time <= start+config.calibration_seconds]
    noise_mad = float(np.median(np.abs(noise-np.median(noise))))
    minimum_prominence = max(5., 4*noise_mad)
    step = width*slope
    peaks, properties = find_peaks(envelope, prominence=minimum_prominence,
                                   distance=max(1, round(.025/step)), wlen=max(3, round(.080/step)))
    baseline = float(np.median(noise))
    high = baseline+max(minimum_prominence, .5*baseline)
    low = baseline+max(3*noise_mad, .2*baseline, 2.)
    reset_bins = max(1, round(config.audio_reset_ms/1000/step))
    quiet = np.convolve((envelope <= low).astype(int), np.ones(reset_bins, dtype=int), mode='full')[:len(envelope)]
    quiet_ends = np.flatnonzero(quiet >= reset_bins)
    rows = []
    for i, peak in enumerate(peaks):
        prominence = float(properties['prominences'][i])
        previous = np.searchsorted(quiet_ends, peak, side='right')-1
        if previous < 0:
            continue
        left = int(quiet_ends[previous])+1
        if peak-left > round(.100/step):
            continue
        # Two consecutive bins above the start-noise threshold reject tiny spikes.
        rising = np.flatnonzero((envelope[left:peak] > high) & (envelope[left+1:peak+1] > high))
        if not len(rising):
            continue
        onset = left+int(rising[0])
        if not start-config.max_lag_s <= time[onset] <= stop+config.settle_seconds+config.max_lag_s:
            continue
        if rows and time[onset]-rows[-1]['audio_time'] < .008:
            if prominence <= rows[-1]['prominence']:
                continue
            rows.pop()
        rows.append(dict(audio_time=float(time[onset]), peak_time=float(time[peak]),
                         prominence=prominence, peak_rms=float(envelope[peak])))
    return rows, dict(time=time, envelope=envelope, minimum_prominence=minimum_prominence,
                      bin_seconds=step, onset_high=high, reset_low=low)


def match_groups(groups, onsets, config):
    """Keep plausible acoustic rises for sequence-level consensus, not peak picking."""
    matches = []
    for index, group in enumerate(groups):
        candidates = [dict(row, lag_s=row['audio_time']-group['sensor_time'])
                      for row in onsets if abs(row['audio_time']-group['sensor_time']) <= config.max_lag_s]
        candidates.sort(key=lambda row: row['prominence'], reverse=True)
        if candidates:
            candidates = [row for row in candidates if row['prominence'] >= .2*candidates[0]['prominence']]
        matches.append(dict(group, group_id=index, candidates=candidates, matched=False,
                            reason='NO_AUDIO_ONSET' if not candidates else 'NO_CONSENSUS',
                            audio_time=None, lag_s=None))
    return matches


def select_consensus(events, config):
    """Each footfall has one vote. Competing lag modes are explicitly rejected.

    This avoids locking onto a loud late resonance instead of the acoustic onset.
    Calibration and validation call this independently, with the same settings.
    """
    grid = np.arange(-config.max_lag_s, config.max_lag_s+.0005, .001)
    votes = np.zeros(len(grid))
    strengths = np.zeros(len(grid))
    for item in events:
        if not item['candidates']:
            continue
        lags = np.array([c['lag_s'] for c in item['candidates']])
        prominence = np.array([c['prominence'] for c in item['candidates']])
        near = abs(grid[:, None]-lags) <= config.consensus_radius_s
        votes += near.any(axis=1)
        strengths += np.max(near*(prominence/prominence.max()), axis=1)
    if not votes.any():
        return False
    best = int(np.argmax(votes+.01*strengths))
    competing = abs(grid-grid[best]) > 2*config.tolerance_s
    ambiguous = bool(np.any(votes[competing] >= .9*votes[best]))
    if ambiguous:
        for item in events:
            if item['candidates']:
                item['reason'] = 'COMPETING_LAG_MODES'
        return True
    for item in events:
        near = [c for c in item['candidates'] if abs(c['lag_s']-grid[best]) <= config.consensus_radius_s+1e-9]
        if near:
            chosen = min(near, key=lambda c: abs(c['lag_s']-grid[best]))
            item.update(audio_time=chosen['audio_time'], lag_s=chosen['lag_s'],
                        prominence=chosen['prominence'], matched=True, reason='MATCHED')
    owners = {}
    for item in events:
        if item['matched']:
            owners.setdefault(item['audio_time'], []).append(item)
    for items in owners.values():
        if len(items) > 1:
            for item in items:
                item.update(matched=False, reason='SHARED_AUDIO_CANDIDATE')
    return False


def summarize_window(events, config):
    matched = [item for item in events if item['matched']]
    lags = np.array([item['lag_s'] for item in matched], float)
    result = dict(events=len(events), matched=len(matched), inliers=0, median_s=None,
                  mad_s=None, usable=False, inlier_group_ids=[])
    if not len(lags):
        return result
    median = float(np.median(lags))
    inliers = np.abs(lags-median) <= config.tolerance_s+1e-9
    selected = lags[inliers]
    if not len(selected):
        return result
    median = float(np.median(selected))
    mad = float(np.median(np.abs(selected-median)))
    count = int(inliers.sum())
    result.update(inliers=count, median_s=median, mad_s=mad,
                  inlier_group_ids=[item['group_id'] for item, keep in zip(matched, inliers) if keep],
                  usable=bool(count >= config.min_events and
                              count >= config.min_inlier_fraction*len(events) and
                              mad <= config.max_mad_s+1e-9))
    return result


def estimate_footfall_delay(trial, config=None):
    config = config or SyncConfig()
    start, stop = movement_bounds(trial)
    if stop-start <= config.calibration_seconds+config.end_seconds:
        raise ValueError('start calibration and end validation windows must be disjoint')
    groups, thresholds = contact_groups(trial, config)
    onsets, audio = audio_onsets(trial, config)
    # Last window follows the actual last detected contact, including settling.
    last = groups[-1]['sensor_time'] if groups else stop
    for group in groups:
        group['window'] = ('start' if group['sensor_time'] <= start+config.calibration_seconds
                           else 'end' if group['sensor_time'] >= max(stop-config.end_seconds,
                                                                 last-config.end_seconds)
                           else 'middle')
    events = match_groups([g for g in groups if g['window'] != 'middle'], onsets, config)
    start_events = [e for e in events if e['window'] == 'start']
    end_events = [e for e in events if e['window'] == 'end']
    start_ambiguous = select_consensus(start_events, config)
    end_ambiguous = select_consensus(end_events, config)
    calibration = summarize_window(start_events, config)
    validation = summarize_window(end_events, config)
    calibration['ambiguous'] = start_ambiguous
    validation['ambiguous'] = end_ambiguous
    delay = calibration['median_s'] if calibration['usable'] else 0.
    residual = validation['median_s']-delay if validation['median_s'] is not None else None
    end_inliers = [e for e in events if e['group_id'] in validation['inlier_group_ids']]
    errors = [abs(e['lag_s']-delay) for e in end_inliers]
    p90 = float(np.quantile(errors, .9)) if errors else None
    if not calibration['usable']:
        status = 'CALIBRATION_REVIEW'
    elif not validation['usable']:
        status = 'INSUFFICIENT_END_EVENTS'
    elif abs(residual) > config.tolerance_s+1e-9 or p90 > config.tolerance_s+1e-9:
        status = 'END_RESIDUAL_REVIEW'
    else:
        status = 'CONSISTENT_CANDIDATE'
    for item in events:
        item['applied_residual_ms'] = (item['lag_s']-delay)*1000 if item['lag_s'] is not None else None
        item['used_for_calibration'] = item['group_id'] in calibration['inlier_group_ids'] and calibration['usable']
        item['validation_inlier'] = item['group_id'] in validation['inlier_group_ids']
    summary = dict(trial=trial.name, status=status, passed=status == 'CONSISTENT_CANDIDATE',
                   applied_audio_delay_s=float(delay), correction_accepted=calibration['usable'],
                   calibration=calibration, validation=validation,
                   end_residual_ms=residual*1000 if residual is not None else None,
                   end_abs_error_p90_ms=p90*1000 if p90 is not None else None,
                   contact_thresholds=thresholds, total_contact_groups=len(groups),
                   first_contact_match=events[0] if events else None,
                   last_contact_match=events[-1] if events else None,
                   first_contact_time=groups[0]['sensor_time'] if groups else None,
                   last_contact_time=groups[-1]['sensor_time'] if groups else None,
                   move_start=start, move_stop=stop, config=asdict(config),
                   policy='per-sequence start footfalls only; end held out; constant offset; no drift fit',
                   timing_basis='PC receive clock; force-threshold onset versus acoustic rise; not hardware sync')
    return summary, events, audio


def apply_audio_delay(data, delay_s):
    """Positive delay moves audio earlier; never changes robot timestamps."""
    result = dict(data)
    time = data['time_abs']
    sample_at = time+delay_s
    for key in ('audio_rms_20ms', 'audio_dbfs_20ms', 'audio_peak_2ms'):
        result[key] = np.interp(sample_at, time, data[key], left=np.nan, right=np.nan)
    shifted = np.interp(sample_at, time, data['audio_valid'].astype(float), left=0., right=0.) >= .999
    result['audio_valid'] = shifted
    robot_valid = np.ones(len(time), dtype=bool)
    for key in data['robot_keys']:
        robot_valid &= data[f'{key}_valid']
    result['valid'] = robot_valid & shifted
    result['audio_delay_s'] = np.array(delay_s)
    return result
