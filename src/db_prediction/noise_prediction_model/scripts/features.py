"""Build causal forecasts from an explicit preprocessed dataset; preserve recording splits."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from noise_prediction_model.scripts.lowstate_kinematics import (
    FEATURE_SCHEMA, VELOCITY_DESCRIPTION, LEGS, frame_lowstate, vertical_motion,
)

OUT = ROOT / 'noise_prediction_model'
SOURCE = ROOT / 'preprocessed/1007'
DT = .002
LAGS = (0, 10, 20, 40)
FAMILIES = ('force', 'velocity', 'force_change', 'velocity_change')
SEED = 20261027


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def continuous_segments(frame):
    times = frame.time_abs.to_numpy(float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Need finite, increasing times from one recording')
    breaks = np.r_[False, np.diff(times) > 1.5*DT + 1e-9]
    if 'segment' in frame:
        breaks |= np.r_[False, np.diff(frame.segment.to_numpy()) != 0]
    segments = np.cumsum(breaks)
    same = segments[1:] == segments[:-1]
    if not np.allclose(np.diff(times)[same], DT, atol=1e-8, rtol=0):
        raise ValueError('Expected the preprocessed 500Hz grid')
    return segments


def backward_rate(values, segments, steps=3):
    result = np.full_like(values, np.nan, dtype=float)
    ids = np.flatnonzero(segments[steps:] == segments[:-steps]) + steps
    result[ids] = (values[ids] - values[ids-steps]) / (steps*DT)
    return result


def physical_arrays(frame):
    segments = continuous_segments(frame)
    arrays = dict(force=frame[[f'ls_ff_{leg}' for leg in LEGS]].to_numpy(float),
                  velocity=vertical_motion(frame_lowstate(frame)))
    arrays.update(force_change=backward_rate(arrays['force'], segments),
                  velocity_change=backward_rate(arrays['velocity'], segments))
    return arrays, segments


def motion_mode(frame):
    columns = ['command_vx', 'command_vy', 'command_vyaw']
    if not all(c in frame for c in columns):
        return 'unknown'
    command = frame[columns].to_numpy()
    active = abs(command).max(axis=1) > .01
    if not active.any():
        return 'zero_command'
    shares = np.bincount(np.argmax(abs(command[active]), axis=1), minlength=3) / active.sum()
    return ('forward_backward', 'lateral', 'rotation')[int(shares.argmax())] if shares.max() >= .85 else 'mixed'


def causal_windows(frame, arrays, segments, trial, split, mode='unknown', events=None, with_targets=True):
    # The unused events argument preserves existing realtime feature-parity callers.
    names = [f'{family}_{leg}__lag{lag}ms' for lag in LAGS for family in FAMILIES for leg in LEGS]
    families = [family for lag in LAGS for family in FAMILIES for leg in LEGS]
    xs, ys, metadata = [], [], []
    for segment in np.unique(segments):
        pos = np.flatnonzero(segments == segment)
        # 46ms physical history; the target occupies [anchor+10, anchor+30)ms.
        anchors = np.arange(pos[0]+23, pos[-1]+1-(15 if with_targets else 0), 10)
        if not len(anchors):
            continue
        x = np.concatenate([arrays[family][anchors-lag//2]
                            for lag in LAGS for family in FAMILIES], axis=1)
        y = frame.audio_dbfs_20ms.to_numpy()[anchors+10] if with_targets else np.full(len(anchors), np.nan)
        good = np.isfinite(x).all(axis=1)
        if with_targets:
            good &= np.isfinite(y)
        anchors, x, y = anchors[good], x[good], y[good]
        t = frame.time_abs.to_numpy()[anchors]
        m = pd.DataFrame(dict(trial=trial, split=split, mode=mode,
                              segment=segments[anchors], anchor_time=t,
                              target_time=t+.020, target_support_start=t+.010,
                              target_support_stop=t+.030))
        xs.append(x)
        ys.append(y)
        metadata.append(m)
    if not xs:
        return np.empty((0, 64)), np.empty(0), pd.DataFrame(), names, families
    return np.concatenate(xs), np.concatenate(ys), pd.concat(metadata, ignore_index=True), names, families


def prepare(source=SOURCE, output=OUT):
    source, output = Path(source).resolve(), Path(output)
    plan = json.loads((source / 'split.json').read_text())
    train, test = plan['train_trials'], plan['test_trials']
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise ValueError('Duplicate or overlapping recording IDs')
    (output / 'datasets').mkdir(parents=True, exist_ok=True)
    chunks, metadata, coverage, hashes = [], [], [], {}
    middle_path = source / 'figure/middle_synchronization_check.csv'
    middle = {}
    if middle_path.exists():
        middle_table = pd.read_csv(middle_path)
        middle = middle_table.groupby('trial').passed.all().to_dict()
        hashes['figure/middle_synchronization_check.csv'] = digest(middle_path)
    hashes['split.json'] = digest(source / 'split.json')
    for split, trials in [('train', train), ('test', test)]:
        for trial in trials:
            path = source / split / f'{trial}.csv'
            hashes[f'{split}/{trial}.csv'] = digest(path)
            frame = pd.read_csv(path, float_precision='round_trip')
            arrays, segments = physical_arrays(frame)
            x, y, m, names, families = causal_windows(frame, arrays, segments, trial, split, motion_mode(frame))
            if not len(y):
                raise ValueError(f'No usable causal samples: {path}')
            sync_status = str(frame.synchronization_status.iloc[0]) if 'synchronization_status' in frame else 'UNREPORTED'
            sync_passed = bool(frame.synchronization_passed.iloc[0]) if 'synchronization_passed' in frame else None
            sync_group = ('unreported' if sync_passed is None else 'sync_review' if sync_passed is False else
                          'middle_review' if middle.get(trial) is False else
                          'all_checks_passed' if middle.get(trial) is True else 'start_end_passed')
            for column in ('command_vx', 'sp_vel_x'):
                if column in frame:
                    m[column] = np.interp(m.anchor_time, frame.time_abs, frame[column])
            m['synchronization_status'] = sync_status
            m['sync_group'] = sync_group
            chunks.append((x.astype(np.float32), y))
            metadata.append(m)
            coverage.append(dict(split=split, trial=trial, n_csv_rows=len(frame), n_samples=len(y), mode=motion_mode(frame),
                                 command_vx=float(frame.command_vx.median()) if 'command_vx' in frame else None,
                                 duration_s=len(frame)*DT, sync_group=sync_group, synchronization_status=sync_status))
            print(f'Features {split}/{trial}: {len(y)} samples', flush=True)
    x = np.concatenate([v[0] for v in chunks])
    y = np.concatenate([v[1] for v in chunks])
    meta = pd.concat(metadata, ignore_index=True)
    meta['sample_id'] = [f'causal_forecast:{i:07d}' for i in range(len(meta))]
    np.savez_compressed(output / 'datasets/causal_forecast.npz', x=x, y=y,
                        names=np.array(names), families=np.array(families))
    meta.to_csv(output / 'datasets/causal_forecast_metadata.csv', index=False, float_format='%.17g')
    pd.DataFrame(coverage).to_csv(output / 'datasets/coverage.csv', index=False)
    pd.DataFrame(dict(feature=names, family=families)).to_csv(output / 'datasets/feature_schema.csv', index=False)
    protocol = dict(task='causal_forecast', backend='pytorch', model_families=['ridge', 'hgb', 'sparse_gpr'],
                    training_backends=dict(ridge='pytorch', hgb='sklearn', sparse_gpr='pytorch'), source=str(source), seed=SEED,
                    train_trials=train, test_trials=test, n_features=64,
                    synchronization_policy='Include every trial listed in split.json, including synchronization review cases; no filtering by all_trials.csv',
                    selection='5-fold GroupKFold on train recordings only; equal-recording mean RMSE',
                    training_weights='equal sample weights for all three model families',
                    evaluation_order=['train-only CV', 'refit all train recordings', 'freeze all three models', 'evaluate test'],
                    causal=dict(lags_ms=LAGS, derivative_ms=6, sample_ms=20, horizon_ms=20,
                                target_support_relative_to_anchor_ms=[10, 30]),
                    feature_schema=FEATURE_SCHEMA, velocity=VELOCITY_DESCRIPTION,
                    force_units='uncalibrated LowState sensor units', target_units='dBFS, not calibrated dB SPL',
                    data_sources='Only split.json and recording CSVs under the source directory; no raw audio or external evaluation data')
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'datasets/source_manifest.json', dict(root=str(source), sha256=hashes))
    return dict(x=x, y=y, names=np.array(names), families=np.array(families), meta=meta), protocol


def load_dataset(output=OUT):
    data = dict(np.load(Path(output) / 'datasets/causal_forecast.npz', allow_pickle=False))
    data['meta'] = pd.read_csv(Path(output) / 'datasets/causal_forecast_metadata.csv', float_precision='round_trip')
    return data


if __name__ == '__main__':
    prepare()
