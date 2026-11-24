#!/usr/bin/env python3
"""Analyze paired logs, publish one result bundle, then export that saved bundle."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'real2sim'

from .config import PROJECT, DEFAULT_DATASET, JOINTS, LEGS, RESULT_SCHEMA, UI_SCHEMA, result_path, validate_dataset
from .preprocessing import load, prepare_pair, named_mapping, paired_paths, sha
from .time_alignment import estimate_recorded_alignment, apply_alignment
from .deployment_mapping import load_contract, apply_contract
from .metrics import metric, rolling_rmse, evaluate_view
from .reporting import write_report
from .export_ui import export_ui
from .storage import publish_directory


def estimate_legacy_alignment(*args):
    from phase_synchronization import estimate_alignment
    return estimate_alignment(*args)


def map_pair(raw, rm, sim, sm, policy_dir=None):
    raw, sim, contract = named_mapping(raw, rm, sim, sm)
    data, timing = prepare_pair(raw, sim, 15., .005)
    if rm.get('shared_pc_clock') and sm.get('shared_pc_clock'):
        sync, events = estimate_recorded_alignment(data), []
    else:
        sync, events = estimate_legacy_alignment(data, timing['move_start_s'], timing['move_stop_s'])
    lag = sync['applied_lag_s']
    aligned = apply_alignment(data, lag)
    # Both views use the exact same overlap, valid mask, and evaluation samples.
    query = data['t'] + lag
    keep = (query >= data['t'][0]) & (query <= data['t'][-1])
    before = {key: value.copy() for key, value in aligned.items() if not key.endswith('_unaligned')}
    shifted_valid = np.interp(query[keep], data['t'], data['valid'].astype(float)) > .999
    valid = before['valid'] & shifted_valid
    steady = before['steady'] & valid
    if steady.sum() < 100:
        raise ValueError('Insufficient common steady-walking data')
    for record in (before,):
        record['valid'], record['steady'] = valid.copy(), steady.copy()
        record['sensor_counts'] = record['real_foot_sensor_raw']
        record['sim_fz_N'] = record['sim_foot_fz_N']
    # Compare time alignment itself in physical units, on exactly the same samples.
    comparison = dict(units=dict(q='rad', dq='rad/s', ddq='rad/s²', tau='N·m'),
        scope='same recording-interior samples in physical units; clock only vs applied common shift', quantities={})
    for quantity in ('q', 'dq', 'ddq', 'tau'):
        real = before['real_' + quantity]
        original = data['sim_' + quantity][keep]
        before['sim_' + quantity + '_clock_only'] = original.copy()
        shifted = before['sim_' + quantity]
        comparison['quantities'][quantity] = dict(
            clockOnly=[metric(real[:, j], original[:, j], steady) for j in range(12)],
            aligned=[metric(real[:, j], shifted[:, j], steady) for j in range(12)])
    sync['physicalComparison'] = comparison
    before['sim_foot_fz_N_clock_only'] = data['sim_foot_fz_N'][keep].copy()
    checkpoint = rm.get('checkpoint_sha256')
    if checkpoint is None:
        # These collector logs omit policy hashes. Use the documented deployment
        # contract, but never claim the recording itself proves policy identity.
        checkpoint = json.loads((Path(policy_dir) / 'policy.json' if policy_dir else
            Path(__file__).with_name('deployment_normalization.json')).read_text())['checkpoint_sha256']
    deployment = load_contract(checkpoint, policy_dir)
    deployment['provenance']['recording_policy_hash_verified'] = bool(
        rm.get('checkpoint_sha256') == sm.get('checkpoint_sha256') == checkpoint)
    deployment['provenance']['recording_policy_source'] = (
        'recording metadata' if deployment['provenance']['recording_policy_hash_verified']
        else 'data/explain_data.txt baseline policy; collector NPZ metadata has no policy hash')
    after, coordinates, roundtrip = apply_contract(before, deployment, raw, sim)
    contract['time_mapping'] = sync['sign']
    contract['time_alignment_in_both_views'] = True
    contract['deployment'] = dict(**deployment['provenance'],**roundtrip)
    return before, after, timing, sync, events, contract, coordinates


def _analyze(data_dir, directory, policy_dir, dataset, plots):
    for name in ('processed', 'metrics', 'figures'):
        (directory / name).mkdir(parents=True)
    manifest = dict(schema=UI_SCHEMA, resultSchema=RESULT_SCHEMA, status='building',
        dataset=dataset, joints=list(JOINTS), feet=list(LEGS),
        units=dict(q='rad', dq='rad/s', ddq='rad/s²', tau='N·m', realForce='counts', simForce='N'),
        trials=[], excludedTrials=[], generatedAt=datetime.now(timezone.utc).isoformat(),
        settings=dict(filter_hz=15., dt_s=.005, plots=plots))
    results = []
    for real_path, sim_path in paired_paths(data_dir, dataset):
        name = real_path.stem.removesuffix('_real')
        raw, rm = load(real_path)
        sim, sm = load(sim_path)
        if not len(raw['ls_t']) or not len(sim['ls_t']):
            reason = 'Empty real or sim ls_t: no state samples to compare'
            manifest['excludedTrials'].append(dict(id=name, realPath=str(real_path.resolve()),
                simPath=str(sim_path.resolve()), reason=reason))
            print(f'{name}: EXCLUDED ({reason})', flush=True)
            continue
        if sm.get('completed') is False or rm.get('checkpoint_sha256') != sm.get('checkpoint_sha256'):
            raise ValueError('Invalid simulation/reference pair')
        before, after, timing, sync, events, contract, coordinates = map_pair(raw, rm, sim, sm, policy_dir)
        rolling = {}
        first = evaluate_view(before, 'before', 'Existing real/sim logs; time aligned baseline',
            before['steady'], rolling_mask=before['valid'], rolling_arrays=rolling)
        reference = {key: first['force'][key] for key in ('real_mean', 'real_std', 'sim_mean', 'sim_std')}
        second = evaluate_view(after, 'after', 'Same time aligned logs; actual deployed actor input mapping without numeric inverse',
            after['steady'], rolling_mask=after['valid'], force_reference=reference, rolling_arrays=rolling)
        first['units'] = dict(q='rad', dq='rad/s', ddq='rad/s²', tau='N·m')
        second['units'] = dict(q='무차원', dq='무차원', ddq='rad/s²', tau='N·m')
        first['space'] = 'time-aligned physical measurements'
        second['space'] = 'q/dq: deployed normalized actor input; ddq/tau/force: physical (no deployment rule)'
        second['physicalMetrics'] = first['metrics']
        for item in (first, second):
            item['torqueSemantics'] = dict(real='ls_tau: estimated motor torque', sim='ls_tau: simulated actuator torque',
                mapping='joint order only; no deployment torque scale/offset')
            item['suggestedTime'] = timing['move_start_s'] + 4.5
        velocity = rm.get('vx')
        if velocity is not None and not np.isfinite(velocity): velocity = None
        entry = dict(id=name, file=f'{name}.json', velocity=velocity, syncPassed=sync['passed'],
            dataset=dataset, realPath=str(real_path.resolve()), simPath=str(sim_path.resolve()),
            lagMs=round(sync['applied_lag_s'] * 1000, 1), realSha256=sha(real_path), simSha256=sha(sim_path))
        trial = dict(trial=name, velocity=velocity, synchronization=sync, timing=timing,
            mapping=contract, provenance=entry, before=first, after=second)
        (directory / 'metrics' / f'{name}.json').write_text(json.dumps(trial, separators=(',', ':'), allow_nan=False))
        manifest['trials'].append(entry)
        arrays = dict(t=before['t'], valid=before['valid'], steady=before['steady'])
        for key in ('real_q', 'real_dq', 'real_ddq', 'real_tau'):
            arrays[key] = before[key]
            arrays[key + '_before'] = before[key]
            arrays[key + '_after'] = after[key]
        arrays['real_foot_sensor_raw'] = before['real_foot_sensor_raw']
        for key in ('sim_q', 'sim_dq', 'sim_ddq', 'sim_tau', 'sim_foot_fz_N'):
            arrays[key + '_before'] = before[key]
            arrays[key + '_after'] = after[key]
            arrays[key + '_clock_only'] = before[key + '_clock_only']
        arrays['applied_lag_s'] = sync['applied_lag_s']
        np.savez_compressed(directory / 'processed' / f'{name}_mapping.npz', **arrays, **coordinates, **rolling)
        summary = dict(trial=name, provenance=entry, mapping=contract, synchronization=sync, events=events,
            n_steady=int(before['steady'].sum()), metrics={})
        for label, record, view in (('before', before, first), ('after', after, second)):
            mask = record['steady']
            summary['metrics'][label] = dict(
                **{q: metric(record['real_' + q].reshape(-1), record['sim_' + q].reshape(-1), np.repeat(mask, 12))
                   for q in ('q', 'dq', 'ddq', 'tau')},
                units=view['units'], joints=view['metrics'], force=view['force']['zscoreRmse'],
                force_correlation=view['force']['correlation'])
        results.append(summary)
        print(f'{name}: common shift {entry["lagMs"]:+.1f} ms; {sync["status"]}; {summary["n_steady"]} paired evaluation samples', flush=True)
    if not results:
        raise ValueError('No nonempty paired recordings to compare')
    (directory / 'metrics/mapping_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    write_report(results, directory, manifest['excludedTrials'])
    manifest_path = directory / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    # The exporter consumes only the files just saved, not these in-memory arrays.
    export_ui(directory, require_complete=False)
    if plots:
        # Keep plotting libraries isolated from the analysis NumPy environment.
        # This process reads the saved bundle only; it does not repeat analysis.
        with tempfile.TemporaryDirectory(prefix='go2-plot-config-') as cache:
            environment = dict(os.environ)
            environment.setdefault('MPLCONFIGDIR', cache)
            subprocess.run([sys.executable, '-s', str(Path(__file__).with_name('render_results.py')),
                '--result-dir', str(directory)], check=True, env=environment)
    sources = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('deployment_normalization.json')]
    manifest['analysisCodeSha256'] = hashlib.sha256(''.join(sha(path) for path in sources).encode()).hexdigest()
    manifest['artifacts'] = {str(path.relative_to(directory)): sha(path)
        for path in sorted(directory.rglob('*')) if path.is_file() and path != manifest_path}
    manifest['status'] = 'complete'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def build(data_dir=None, result_dir=None, policy_dir=None, dataset=DEFAULT_DATASET, plots=True):
    validate_dataset(dataset)
    data_dir = Path(data_dir) if data_dir is not None else PROJECT / 'data'
    result_dir = Path(result_dir) if result_dir is not None else result_path(dataset)
    result_dir = result_dir.absolute()
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{dataset}-build-', dir=result_dir.parent) as temporary:
        stage = Path(temporary) / 'result'
        stage.mkdir()
        manifest = _analyze(data_dir, stage, policy_dir, dataset, plots)
        summary = None
        if result_dir.resolve() == result_path(dataset).resolve():
            from .dataset_summary import write_summary
            summary = write_summary(stage, Path(temporary) / 'summary.txt', display_directory=result_dir)
        publish_directory(stage, result_dir)
        if summary is not None:
            os.replace(summary, result_dir.parent / f'{dataset}_real_sim_analysis.txt')
    print(f'Published {len(manifest["trials"])} trials: {result_dir}', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT / 'data')
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--policy-dir', type=Path)
    parser.add_argument('--skip-plots', action='store_true', help='Publish arrays, metrics and UI without static figures')
    args = parser.parse_args()
    build(args.data_dir, args.result_dir, args.policy_dir, args.dataset, plots=not args.skip_plots)


