#!/usr/bin/env python3
"""Deployment-contract Real2Sim conversion. Never connects to a real robot.

convert: existing real Python/ONNX Runtime environment, NumPy only.
analyze: existing ML Python environment (SciPy, ONNX, MuJoCo, Matplotlib).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src/go2_autonomous_driving'))
from go2_autonomous_driving.real.real_control import PolicyMapping
from go2_autonomous_driving.real.control_transitions import JointTargetRamp

POLICY_HASH = '3fb9c6f82b20c3e048506deeb89b0e1f33b90194f4b809a3e46dd62c0d5e2583'
JOINTS = [f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for joint in ('hip', 'thigh', 'calf')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def restore_output_owner(directory):
    """Keep bind-mounted artifacts editable by the owner of result/."""
    owner = directory.parent.stat()
    for path in [directory, *directory.rglob('*')]:
        try:
            os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
        except PermissionError:
            pass


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files if k != 'meta'}, json.loads(z['meta'].item())


def hold_indices(source, query, max_age):
    """Causal receive-clock sample selection, with explicit stale-data mask."""
    source, query = np.asarray(source), np.asarray(query)
    if np.any(np.diff(source) <= 0):
        raise ValueError('Source clock must be strictly increasing')
    index = np.searchsorted(source, query, side='right') - 1
    safe = np.clip(index, 0, len(source) - 1)
    age = query - source[safe]
    return safe, (index >= 0) & (age >= 0) & (age <= max_age)


def targets_to_action(target_sdk, mapping):
    return (np.asarray(target_sdk)[..., mapping.order] - mapping.default) / mapping.scale


def action_to_targets(action, mapping):
    action = np.asarray(action)
    if mapping.meta['clip_actions'] is not None:
        action = np.clip(action, -mapping.meta['clip_actions'], mapping.meta['clip_actions'])
    policy = mapping.default + mapping.scale * action
    sdk = np.empty_like(policy)
    sdk[..., mapping.order] = policy
    return sdk


def verify_assets(directory):
    import onnxruntime as ort

    meta = json.loads((directory / 'policy.json').read_text())
    if meta.get('policy_variant') != 'baseline' or not meta.get('normalization_included'):
        raise ValueError('Requires baseline deployment policy with normalization included')
    if digest(directory / 'policy.onnx') != meta['onnx_sha256']:
        raise ValueError('ONNX SHA256 mismatch')
    mapping = PolicyMapping(meta)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory / 'policy.onnx'), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    key = session.get_inputs()[0].name
    if session.get_inputs()[0].shape != [1, 47] or session.get_outputs()[0].shape != [1, 12]:
        raise ValueError('Unexpected policy interface')

    def infer(obs):
        return np.concatenate([session.run(None, {key: row[None].astype(np.float32)})[0] for row in obs])

    with np.load(directory / 'parity.npz', allow_pickle=False) as z:
        obs, action = z['observations'], z['actions']
    samples = json.loads((directory / 'samples.json').read_text())
    if obs.shape != (len(samples), 47) or action.shape != (len(samples), 12) or not len(samples):
        raise ValueError('Invalid parity samples')
    output = infer(obs)
    rebuilt = np.concatenate([mapping.observation(mapping.to_sdk(s['q']), mapping.to_sdk(s['dq']),
        s['quaternion'], s['gyro'], o[6:9], s['step'], o[35:47]) for s, o in zip(samples, obs)])
    err_policy = float(np.max(abs(output - action)))
    err_mapping = float(np.max(abs(rebuilt - obs)))
    if not np.isfinite(output).all() or max(err_policy, err_mapping) >= 1e-4:
        raise ValueError('Deployment parity validation failed')
    verification = dict(frames=len(samples), onnx_torch_max_error=err_policy,
                        observation_mapping_max_error=err_mapping,
                        assets={name: digest(directory / name) for name in
                                ('policy.onnx', 'policy.json', 'parity.npz', 'samples.json')},
                        checkpoint_sha256=meta['checkpoint_sha256'], passed=True)
    return meta, mapping, infer, verification


def event(data, label):
    return float(data['ev_t'][np.flatnonzero(data['ev_label'] == label)[0]])


def reconstruct(real, meta, mapping, infer):
    origin, dt = event(real, 'record_start'), mapping.meta['step_dt']
    start = max(real['ls_t'][0], real['lc_t'][0] + dt, real['rl_t'][0])
    stop = min(real['ls_t'][-1], real['lc_t'][-1], real['rl_t'][-1])
    t = np.arange(np.ceil((start - origin) / dt) * dt, stop - origin, dt)
    absolute = t + origin
    si, sv = hold_indices(real['ls_t'], absolute, .010)
    ci, cv = hold_indices(real['lc_t'], absolute, .010)
    pi, pv = hold_indices(real['lc_t'], absolute - dt, .010)
    ri, rv = hold_indices(real['rl_t'], absolute, .100)
    valid = sv & cv & pv & rv
    q, dq = real['ls_q'][si], real['ls_dq'][si]
    quat, gyro = real['ls_quaternion'][si], real['ls_gyro'][si]
    command = real['rl_command'][ri]
    previous = targets_to_action(real['lc_q'][pi], mapping)
    logged = real['lc_q'][ci]
    steps = np.rint(t / dt).astype(int)
    move_start, move_stop = event(real, 'move_start') - origin, event(real, 'move_stop') - origin
    calibration = valid & (t >= move_start + 2) & (t <= move_start + 7)
    validation = valid & (t >= move_stop - 6) & (t <= move_stop - 1)
    if calibration.sum() < 100 or validation.sum() < 100:
        raise ValueError('Insufficient separate phase calibration/validation windows')

    def observations(offset):
        return np.concatenate([mapping.observation(*values) for values in
            zip(q, dq, quat, gyro, command, steps + offset, previous)])

    count = int(round(mapping.meta['phase_period'] / dt))
    candidates = []
    for offset in range(count):
        obs = observations(offset)
        prediction = action_to_targets(infer(obs[calibration]), mapping)
        candidates.append(float(np.mean((prediction - logged[calibration]) ** 2)))
    best = int(np.argmin(candidates))
    obs = observations(best)
    action = infer(obs)
    raw_targets = action_to_targets(action, mapping)
    config = meta['control_config']
    ramp = JointTargetRamp(config['max_joint_target_rate'], dt,
        transition_rate=config.get('phase_transition_joint_rate'),
        transition_seconds=config.get('phase_transition_seconds', 0.))
    filtered = []
    for i, target in enumerate(raw_targets):
        # Teacher-forced one-step replay: observed preceding motor target anchors
        # the slew limiter. Preserve its transition state, but not output drift.
        ramp.value = real['lc_q'][pi[i]].astype(np.float32).copy()
        filtered.append(ramp.step(target, moving=bool(np.any(obs[i, 9:11]))))
    filtered = np.asarray(filtered)
    reconstructed_q = np.stack([mapping.to_sdk(row[11:23] + mapping.default) for row in obs])
    inverse_target = action_to_targets(targets_to_action(logged, mapping), mapping)
    limits = mapping.limits
    violation = ((raw_targets[:, mapping.order] < limits[:, 0]) |
                 (raw_targets[:, mapping.order] > limits[:, 1])).any(axis=1)
    info = dict(
        phase_offset_steps=best, phase_offset_s=best * dt,
        phase_fit='30 discrete phase offsets; first 5 steady seconds only; match raw ONNX targets to logged LowCmd',
        phase_candidate_mse=candidates, dropped_frames=int((~valid).sum()),
        q_roundtrip_max_error_rad=float(abs(reconstructed_q - q).max()),
        target_roundtrip_max_error_rad=float(abs(inverse_target - logged).max()),
        raw_target_limit_violation_frames=int((violation & valid).sum()),
        raw_target_rmse_calibration_rad=float(np.sqrt(np.mean((raw_targets[calibration] - logged[calibration]) ** 2))),
        raw_target_rmse_validation_rad=float(np.sqrt(np.mean((raw_targets[validation] - logged[validation]) ** 2))),
        rate_target_rmse_validation_rad=float(np.sqrt(np.mean((filtered[validation] - logged[validation]) ** 2))),
        previous_target_rmse_validation_rad=float(np.sqrt(np.mean((real['lc_q'][pi][validation] - logged[validation]) ** 2))),
        clock='nominal 50 Hz receive-clock grid; latest causal sensors/command; unknown actual inference timing',
        replay_kind='teacher-forced observation/action reconstruction; NOT an exact historical policy trace',
        move_start_s=move_start, move_stop_s=move_stop)
    arrays = dict(t=t, valid=valid, calibration=calibration, validation=validation,
        steady=valid & (t >= move_start + 2) & (t <= move_stop - 1),
        real_q=q, real_dq=dq, real_quaternion=quat, real_gyro=gyro,
        real_sensor_counts=real['ls_ff'][si], observation=obs, action_onnx=action,
        action_logged=targets_to_action(logged, mapping), previous_action=previous,
        target_logged_sdk=logged, target_onnx_raw_sdk=raw_targets,
        target_onnx_rate_sdk=filtered, q_roundtrip_sdk=reconstructed_q)
    return arrays, info


def convert(args):
    meta, mapping, infer, verification = verify_assets(args.policy_dir)
    target = args.output_dir / 'converted'; target.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'asset_verification.json', verification)
    summary = []
    for path in sorted((args.data_dir / 'real').glob('*.npz')):
        real, rm = load(path); sim_path = args.data_dir / 'sim' / path.name
        sim, sm = load(sim_path)
        if rm['checkpoint_sha256'] != meta['checkpoint_sha256'] or sm['checkpoint_sha256'] != meta['checkpoint_sha256']:
            raise ValueError('Recorded checkpoint differs from deployment assets')
        if not sm['completed'] or sm['reference_sha256'] != digest(path):
            raise ValueError('Simulation/reference identity mismatch')
        arrays, info = reconstruct(real, rm, mapping, infer)
        sim_onnx = infer(sim['policy_observation'].astype(np.float32))
        info.update(trial=path.stem, real_sha256=digest(path), sim_sha256=digest(sim_path),
            sim_onnx_vs_saved_pt_max_error=float(abs(sim_onnx - sim['policy_action_raw']).max()))
        if info['sim_onnx_vs_saved_pt_max_error'] >= 1e-4:
            raise ValueError('Saved simulation and deployment ONNX policies disagree')
        arrays.update(sim_t=sim['lc_t'], sim_observation=sim['policy_observation'],
            sim_action_onnx=sim_onnx, sim_action_logged=sim['policy_action_applied'],
            sim_target_sdk=sim['lc_q'])
        np.savez_compressed(target / path.name, **arrays)
        write_json(target / (path.stem + '.json'), info)
        summary.append(info)
        print(f'{path.stem}: phase {info["phase_offset_s"]:.2f}s; held-out target RMSE '
              f'{info["rate_target_rmse_validation_rad"]:.5f} rad', flush=True)
    if not summary:
        raise ValueError('No paired recordings')
    write_json(args.output_dir / 'conversion_summary.json', dict(verification=verification, trials=summary,
        scope='deployment interface conversion only; no physics parameter fitting'))
    restore_output_owner(args.output_dir)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('convert', 'analyze'), required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'src/Real2Sim/data')
    p.add_argument('--policy-dir', type=Path, default=ROOT / 'artifacts/go2_real/baseline' / POLICY_HASH)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'src/Real2Sim/result/legacy')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    if args.stage == 'convert':
        convert(args)
    else:
        from analyze_converter import analyze
        analyze(args)
