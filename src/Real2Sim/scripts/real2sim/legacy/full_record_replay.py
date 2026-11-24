#!/usr/bin/env python3
"""One initialization, then replay the entire recorded LowCmd without resets."""
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from real2sim_converter import ROOT, load, digest, write_json, restore_output_owner
sys.path.insert(0, str(ROOT / 'src/Real2Sim/scripts/offline_identification'))
from real2sim_common import state_clock, command_at, interpolate_quaternion, hysteresis
from mujoco_replay import Replay, DEFAULT_PARAMETERS


def full_replay(real_path, output, engine):
    raw, _ = load(real_path)
    clock, keep, _ = state_clock(raw)
    origin = float(raw['ev_t'][raw['ev_label'] == 'record_start'][0])
    dt = .005
    start = max(clock[0], raw['lc_t'][0])
    end = min(clock[-1], raw['lc_t'][-1])
    times = origin + np.arange(np.ceil((start - origin) / dt) * dt, end - origin, dt)
    interpolate = lambda x: np.stack([np.interp(times, clock, column) for column in x[keep].T], axis=1)
    data = dict(t=times, q=interpolate(raw['ls_q']), dq=interpolate(raw['ls_dq']),
                gyro=interpolate(raw['ls_gyro']),
                quaternion=interpolate_quaternion(clock, raw['ls_quaternion'][keep], times))
    idx = np.searchsorted(clock, times, side='right') - 1
    data['ff'] = raw['ls_ff'][keep][idx]
    move_start = float(raw['ev_t'][raw['ev_label'] == 'move_start'][0])
    calibration = (clock >= move_start + 2) & (clock <= move_start + 7)
    low, high = np.percentile(raw['ls_ff'][keep][calibration], [5, 95], axis=0)
    span = np.maximum(high - low, 1.)
    data['contact'] = hysteresis(data['ff'], low + .30 * span, low + .15 * span, dt)
    valid = np.ones(len(times), bool); valid[0] = False
    for i in np.flatnonzero(np.diff(clock) > .010 + 1e-9):
        valid &= ~((times >= clock[i] - .025) & (times <= clock[i + 1] + .025))
    for i in np.flatnonzero(np.diff(raw['lc_t']) > .010 + 1e-9):
        valid &= ~((times >= raw['lc_t'][i]) & (times <= raw['lc_t'][i + 1]))
    initialization = engine.initialize(data, 0, DEFAULT_PARAMETERS)
    model, physical = engine.model, engine.data
    substeps = int(round(dt / engine.timestep))
    queries = times[0] + np.arange((len(times) - 1) * substeps + 1) * engine.timestep
    commands = command_at(raw['lc_t'], queries)
    q, dq, force = [], [], []
    base_position, base_quaternion, foot_height, nonfoot_support = [], [], [], []
    nonfoot_bodies = set()
    failure = None
    for step, ci in enumerate(commands):
        physical.ctrl[engine.aids] = np.clip(
            raw['lc_kp'][ci] * (raw['lc_q'][ci] - physical.qpos[engine.qids])
            + raw['lc_kd'][ci] * (raw['lc_dq'][ci] - physical.qvel[engine.dids])
            + raw['lc_tau'][ci], -engine.limits, engine.limits)
        if step % substeps == 0:
            engine.mj.mj_forward(model, physical)
            q.append(physical.qpos[engine.qids].copy())
            dq.append(physical.qvel[engine.dids].copy())
            force.append(engine.forces())
            base_position.append(physical.qpos[:3].copy())
            base_quaternion.append(physical.qpos[3:7].copy())
            foot_height.append(physical.geom_xpos[engine.feet, 2] - model.geom_size[engine.feet, 0])
            support = 0.
            for contact_id in range(physical.ncon):
                contact = physical.contact[contact_id]
                if engine.floor not in (contact.geom1, contact.geom2):
                    continue
                robot_geom = contact.geom2 if contact.geom1 == engine.floor else contact.geom1
                if robot_geom in engine.feet:
                    continue
                wrench = np.zeros(6)
                engine.mj.mj_contactForce(model, physical, contact_id, wrench)
                sign = 1 if contact.geom2 == robot_geom else -1
                support += (sign * contact.frame.reshape(3, 3).T @ wrench[:3])[2]
                nonfoot_bodies.add(model.body(int(model.geom_bodyid[robot_geom])).name)
            nonfoot_support.append(support)
        if step < len(commands) - 1:
            engine.mj.mj_step(model, physical)
            if not np.isfinite(physical.qpos).all() or not np.isfinite(physical.qvel).all() or np.max(abs(physical.qvel)) > 1000:
                failure = f'Numerical divergence at {step * engine.timestep:.3f}s; no reset performed'
                break
    count = len(q); times = times[:count]; valid = valid[:count]
    filt = butter(4, 15., fs=200., output='sos')
    real_ddq = np.gradient(sosfiltfilt(filt, data['dq'][:count], axis=0), dt, axis=0)
    replay_ddq = np.gradient(sosfiltfilt(filt, np.asarray(dq), axis=0), dt, axis=0)
    ddq_valid = valid.copy(); ddq_valid[:5] = False; ddq_valid[-5:] = False
    arrays = dict(t=times - origin, real_q=data['q'][:count], real_dq=data['dq'][:count],
        real_ddq=real_ddq, replay_q=np.asarray(q), replay_dq=np.asarray(dq), replay_ddq=replay_ddq,
        force_world_N=np.asarray(force), sensor_counts=data['ff'][:count],
        valid=valid, ddq_valid=ddq_valid,
        real_base_quaternion=data['quaternion'][:count], replay_base_quaternion=np.asarray(base_quaternion),
        replay_base_position=np.asarray(base_position), replay_foot_height_m=np.asarray(foot_height),
        replay_nonfoot_support_N=np.asarray(nonfoot_support))
    real_rpy = Rotation.from_quat(arrays['real_base_quaternion'][:, [1, 2, 3, 0]]).as_euler('xyz')
    replay_rpy = Rotation.from_quat(arrays['replay_base_quaternion'][:, [1, 2, 3, 0]]).as_euler('xyz')
    arrays.update(real_base_rpy=real_rpy, replay_base_rpy=replay_rpy)
    up = 1 - 2 * np.sum(arrays['replay_base_quaternion'][:, 1:3] ** 2, axis=1)
    tilt = np.rad2deg(np.arccos(np.clip(up, -1, 1)))
    real_up = 1 - 2 * np.sum(arrays['real_base_quaternion'][:, 1:3] ** 2, axis=1)
    real_tilt = np.rad2deg(np.arccos(np.clip(real_up, -1, 1)))
    large_tilt = tilt > 45
    # Sustained tilt avoids labeling a single noisy sample as a fall.
    sustained = np.convolve(large_tilt.astype(int), np.ones(20, dtype=int), mode='valid') >= 20
    first = int(np.flatnonzero(sustained)[0]) if sustained.any() else None
    steady = valid & (times >= move_start + 2) & (times <= move_start + 14)
    diagnostics = dict(
        sustained_tilt_over_45deg_first_s=float(times[first] - origin) if first is not None else None,
        max_replay_tilt_deg=float(tilt.max()), max_real_tilt_deg=float(real_tilt.max()),
        nominal_weight_N=float(model.body_mass.sum() * abs(model.opt.gravity[2])),
        steady_mean_foot_support_N=np.asarray(force)[steady, :, 2].mean(axis=0).tolist(),
        steady_contact_fraction=(np.asarray(force)[steady, :, 2] > 1).mean(axis=0).tolist(),
        steady_mean_foot_height_m=np.asarray(foot_height)[steady].mean(axis=0).tolist(),
        steady_mean_nonfoot_support_N=float(np.asarray(nonfoot_support)[steady].mean()),
        nonfoot_ground_contact_bodies=sorted(nonfoot_bodies),
        meaning='Physical free replay can lose balance; foot GRF excludes body/leg-ground support. This is not policy closed-loop simulation.')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'{real_path.stem}_predictions.npz', **{k: v[None] for k, v in arrays.items()})
    metadata = dict(source_sha256=digest(real_path), initialized_once=True, resets=0,
        completed=failure is None, failure=failure, initialization=initialization,
        requested_duration_s=float(end - start), recorded_duration_s=float(times[-1] - times[0]),
        first_record_time_s=float(times[0] - origin), last_record_time_s=float(times[-1] - origin),
        physics=engine.metadata(), parameters_fitted=False, diagnostics=diagnostics,
        gaps='Commands held causally through gaps; affected samples excluded from metrics; no state reinjection',
        sensor_force_unit='raw counts', simulation_force_unit='N')
    write_json(output / f'{real_path.stem}_metadata.json', metadata)
    print(f'{real_path.stem}: {metadata["recorded_duration_s"]:.2f}s, initialized once, resets=0, completed={metadata["completed"]}', flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'src/Real2Sim/data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'src/Real2Sim/result/legacy')
    args = parser.parse_args()
    engine = Replay()
    for path in sorted((args.data_dir / 'real').glob('*.npz')):
        full_replay(path, args.output_dir / 'replay_full', engine)
    from force_diagnostics import write_diagnostics
    write_diagnostics(args.output_dir)
    restore_output_owner(args.output_dir)
