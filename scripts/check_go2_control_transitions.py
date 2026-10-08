#!/usr/bin/env python3
"""Docker-only MuJoCo regression: walking -> Space/S at six gait phases.

Uses the trained actor, actual simulated IMU/joints, and production mapping/ramp
classes. No ROS, sockets, DDS or physical robot output. Compares the previous
velocity/joint ramps with slower braking and gait-transition target easing.
Checks at the 50 Hz policy boundary, not the real 500 Hz motor loop.
Simulation evidence does not establish hardware stability.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import src.locomotion_rl  # noqa: F401

from go2_autonomous_driving.real.control_transitions import JointTargetRamp, VelocityRamp
from go2_autonomous_driving.real.real_control import PolicyMapping, gravity_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speeds', type=float, nargs='+', default=[.5, .78, 1.5])
    parser.add_argument('--modes', choices=('space', 'reverse'), nargs='+', default=['space', 'reverse'])
    parser.add_argument('--forward-seconds', type=float, help='Stop before reaching the requested speed')
    parser.add_argument('--randomized', action='store_true', help='Keep play-mode physical randomization')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if any(not np.isfinite(speed) or speed <= 0 for speed in args.speeds):
        parser.error('speeds must be finite and positive')
    if args.forward_seconds is not None and (not np.isfinite(args.forward_seconds) or args.forward_seconds <= 0):
        parser.error('forward-seconds must be finite and positive')
    config = json.loads(Path('src/go2_autonomous_driving/config/real_closed_loop.json').read_text())
    mapping = PolicyMapping(json.loads(Path('artifacts/go2_real/policy.json').read_text()))
    cases = [(revised, mode, offset / 10, speed) for revised in (False, True)
             for speed in args.speeds for mode in args.modes for offset in range(6)]
    cfg = load_env_cfg('Unitree-Go2-Flat', play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = len(cases)
    cfg.episode_length_s = 60
    if not args.randomized:
        # Identical nominal initial states/physical parameters for paired cases.
        # Randomized trials are useful separately, but are not paired comparisons.
        for name in ('foot_friction', 'encoder_bias', 'base_com'):
            cfg.events.pop(name, None)
        cfg.events['reset_base'].params['pose_range'] = {}
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        agent = load_rl_cfg('Unitree-Go2-Flat')
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls('Unitree-Go2-Flat')(wrapped, asdict(agent), device='cuda:0')
        checkpoint = Path(__file__).resolve().parents[1] / 'src/go2_autonomous_driving/model_pt/model_10000.pt'
        runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location='cuda:0')
        policy = runner.get_inference_policy(device='cuda:0')
        wrapped.reset()
        ramps = [VelocityRamp(config['command_acceleration'],
                              config['command_deceleration'] if revised else [1., .8, 2.], env.step_dt)
                 for revised, _, _, _ in cases]
        joints = [JointTargetRamp(config['max_joint_target_rate'], env.step_dt,
                                  transition_rate=config['phase_transition_joint_rate'] if revised else None,
                                  transition_seconds=config['phase_transition_seconds'] if revised else 0.)
                  for revised, _, _, _ in cases]
        stops = [1 + (speed / config['command_acceleration'][0] + 1
                      if args.forward_seconds is None else args.forward_seconds) + offset
                 for _, _, offset, speed in cases]
        ends = [stop + speed / config['command_deceleration'][0]
                + speed / config['command_acceleration'][0] + 1
                for stop, (_, _, _, speed) in zip(stops, cases)]
        duration = max(end + speed / config['command_deceleration'][0] + 2
                       for end, (_, _, _, speed) in zip(ends, cases))
        previous = np.zeros((len(cases), 12), dtype=np.float32)
        last_target = None
        metrics = [dict(revised=f, mode=m, offset=o, speed=s, max_error=0., max_transition_error=0.,
                        max_walking_error=0., max_target_step=0., max_tilt_deg=0., resets=0,
                        first_guard=None, forward_speeds=[], stop_speeds=[])
                   for f, m, o, s in cases]
        with torch.inference_mode():
            for step in range(round(duration / env.step_dt)):
                t = step * env.step_dt
                robot = env.scene['robot'].data
                q = robot.joint_pos.detach().cpu().numpy()
                dq = robot.joint_vel.detach().cpu().numpy()
                quat = robot.root_link_quat_w.detach().cpu().numpy()
                gyro = robot.root_link_ang_vel_b.detach().cpu().numpy()
                velocity = robot.root_link_lin_vel_b.detach().cpu().numpy()
                obs_array = []
                commands = []
                for i, (_, mode, _, speed) in enumerate(cases):
                    vx = 0 if t < 1 or t >= ends[i] else speed if t < stops[i] else (0 if mode == 'space' else -speed)
                    command = ramps[i].step([vx, 0, 0])
                    commands.append(command)
                    obs_array.append(mapping.observation(mapping.to_sdk(q[i]), mapping.to_sdk(dq[i]),
                                                         quat[i], gyro[i], command, step, previous[i])[0])
                obs = wrapped.get_observations()
                obs['actor'] = torch.from_numpy(np.stack(obs_array)).to(env.device)
                output = policy(obs).detach().cpu().numpy()
                target_array = []
                for i, (_, _, _, speed) in enumerate(cases):
                    metric = metrics[i]
                    if metric['first_guard'] is not None:
                        previous[i] = 0
                        target_array.append(mapping.default.copy())
                        continue
                    # Preserve raw-output limit validation used by the real backend.
                    try:
                        _, raw = mapping.targets(output[i])
                    except ValueError as error:
                        metric['first_guard'] = dict(time=t, command=float(commands[i][0]),
                                                    after_stop_request=t >= stops[i], reason=str(error))
                        previous[i] = 0
                        target_array.append(mapping.default.copy())
                        continue
                    target = joints[i].step(raw, initial=mapping.to_sdk(q[i]) if step == 0 else None,
                                            moving=bool(np.any(obs_array[i][9:11])))
                    previous[i] = (target[mapping.order] - mapping.default) / mapping.scale
                    target_array.append(target[mapping.order])
                    error = float(np.max(np.abs(target[mapping.order] - q[i])))
                    if t >= 1:
                        metric['max_error'] = max(metric['max_error'], error)
                        tilt = np.degrees(np.arccos(np.clip(-gravity_body(quat[i])[2], -1, 1)))
                        metric['max_tilt_deg'] = max(metric['max_tilt_deg'], float(tilt))
                        if last_target is not None:
                            metric['max_target_step'] = max(metric['max_target_step'], float(np.max(np.abs(target[mapping.order] - last_target[i]))))
                    if t >= stops[i]:
                        metric['max_transition_error'] = max(metric['max_transition_error'], error)
                    elif t >= 1:
                        metric['max_walking_error'] = max(metric['max_walking_error'], error)
                    if error >= config['max_tracking_error'] or metric['max_tilt_deg'] >= config['max_tilt_degrees']:
                        metric['first_guard'] = dict(time=t, command=float(commands[i][0]),
                                                    after_stop_request=t >= stops[i], error=error,
                                                    reason='joint tracking or tilt')
                        previous[i] = 0
                    if stops[i] - .5 <= t < stops[i]:
                        metric['forward_speeds'].append(float(velocity[i, 0]))
                    if t >= duration - 1:
                        metric['stop_speeds'].append(float(abs(velocity[i, 0])))
                last_target = np.array(target_array)
                _, _, done, _ = wrapped.step(torch.from_numpy(previous.copy()).to(env.device))
                for i, reset in enumerate(done.cpu().numpy()):
                    if metrics[i]['first_guard'] is None:
                        metrics[i]['resets'] += int(reset)
                        if reset:
                            metrics[i]['first_guard'] = dict(time=t, reason='simulation reset')
        for metric in metrics:
            forward, stopped = metric.pop('forward_speeds'), metric.pop('stop_speeds')
            metric['mean_forward_speed'] = float(np.mean(forward)) if forward else None
            metric['mean_abs_stopped_speed'] = float(np.mean(stopped)) if stopped else None
        print('TRANSITION_METRICS ' + json.dumps(metrics), flush=True)
        if args.report:
            args.report.write_text(json.dumps(dict(config=config, seed=args.seed, forward_seconds=args.forward_seconds,
                                                   randomized=args.randomized, metrics=metrics), indent=2) + '\n')
        revised = [m for m in metrics if m['revised']]
        if any(m['first_guard'] or m['resets'] or m['max_tilt_deg'] >= config['max_tilt_degrees']
               or m['max_error'] >= config['max_tracking_error'] for m in revised):
            raise SystemExit('FAIL: revised transitions still trigger a guard in MuJoCo')
        print('PASS: revised Space/reverse cases stay below unchanged guards in MuJoCo; hardware unverified')
    finally:
        env.close()


if __name__ == '__main__':
    main()
