#!/usr/bin/env python3
"""Run in the mjlab Docker environment to export the trained Go2 Flat actor."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import src.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'src/go2_autonomous_driving/model_pt/model_10000.pt')
    parser.add_argument('--output', type=Path, default=Path('artifacts/go2_real'))
    args = parser.parse_args()
    cfg = load_env_cfg('Unitree-Go2-Flat', play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        agent = load_rl_cfg('Unitree-Go2-Flat')
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls('Unitree-Go2-Flat')(wrapped, asdict(agent), device='cuda:0')
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True, map_location='cuda:0')
        metadata = get_base_metadata(env, str(args.checkpoint.name))
        expected = ['base_ang_vel', 'projected_gravity', 'command', 'phase',
                    'joint_pos', 'joint_vel', 'actions']
        if metadata['observation_names'] != expected:
            raise ValueError('Unsupported observation layout: ' + repr(metadata['observation_names']))
        sdk_names = [f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
                     for joint in ('hip', 'thigh', 'calf')]
        model = env.sim.mj_model
        limits = [model.jnt_range[model.joint('robot/' + name).id].tolist()
                  for name in metadata['joint_names']]
        metadata.update(
            version=1, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            policy_to_sdk=[sdk_names.index(name) for name in metadata['joint_names']],
            joint_limits=limits, step_dt=env.step_dt,
            phase_period=cfg.observations['actor'].terms['phase'].params['period'],
            clip_actions=agent.clip_actions,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        runner.export_policy_to_onnx(str(args.output), 'policy.onnx')
        metadata['onnx_sha256'] = hashlib.sha256((args.output / 'policy.onnx').read_bytes()).hexdigest()
        (args.output / 'policy.json').write_text(json.dumps(metadata, indent=2) + '\n')
        # Save raw observations and the original normalized actor's outputs for
        # independent host ONNX parity and real-observation mapping checks.
        policy = runner.get_inference_policy(device='cuda:0')
        wrapped.reset()
        observations, actions, samples = [], [], []
        with torch.inference_mode():
            for step in range(100):
                obs = wrapped.get_observations()
                action = policy(obs)
                observations.append(obs['actor'].cpu().numpy())
                actions.append(action.cpu().numpy())
                robot = env.scene['robot'].data
                samples.append(dict(
                    q=robot.joint_pos[0].cpu().tolist(), dq=robot.joint_vel[0].cpu().tolist(),
                    quaternion=robot.root_link_quat_w[0].cpu().tolist(),
                    gyro=env.scene['robot'].data.root_link_ang_vel_b[0].cpu().tolist(),
                    step=int(env.episode_length_buf[0]),
                ))
                wrapped.step(action)
        np.savez(args.output / 'parity.npz', observations=np.concatenate(observations),
                 actions=np.concatenate(actions))
        (args.output / 'samples.json').write_text(json.dumps(samples))
        print('Exported normalized actor and mapping to', args.output)
    finally:
        env.close()


if __name__ == '__main__':
    main()
