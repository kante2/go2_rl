#!/usr/bin/env python3
"""Export the saved QuietPaw actor, including its 48-input normalization.

Run in the mjlab Docker environment. No ROS nodes or motor publishers.
QuietPaw Flat and this repository's baseline share robot, actions and the
first 47 observations. Use that environment to extract joint metadata and
verify those observations; load the actor and cost limits from the PT itself.
This does not require copying QuietPaw's training/critic code onto branch 1009.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from go2_autonomous_driving.real.real_quietpaw import (
    BASE_OBSERVATIONS, QUIETPAW_OBSERVATIONS, cost_limit_from_quietness,
)


def load_actor(checkpoint):
    infos = checkpoint.get('infos') or {}
    if 'quietpaw' not in infos or 'quietpaw_agent' not in infos:
        raise ValueError('Expected a QuietPaw checkpoint with saved model and cost settings')
    agent = infos['quietpaw_agent']
    condition = deepcopy(infos['quietpaw']['noise_condition'])
    cost_limit_from_quietness(0.5, condition)  # Reject invalid/non-finite saved ranges.
    if tuple(agent['obs_groups']['actor']) != ('actor',):
        raise ValueError('Only a single Flat actor observation group is supported')
    actor_cfg = deepcopy(agent['actor'])
    if actor_cfg.pop('class_name') != 'MLPModel':
        raise ValueError('Only the saved QuietPaw MLPModel actor is supported')
    state = checkpoint['actor_state_dict']
    if state['mlp.0.weight'].shape[1] != 48:
        raise ValueError('Expected QuietPaw Flat with 48 actor inputs')
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint actor contains non-finite parameters')
    dummy = TensorDict({'actor': torch.zeros(1, 48)}, batch_size=[1])
    actor = MLPModel(dummy, {'actor': ['actor']}, 'actor', 12, **actor_cfg)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor, agent, condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('src/go2_autonomous_driving/model_pt/model_9999.pt'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/go2_real_quietpaw'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    actor, agent, condition = load_actor(checkpoint)

    # Heavy simulation imports only after validating the checkpoint structure.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import get_base_metadata
    from mjlab.tasks.registry import load_env_cfg
    import src.locomotion_rl  # noqa: F401

    cfg = load_env_cfg('Unitree-Go2-Flat', play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        metadata = get_base_metadata(env, args.checkpoint.name)
        if metadata['observation_names'] != BASE_OBSERVATIONS:
            raise ValueError('Base observation layout changed; verify QuietPaw compatibility')
        if not np.isclose(env.step_dt, checkpoint['infos']['quietpaw']['step_dt'], rtol=0, atol=1e-9):
            raise ValueError('Environment policy period differs from the saved QuietPaw period')
        sdk_names = [f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
                     for joint in ('hip', 'thigh', 'calf')]
        model = env.sim.mj_model
        metadata.update(
            version=1, policy_family='quietpaw', observation_size=48,
            observation_names=QUIETPAW_OBSERVATIONS, noise_condition=condition,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            policy_to_sdk=[sdk_names.index(name) for name in metadata['joint_names']],
            joint_limits=[model.jnt_range[model.joint('robot/' + name).id].tolist()
                          for name in metadata['joint_names']],
            step_dt=env.step_dt, phase_period=cfg.observations['actor'].terms['phase'].params['period'],
            clip_actions=agent['clip_actions'], normalization_included=True,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        exported = actor.as_onnx(verbose=False).eval()
        torch.onnx.export(exported, exported.get_dummy_inputs(), str(args.output / 'policy.onnx'),
                          input_names=exported.input_names, output_names=exported.output_names,
                          opset_version=18, dynamo=False, external_data=False)
        metadata['onnx_sha256'] = hashlib.sha256((args.output / 'policy.onnx').read_bytes()).hexdigest()

        # Compare host ONNX and real-state mapping against the original saved
        # RSL actor on MuJoCo observations, including both ends of the condition.
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent['clip_actions'])
        wrapped.reset()
        observations, actions, samples = [], [], []
        with torch.inference_mode():
            for index in range(120):
                quietness = (0.0, 0.5, 1.0)[index // 40]
                base = wrapped.get_observations()['actor'].cpu()
                if base.shape != (1, 47):
                    raise ValueError('Expected 47 base observations before the QuietPaw condition')
                cost_limit = cost_limit_from_quietness(quietness, condition)
                obs = torch.cat((base, torch.full((1, 1), cost_limit)), dim=1)
                action = actor(TensorDict({'actor': obs}, batch_size=[1]))
                if action.shape != (1, 12) or not torch.isfinite(action).all():
                    raise ValueError('Invalid QuietPaw actor output')
                observations.append(obs.numpy())
                actions.append(action.numpy())
                robot = env.scene['robot'].data
                samples.append(dict(q=robot.joint_pos[0].cpu().tolist(), dq=robot.joint_vel[0].cpu().tolist(),
                                    quaternion=robot.root_link_quat_w[0].cpu().tolist(),
                                    gyro=robot.root_link_ang_vel_b[0].cpu().tolist(),
                                    step=int(env.episode_length_buf[0]), quietness=quietness))
                wrapped.step(action.to(env.device))
        np.savez(args.output / 'parity.npz', observations=np.concatenate(observations),
                 actions=np.concatenate(actions))
        (args.output / 'samples.json').write_text(json.dumps(samples) + '\n')
        (args.output / 'policy.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        print('Exported QuietPaw actor (48 -> 12), normalization and saved cost limits to', args.output)
    finally:
        env.close()


if __name__ == '__main__':
    main()
