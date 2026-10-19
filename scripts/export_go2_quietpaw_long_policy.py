"""QuietPaw LongStride 모델 준비. Docker ML 환경에서 실행하며 로봇에 연결하지 않는다.

PT에 저장된 stride/noise 설정을 복원해 실측 관측 매핑 검증용 샘플을 만든다.
--source-onnx가 있으면 원본을 복사하고, 없으면 PT에서 ONNX를 내보낸다.
호출자는 check_go2_real_policy.py로 출력/관측 매핑을 검증한 뒤 결과를 사용한다.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from tensordict import TensorDict

from export_go2_quietpaw_policy import load_actor
from go2_autonomous_driving.real.real_quietpaw import (
    QUIETPAW_OBSERVATIONS, cost_limit_from_quietness,
)


# ============================================================
# 1. 학습 설정 복원 및 가져온 ONNX 메타데이터 비교
# ============================================================
def restore_environment(checkpoint):
    from mjlab.tasks.registry import load_env_cfg
    import src.locomotion_rl  # noqa: F401 -- task registration
    from src.locomotion_rl.go2_quietpaw_long_stride.step_01_environment.quietpaw_env import QuietPawEnv
    from src.locomotion_rl.go2_quietpaw_long_stride.step_02_learning.costs import NoiseCostCfg
    from src.locomotion_rl.go2_quietpaw_long_stride.step_02_learning.mdp.noise_condition import NoiseConditionCfg
    from src.locomotion_rl.go2_quietpaw_long_stride.step_02_learning.stride import StrideCfg

    saved = checkpoint["infos"]["quietpaw"]
    if "stride" not in saved:
        raise ValueError("LongStride의 stride 설정이 저장된 PT가 필요합니다")
    cfg = load_env_cfg("Unitree-Go2-QuietPaw-LongStride-Flat", play=True)
    cfg.scene.num_envs = 1
    cfg.stride = StrideCfg(**saved["stride"])
    cfg.noise_cost = NoiseCostCfg(**saved["noise_cost"])
    cfg.noise_condition = NoiseConditionCfg(**saved["noise_condition"])
    cfg.noise_condition.quietness = 0.5
    # QuietPawEnv가 stride 설정을 보상과 actor/critic 위상에 함께 적용한다.
    return QuietPawEnv(cfg=cfg, device="cuda:0")


def check_source_metadata(path, metadata):
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    saved = {item.key: item.value for item in model.metadata_props}
    for name in ("joint_names", "observation_names"):
        if saved.get(name, "").split(",") != metadata[name]:
            raise ValueError(f"가져온 ONNX의 {name}이 학습 환경과 다릅니다")
    # mjlab의 ONNX 메타데이터는 배열을 소수점 세 자리 문자열로 저장한다.
    for name in ("default_joint_pos", "joint_stiffness", "joint_damping", "action_scale"):
        values = np.asarray([float(v) for v in saved[name].split(",")])
        expected = np.asarray(metadata[name]).reshape(-1)
        if values.shape != expected.shape or not np.allclose(values, expected, rtol=0, atol=0.00051):
            raise ValueError(f"가져온 ONNX의 {name}이 학습 환경과 다릅니다")
    scalars = {}
    if metadata.get("policy_family") == "quietpaw":
        scalars.update(
            policy_dt=metadata["step_dt"],
            quietpaw_cost_limit_min=metadata["noise_condition"]["min_cost_limit"],
            quietpaw_cost_limit_max=metadata["noise_condition"]["max_cost_limit"],
        )
    if "stride" in metadata:
        scalars["stride_gait_period"] = metadata["phase_period"]
    for name, expected in scalars.items():
        if not np.isclose(float(saved[name]), expected, rtol=0, atol=1e-9):
            raise ValueError(f"가져온 ONNX의 {name}이 PT 설정과 다릅니다")


# ============================================================
# 2. 모델 및 실로봇 제어 정보 저장
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-onnx", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actor, agent, condition = load_actor(checkpoint)

    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import get_base_metadata

    env = restore_environment(checkpoint)
    try:
        saved = checkpoint["infos"]["quietpaw"]
        metadata = get_base_metadata(env, args.checkpoint.name)
        if metadata["observation_names"] != QUIETPAW_OBSERVATIONS:
            raise ValueError("48개 QuietPaw Flat 관측 순서가 아닙니다")
        if not np.isclose(env.step_dt, saved["step_dt"], rtol=0, atol=1e-9):
            raise ValueError("현재 환경과 PT의 제어 주기가 다릅니다")
        sdk_names = [
            f"{leg}_{joint}_joint"
            for leg in ("FR", "FL", "RR", "RL")
            for joint in ("hip", "thigh", "calf")
        ]
        model = env.sim.mj_model
        metadata.update(
            version=1,
            policy_family="quietpaw",
            policy_variant="quietpaw_long_stride",
            observation_size=48,
            noise_condition=condition,
            stride=deepcopy(saved["stride"]),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            policy_to_sdk=[sdk_names.index(name) for name in metadata["joint_names"]],
            joint_limits=[
                model.jnt_range[model.joint("robot/" + name).id].tolist()
                for name in metadata["joint_names"]
            ],
            step_dt=env.step_dt,
            phase_period=env.cfg.observations["actor"].terms["phase"].params["period"],
            clip_actions=agent["clip_actions"],
            normalization_included=True,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        policy_path = args.output / "policy.onnx"
        if args.source_onnx is not None:
            check_source_metadata(args.source_onnx, metadata)
            shutil.copyfile(args.source_onnx, policy_path)
            metadata["source_onnx_sha256"] = hashlib.sha256(args.source_onnx.read_bytes()).hexdigest()
        else:
            exported = actor.as_onnx(verbose=False).eval()
            torch.onnx.export(
                exported, exported.get_dummy_inputs(), str(policy_path),
                input_names=exported.input_names, output_names=exported.output_names,
                opset_version=18, dynamo=False, external_data=False,
            )
        metadata["onnx_sha256"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()

        # 세 가지 소음 조건에서 실제 학습 환경의 관측과 PT 출력을 저장한다.
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
        wrapped.reset()
        observations, actions, samples = [], [], []
        with torch.inference_mode():
            for index in range(120):
                quietness = (0.0, 0.5, 1.0)[index // 40]
                env.noise_condition.set_quietness(quietness)
                obs = env.observation_manager.compute(update_history=True)["actor"].cpu()
                if obs.shape != (1, 48):
                    raise ValueError("Expected 48 observations")
                expected_cost = cost_limit_from_quietness(quietness, condition)
                if not np.isclose(obs[0, -1].item(), expected_cost):
                    raise ValueError("학습 환경의 소음 조건 입력이 다릅니다")
                action = actor(TensorDict({"actor": obs}, batch_size=[1]))
                if action.shape != (1, 12) or not torch.isfinite(action).all():
                    raise ValueError("Invalid actor output")
                observations.append(obs.numpy())
                actions.append(action.numpy())
                robot = env.scene["robot"].data
                samples.append(dict(
                    q=robot.joint_pos[0].cpu().tolist(),
                    dq=robot.joint_vel[0].cpu().tolist(),
                    quaternion=robot.root_link_quat_w[0].cpu().tolist(),
                    gyro=robot.root_link_ang_vel_b[0].cpu().tolist(),
                    step=int(env.episode_length_buf[0]),
                    quietness=quietness,
                ))
                wrapped.step(action.to(env.device))
        np.savez(args.output / "parity.npz", observations=np.concatenate(observations),
                 actions=np.concatenate(actions))
        (args.output / "samples.json").write_text(json.dumps(samples) + "\n")
        (args.output / "policy.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        print(f"Prepared LongStride: dt={metadata['step_dt']}, phase={metadata['phase_period']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
