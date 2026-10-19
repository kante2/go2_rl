"""Run a Go2 baseline / QuietPaw / LongStride policy in MuJoCo over UDP.

Entry flow: main() -> init() -> run_simulation() -> close().
Keep this module in the original package: it uses ``from ..core``.
기본 모델 폴더는 config/model_pt/quiet_paw_long이며 가장 큰 학습 번호를 선택한다.
시뮬레이션에서는 PT를 직접 불러온다. ONNX 준비는 필요하지 않다.
"""

import argparse
from functools import partial
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import socket
import time
import uuid

from ..core import CommandWatchdog
from ..model_config import read_model_config, is_model_config


MODELS = {
    "baseline": ("baseline", "Unitree-Go2-Flat"),
    "quietpaw": ("quietpaw", "Unitree-Go2-QuietPaw-Flat"),
    "quietpaw_long": ("quiet_paw_long", "Unitree-Go2-QuietPaw-LongStride-Flat"),
}


# ============================================================
# 1. 실행 옵션과 설정 파일
# ============================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", choices=tuple(MODELS),
                        help="config/model_pt 안에서 사용할 모델 종류")
    parser.add_argument("--device")
    parser.add_argument("--quietness", type=float,
                        help="0: 느슨한 소음 제약, 1: 엄격한 제약 (config보다 우선)")
    parser.add_argument("--viewer", choices=("viser", "none"), default="viser")
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--monitor-port", type=int, default=9910,
                        help="보행 UI 측정 UDP 포트. 0이면 측정 비활성화")
    parser.add_argument("--steps", type=int, help="Number of policy steps")
    parser.add_argument("--fast", action="store_true",
                        help="Headless execution without real-time pacing")
    args = parser.parse_args(argv)
    if not 0 <= args.monitor_port <= 65535:
        parser.error("--monitor-port must be in [0, 65535]")
    return args


def select_checkpoint(model_dir):
    """실로봇 실행기와 같이 수정 시간 대신 학습 반복 번호로 선택한다."""
    candidates = []
    for path in model_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"model_<번호>.pt가 없습니다: {model_dir}")
    return max(candidates)[1]


def load_config(args):
    config_path, config = read_model_config(
        Path(__file__).resolve().parents[2] / "config", args.model, args.config,
    )
    print(f"Config: {config_path}", flush=True)
    profile = is_model_config(config)

    # 새 이름 udp_port를 우선 사용하고 기존 JSON의 telemetry_port도 지원한다.
    if "udp_port" not in config:
        config["udp_port"] = config["telemetry_port"]

    # model_dir는 JSON 기준, checkpoint는 model_dir 기준이다.
    # CLI로 지정한 checkpoint는 현재 작업 폴더 기준이다.
    model_dir = (config_path.parent / config["model_dir"]).resolve()
    if args.model is not None and not profile:
        model_dir = config_path.parent / "model_pt" / MODELS[args.model][0]
        # 종류를 바꾸면 기존 config의 다른 종류 checkpoint 지정은 사용하지 않는다.
        config["checkpoint"] = None
    config["model"] = args.model or config.get("model", "quietpaw_long")
    if config["model"] not in MODELS:
        raise ValueError(f"Unknown model: {config['model']}")
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve()
    elif config.get("checkpoint"):
        checkpoint = (model_dir / config["checkpoint"]).resolve()
    else:
        checkpoint = select_checkpoint(model_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    quietness = args.quietness if args.quietness is not None else config["quietness"]
    if type(quietness) not in (int, float) or not math.isfinite(quietness) or not 0 <= quietness <= 1:
        raise ValueError("quietness must be finite and in [0, 1]")
    config["quietness"] = float(quietness)
    return config, checkpoint


def check_policy_shape(model_state, model="quietpaw_long"):
    infos = model_state.get("infos") or {}
    saved = infos.get("quietpaw")
    if model == "baseline":
        actor = model_state["actor_state_dict"]
        input_shape = tuple(actor["mlp.0.weight"].shape)
        output_shape = tuple(actor["mlp.6.weight"].shape)
        if saved is not None or input_shape != (512, 47) or output_shape != (12, 128):
            raise ValueError("baseline에는 Go2 Flat 47-input/12-action PT가 필요합니다")
        return
    if saved is None or "quietpaw_agent" not in infos:
        raise ValueError("QuietPaw / LongStride 학습 설정이 저장된 PT가 필요합니다")
    if ("stride" in saved) != (model == "quietpaw_long"):
        raise ValueError("선택한 QuietPaw 종류와 PT의 LongStride 설정이 다릅니다")
    actor_cfg = infos["quietpaw_agent"]["actor"]
    hidden_dims = actor_cfg["hidden_dims"]
    if actor_cfg["class_name"] != "MLPModel" or not hidden_dims:
        raise ValueError("Expected a QuietPaw MLPModel actor")
    actor = model_state["actor_state_dict"]
    input_shape = tuple(actor["mlp.0.weight"].shape)
    output_shape = tuple(actor[f"mlp.{2 * len(hidden_dims)}.weight"].shape)
    if input_shape != (hidden_dims[0], 48) or output_shape != (12, hidden_dims[-1]):
        raise ValueError(
            "This runner requires a QuietPaw "
            "48-input/12-action policy"
        )


# ============================================================
# 2. 초기화: UDP, 환경, 모델, 추론 함수 준비
# ============================================================
def init(args):
    # mjlab 프로세스에서만 로딩한다. ROS 노드에서는 이 함수를 호출하지 않는다.
    import torch
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from mjlab.utils.torch import configure_torch_backends
    import src.locomotion_rl  # noqa: F401 -- Go2 task registration

    config, checkpoint = load_config(args)
    print(f"Loading {config['model']}: {checkpoint}", flush=True)
    model_state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model = config["model"]
    task_id = MODELS[model][1]
    check_policy_shape(model_state, model)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    configure_torch_backends()

    # 여러 함수가 함께 사용하는 값은 하나의 딕셔너리로 전달한다.
    state = {
        "config": config,
        "env": None,
        "receiver": None,
        "sender": None,
        "next_udp_state_time": 0.0,
        "session": uuid.uuid4().hex,
        "reset_counter": 0,
        "previous_episode_step": 0,
        "monitor": None,
    }

    try:
        # UDP: 브리지 명령 수신 / MuJoCo 상태 송신
        state["receiver"] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state["receiver"].bind(("127.0.0.1", config["command_port"]))
        state["receiver"].setblocking(False)
        state["sender"] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state["watchdog"] = CommandWatchdog(
            config["command_timeout"],
            config["max_forward_speed"],
            config["max_lateral_speed"],
            config["max_yaw_rate"],
        )

        # PT에 저장된 네트워크 크기, 소음 비용 설정, 보행 주기를 복원한다.
        cfg = load_env_cfg(task_id, play=True)
        env_class = ManagerBasedRlEnv
        if model == "baseline":
            agent_cfg = asdict(load_rl_cfg(task_id))
        else:
            if model == "quietpaw_long":
                from src.locomotion_rl.go2_quietpaw_long_stride.step_01_environment.quietpaw_env import QuietPawEnv
                from src.locomotion_rl.go2_quietpaw_long_stride.step_04_evaluation.checkpoint import load_evaluation_settings
            else:
                from src.locomotion_rl.go2_quietpaw.step_01_environment.quietpaw_env import QuietPawEnv
                from src.locomotion_rl.go2_quietpaw.step_04_evaluation.checkpoint import load_evaluation_settings
            agent_cfg = load_evaluation_settings(cfg, checkpoint)
            cfg.noise_condition.quietness = config["quietness"]
            env_class = QuietPawEnv
        cfg.scene.num_envs = 1
        cfg.commands["twist"] = UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(1e6, 1e6),
            heading_command=False,
            rel_standing_envs=0.0,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-config["max_forward_speed"], config["max_forward_speed"]),
                lin_vel_y=(-config["max_lateral_speed"], config["max_lateral_speed"]),
                ang_vel_z=(-config["max_yaw_rate"], config["max_yaw_rate"]),
                heading=None,
            ),
        )

        if config["obstacles"]:
            cfg.scene.entities["navigation_obstacles"] = EntityCfg(
                spec_fn=partial(create_mujoco_obstacles, config["obstacles"])
            )

        # QuietPaw에서는 마지막 48번째 관측값(cost_limit)을 환경이 만든다.
        if args.monitor_port:
            from src.go2_gait_monitor.go2_gait_monitor.mujoco import add_contact_sensors
            add_contact_sensors(cfg)
        env = env_class(cfg=cfg, device=device)
        state["env"] = env
        saved_dt = model_state.get("infos", {}).get("quietpaw", {}).get("step_dt", env.step_dt)
        if not math.isclose(env.step_dt, saved_dt, rel_tol=0, abs_tol=1e-9):
            raise ValueError("시뮬레이션과 학습 체크포인트의 제어 주기가 다릅니다")

        # 사용자 정의 subclass 대신 기존 명령 객체에 함수를 연결한다.
        # 명령 갱신은 기존처럼 mjlab이 compute()를 호출하는 시점에 실행된다.
        connect_udp_command_to_mujoco(state)

        # 모델과 추론 함수 로딩
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
        runner = load_runner_cls(task_id)(
            wrapped, agent_cfg, device=device
        )
        runner.load(
            str(checkpoint), load_cfg={"actor": True},
            strict=True, map_location=device,
        )
        state["wrapped"] = wrapped
        state["policy"] = runner.get_inference_policy(device=device)
        wrapped.reset()
        if args.monitor_port:
            from src.go2_gait_monitor.go2_gait_monitor.mujoco import MujocoMonitor
            state["monitor"] = MujocoMonitor(env, config, checkpoint, args.monitor_port)
            print(f"Gait monitor: UDP 127.0.0.1:{args.monitor_port}", flush=True)
        condition_status = "47 -> 12"
        if model != "baseline":
            condition_status = (
                f"48 -> 12, quietness={config['quietness']:g}, "
                f"cost_limit={env.noise_condition.cost_limit[0].item():g}"
            )
        print(
            f"Loaded {checkpoint}, iter={model_state.get('iter')}, "
            f"model={model}, {condition_status}, "
            f"phase_period={cfg.observations['actor'].terms['phase'].params['period']:g}, "
            f"step_dt={env.step_dt:g}",
            flush=True,
        )
        return state
    except BaseException:
        # 초기화 도중 실패하거나 Ctrl+C가 들어와도 생성된 자원을 정리한다.
        close(state)
        raise


# ============================================================
# 3. UDP -> MuJoCo: 속도 명령 수신 및 환경에 적용
# ============================================================
def receive_udp_command_for_mujoco(state):
    now = time.monotonic()
    for _ in range(32):
        try:
            packet, _ = state["receiver"].recvfrom(4096)
        except BlockingIOError:
            break
        state["watchdog"].receive(packet, now)
    return state["watchdog"].sample(now)


def update_mujoco_command_from_udp(state, command_term, original_compute, dt):
    """mjlab의 기존 계산 후, 외부에서 받은 목표 속도로 덮어쓴다."""
    import torch

    original_compute(dt)
    command = receive_udp_command_for_mujoco(state)
    command_term.vel_command_b[:] = torch.tensor(
        [command.vx, command.vy, command.wz], device=command_term.device
    )


def disable_velocity_gui(*args, **kwargs):
    """뷰어 속도 슬라이더가 UDP 명령에 개입하지 않도록 한다."""
    pass


def connect_udp_command_to_mujoco(state):
    """현재 환경의 twist 객체 하나에만 함수를 연결한다."""
    command_term = state["env"].command_manager.get_term("twist")
    original_compute = command_term.compute
    command_term.compute = partial(
        update_mujoco_command_from_udp, state, command_term, original_compute
    )
    command_term.create_gui = disable_velocity_gui


# ============================================================
# 4. MuJoCo 장애물 및 가상 점군
# ============================================================
def create_mujoco_obstacles(boxes):
    import mujoco

    spec = mujoco.MjSpec()
    for index, box in enumerate(boxes):
        spec.worldbody.add_geom(
            name=f"navigation_box_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box["position"],
            size=box["half_size"],
            rgba=[0.8, 0.3, 0.2, 1.0],
            contype=1,
            conaffinity=1,
        )
    return spec


def create_virtual_scan_points(boxes, robot_position, scan_range):
    """박스의 실제 위치로 점군을 만든다. LiDAR 가림/노이즈는 모델링하지 않는다."""
    points = []
    for box in boxes:
        x, y, z = box["position"]
        hx, hy, hz = box["half_size"]
        for fraction in range(11):
            f = -1 + fraction / 5
            for level in (0.1, 0.3, 0.5, 0.7, 0.9):
                height = z - hz + 2 * hz * level
                points.extend((
                    (x - hx, y + hy * f, height),
                    (x + hx, y + hy * f, height),
                    (x + hx * f, y - hy, height),
                    (x + hx * f, y + hy, height),
                ))
    return [
        point for point in points
        if math.hypot(point[0] - robot_position[0],
                      point[1] - robot_position[1]) <= scan_range
    ]


# ============================================================
# 5. MuJoCo -> UDP: 상태 패킷 생성 및 송신
# ============================================================
def build_udp_state_from_mujoco(state, now):
    env = state["env"]
    config = state["config"]
    robot = env.scene["robot"].data

    episode_step = int(env.episode_length_buf[0].item())
    if episode_step < state["previous_episode_step"]:
        state["reset_counter"] += 1
    state["previous_episode_step"] = episode_step

    position = robot.root_link_pos_w[0].detach().cpu().tolist()
    return {
        "version": 1,
        "monotonic": now,
        "session": state["session"],
        "reset_counter": state["reset_counter"],
        "position": position,
        "quaternion_wxyz": robot.root_link_quat_w[0].detach().cpu().tolist(),
        "linear_velocity_body": robot.root_link_lin_vel_b[0].detach().cpu().tolist(),
        "angular_velocity_body": robot.root_link_ang_vel_b[0].detach().cpu().tolist(),
        "points": create_virtual_scan_points(
            config["obstacles"], position, config["scan_range"]
        ),
    }


def send_mujoco_state_to_udp(state, force=False):
    now = time.monotonic()
    if not force and now < state["next_udp_state_time"]:
        return
    state["next_udp_state_time"] = now + 0.1

    data = build_udp_state_from_mujoco(state, now)
    packet = json.dumps(data, allow_nan=False).encode()
    state["sender"].sendto(packet, ("127.0.0.1", state["config"]["udp_port"]))


# ============================================================
# 6. 정책 추론과 시뮬레이션 실행
# ============================================================
def infer_action_and_send_mujoco_state(state, obs):
    if state.get("monitor") is not None:
        state["monitor"].publish()
    send_mujoco_state_to_udp(state)
    return state["policy"](obs)


def run_simulation(args, state):
    import torch

    # partial은 state를 미리 연결해, 뷰어가 함수(obs) 형태로 호출하게 한다.
    policy_step = partial(infer_action_and_send_mujoco_state, state)
    wrapped = state["wrapped"]

    with torch.inference_mode():
        if args.viewer == "viser":
            import viser
            from mjlab.viewer import ViserPlayViewer

            # 웹 서버 실행
            server = viser.ViserServer(port=args.viewer_port)
            # mujuco 환경과, 학습 모델을 뷰어에 연결하고 실행
            try:
                viewer = ViserPlayViewer(wrapped, policy_step, viser_server=server)
                viewer.run(num_steps=args.steps)
            finally:
                server.stop()
        else:
            deadline = time.monotonic()
            index = 0
            while args.steps is None or index < args.steps:
                obs = wrapped.get_observations()
                actions = policy_step(obs)
                wrapped.step(actions)
                index += 1
                deadline += state["env"].step_dt
                if not args.fast:
                    time.sleep(max(0.0, deadline - time.monotonic()))
            send_mujoco_state_to_udp(state, force=True)
            if state.get("monitor") is not None:
                state["monitor"].publish(force=True)


# ============================================================
# 7. 종료 및 시작점
# ============================================================
def close(state):
    if state is None:
        return
    try:
        if state.get("monitor") is not None:
            state["monitor"].close()
        if state.get("env") is not None:
            state["env"].close()
    finally:
        for name in ("receiver", "sender"):
            if state.get(name) is not None:
                state[name].close()


def main():
    args = parse_args()
    state = None
    try:
        state = init(args)
        run_simulation(args, state)
    except KeyboardInterrupt:
        pass
    finally:
        close(state)


if __name__ == "__main__":
    main()
