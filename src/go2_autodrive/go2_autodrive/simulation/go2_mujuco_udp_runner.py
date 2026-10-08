"""Run a trained Go2 policy in MuJoCo and exchange commands/state over UDP.

Entry flow: main() -> init() -> run_simulation() -> close().
Keep this module in the original package: it uses ``from ..core``.
"""

import argparse
from dataclasses import asdict
from functools import partial
import json
import math
from pathlib import Path
import socket
import time
import uuid

from ..core import CommandWatchdog


# ============================================================
# 1. 실행 옵션과 설정 파일
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--viewer", choices=("viser", "none"), default="viser")
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--steps", type=int, help="Number of policy steps")
    parser.add_argument("--fast", action="store_true",
                        help="Headless execution without real-time pacing")
    return parser.parse_args()


def load_config(args):
    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config/simulation.json"
        if not config_path.is_file():
            from ament_index_python.packages import get_package_share_directory

            config_path = (
                Path(get_package_share_directory("go2_autonomous_driving"))
                / "config/simulation.json"
            )

    config = json.loads(config_path.read_text(encoding="utf-8"))

    # 새 이름 udp_port를 우선 사용하고 기존 JSON의 telemetry_port도 지원한다.
    if "udp_port" not in config:
        config["udp_port"] = config["telemetry_port"]

    # CLI 경로는 현재 작업 폴더 기준, JSON 경로는 설정 파일 기준이다.
    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else config_path.parent / config["checkpoint"]
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    return config, checkpoint


def check_policy_shape(model_state):
    actor = model_state["actor_state_dict"]
    input_shape = tuple(actor["mlp.0.weight"].shape)
    output_shape = tuple(actor["mlp.6.weight"].shape)
    if input_shape != (512, 47) or output_shape != (12, 128):
        raise ValueError(
            "This runner requires the existing Go2 Flat "
            "47-input/12-action policy"
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
    import src.tasks  # noqa: F401 -- Go2 task registration

    config, checkpoint = load_config(args)
    model_state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    check_policy_shape(model_state)
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

        # 단일 Go2 환경 구성
        cfg = load_env_cfg("Unitree-Go2-Flat", play=True)
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

        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        state["env"] = env

        # 사용자 정의 subclass 대신 기존 명령 객체에 함수를 연결한다.
        # 명령 갱신은 기존처럼 mjlab이 compute()를 호출하는 시점에 실행된다.
        connect_udp_command_to_mujoco(state)

        # 모델과 추론 함수 로딩
        agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = load_runner_cls("Unitree-Go2-Flat")(
            wrapped, asdict(agent_cfg), device=device
        )
        runner.load(
            str(checkpoint), load_cfg={"actor": True},
            strict=True, map_location=device,
        )
        state["wrapped"] = wrapped
        state["policy"] = runner.get_inference_policy(device=device)
        wrapped.reset()
        print(f"Loaded {checkpoint}, iter={model_state.get('iter')}", flush=True)
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


# ============================================================
# 7. 종료 및 시작점
# ============================================================
def close(state):
    if state is None:
        return
    try:
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