"""Run the existing Go2 policy in mjlab; ROS stays in its own Python environment."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import socket
import time
import uuid

from .core import CommandWatchdog


def box_surface_points(boxes, position, scan_range):
    """Ground-truth virtual scan, without ray occlusion, for the first simulation."""
    points = []
    for box in boxes:
        x, y, z = box["position"]
        hx, hy, hz = box["half_size"]
        for fraction in range(11):
            f = -1 + fraction / 5
            for level in (0.1, 0.3, 0.5, 0.7, 0.9):
                height = z - hz + 2 * hz * level
                points.extend(((x - hx, y + hy * f, height), (x + hx, y + hy * f, height),
                               (x + hx * f, y - hy, height), (x + hx * f, y + hy, height)))
    return [point for point in points
            if math.hypot(point[0] - position[0], point[1] - position[1]) <= scan_range]


class SimulationTransport:
    def __init__(self, config):
        self.config = config
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", config["command_port"]))
        self.receiver.setblocking(False)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.watchdog = CommandWatchdog(
            config["command_timeout"], config["max_forward_speed"],
            config["max_lateral_speed"], config["max_yaw_rate"],
        )
        self.next_telemetry = 0.0
        self.session = uuid.uuid4().hex
        self.reset_counter = 0
        self.previous_episode_step = 0

    def command(self):
        now = time.monotonic()
        for _ in range(32):
            try:
                packet, _ = self.receiver.recvfrom(4096)
            except BlockingIOError:
                break
            self.watchdog.receive(packet, now)
        return self.watchdog.sample(now)

    def telemetry(self, env, force=False):
        now = time.monotonic()
        if not force and now < self.next_telemetry:
            return
        self.next_telemetry = now + 0.1
        robot = env.scene["robot"].data
        episode_step = int(env.episode_length_buf[0].item())
        if episode_step < self.previous_episode_step:
            self.reset_counter += 1
        self.previous_episode_step = episode_step
        position = robot.root_link_pos_w[0].detach().cpu().tolist()
        data = {
            "version": 1, "monotonic": now,
            "session": self.session, "reset_counter": self.reset_counter,
            "position": position,
            "quaternion_wxyz": robot.root_link_quat_w[0].detach().cpu().tolist(),
            "linear_velocity_body": robot.root_link_lin_vel_b[0].detach().cpu().tolist(),
            "angular_velocity_body": robot.root_link_ang_vel_b[0].detach().cpu().tolist(),
            "points": box_surface_points(self.config["obstacles"], position, self.config["scan_range"]),
        }
        self.sender.sendto(json.dumps(data, allow_nan=False).encode(),
                           ("127.0.0.1", self.config["telemetry_port"]))

    def close(self):
        self.receiver.close()
        self.sender.close()


def run(args):
    # Imported only in the mjlab process, not in the ROS nodes.
    import mujoco
    import torch
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
    from mjlab.utils.torch import configure_torch_backends
    import src.tasks  # noqa: F401

    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config/simulation.json"
        if not config_path.is_file():
            from ament_index_python.packages import get_package_share_directory
            config_path = Path(get_package_share_directory("go2_autonomous_driving")) / "config/simulation.json"
    config = json.loads(config_path.read_text())
    checkpoint = args.checkpoint or Path(config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    actor = state["actor_state_dict"]
    if tuple(actor["mlp.0.weight"].shape) != (512, 47) or tuple(actor["mlp.6.weight"].shape) != (12, 128):
        raise ValueError("This simulation requires the existing Go2 Flat 47-input/12-action policy")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    configure_torch_backends()
    transport = SimulationTransport(config)

    class ExternalCommand(UniformVelocityCommand):
        def compute(self, dt):
            super().compute(dt)
            command = transport.command()
            self.vel_command_b[:] = torch.tensor(
                [command.vx, command.vy, command.wz], device=self.device
            )

        def create_gui(self, *unused_args, **unused_kwargs):
            # Keyboard waypoints own this command; viewer sliders must not override it.
            pass

    class ExternalCommandCfg(UniformVelocityCommandCfg):
        def build(self, env):
            return ExternalCommand(self, env)

    cfg = load_env_cfg("Unitree-Go2-Flat", play=True)
    cfg.scene.num_envs = 1
    cfg.commands["twist"] = ExternalCommandCfg(
        entity_name="robot", resampling_time_range=(1e6, 1e6),
        heading_command=False, rel_standing_envs=0.0, debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-config["max_forward_speed"], config["max_forward_speed"]),
            lin_vel_y=(-config["max_lateral_speed"], config["max_lateral_speed"]),
            ang_vel_z=(-config["max_yaw_rate"], config["max_yaw_rate"]), heading=None,
        ),
    )

    def obstacles_spec():
        spec = mujoco.MjSpec()
        for index, box in enumerate(config["obstacles"]):
            spec.worldbody.add_geom(
                name=f"navigation_box_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=box["position"], size=box["half_size"],
                rgba=[0.8, 0.3, 0.2, 1.0], contype=1, conaffinity=1,
            )
        return spec

    if config["obstacles"]:
        cfg.scene.entities["navigation_obstacles"] = EntityCfg(spec_fn=obstacles_spec)
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = load_runner_cls("Unitree-Go2-Flat")(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        wrapped.reset()
        print(f"Loaded {checkpoint}, iter={state.get('iter')}; navigation is initially stopped", flush=True)

        class TelemetryPolicy:
            def __call__(self, obs):
                transport.telemetry(env)
                return policy(obs)

        telemetry_policy = TelemetryPolicy()
        if args.viewer == "viser":
            import viser
            from mjlab.viewer import ViserPlayViewer
            viewer = ViserPlayViewer(wrapped, telemetry_policy,
                                     viser_server=viser.ViserServer(port=args.viewer_port))
            viewer.run(num_steps=args.steps)
        else:
            deadline = time.monotonic()
            index = 0
            with torch.inference_mode():
                while args.steps is None or index < args.steps:
                    obs = wrapped.get_observations()
                    wrapped.step(telemetry_policy(obs))
                    index += 1
                    deadline += env.step_dt
                    if not args.fast:
                        time.sleep(max(0.0, deadline - time.monotonic()))
            transport.telemetry(env, force=True)
    finally:
        if env is not None:
            env.close()
        transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--viewer", choices=("viser", "none"), default="viser")
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--steps", type=int, help="Stop after this many policy steps")
    parser.add_argument("--fast", action="store_true", help="Headless benchmark without real-time pacing")
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
