"""Run the existing Go2 policy in mjlab; ROS stays in its own Python environment."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import socket
import time
import uuid

from ..core import CommandWatchdog


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


class NativeWindow:
    """Read-only MuJoCo window that mirrors env 0 and follows the robot; runs beside viser.

    viewer.sync() blocks until a frame is rendered (~0.5 s with software GL in Docker), so a
    background thread renders the latest snapshot and the simulation loop never waits for it.
    """

    def __init__(self, env, fps=30.0):
        import threading
        import mujoco
        import mujoco.viewer
        self.mujoco = mujoco
        self.env = env
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)  # private copy; never written back to the sim
        self.period = 1.0 / fps
        self.next_snapshot = 0.0
        self.snapshot = None
        self.snapshot_lock = threading.Lock()
        self.stopped = threading.Event()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                                   show_left_ui=False, show_right_ui=False)
        free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if free:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = int(self.model.jnt_bodyid[free[0]])
            self.viewer.cam.distance, self.viewer.cam.elevation = 2.5, -20.0
        self.thread = threading.Thread(target=self.render_loop, daemon=True)
        self.thread.start()

    def sync(self):
        """Called from the simulation loop: only hand over the latest state."""
        now = time.monotonic()
        if now < self.next_snapshot:
            return
        self.next_snapshot = now + self.period
        sim_data = self.env.sim.data
        state = [sim_data.qpos[0].cpu().numpy(), sim_data.qvel[0].cpu().numpy()]
        if self.model.nmocap:
            state += [sim_data.mocap_pos[0].cpu().numpy(), sim_data.mocap_quat[0].cpu().numpy()]
        with self.snapshot_lock:
            self.snapshot = state

    def render_loop(self):
        while not self.stopped.is_set() and self.viewer.is_running():
            with self.snapshot_lock:
                state, self.snapshot = self.snapshot, None
            if state is None:
                time.sleep(0.005)
                continue
            with self.viewer.lock():
                self.data.qpos[:], self.data.qvel[:] = state[0], state[1]
                if self.model.nmocap:
                    self.data.mocap_pos[:], self.data.mocap_quat[:] = state[2], state[3]
                self.mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
        self.viewer.close()


def quat_rotate_inverse(quat_wxyz, vectors):
    """World -> body for unit quaternions: v_b = q^-1 v q, vectors (..., 3)."""
    w, u = quat_wxyz[0], quat_wxyz[1:].expand_as(vectors)
    return (vectors * (2.0 * w * w - 1.0) - 2.0 * w * u.cross(vectors, dim=-1)
            + 2.0 * u * (u * vectors).sum(-1, keepdim=True))


class NoiseStateSender:
    """Stream real-robot-like state every physics step for the noise model and the recorder.

    "low" matches db_prediction low_vector(): foot_force4, quaternion wxyz4, gyro3, q12, dq12,
    legs in Unitree FR, FL, RR, RL order. Foot force is the MuJoCo contact force norm [N] plus the
    real sensor's airborne reading, since the real Go2 foot sensor reads ~1:1 with Newtons
    (walking sum ~148 vs 15.2 kg x g) and ~8-12 with the foot in the air.
    "raw" carries the rest of what the real recorder stores (LowState, LowCmd, SportModeState
    equivalents) plus simulation-only values, all in the same leg/joint order.
    """

    LEGS = ("FR", "FL", "RR", "RL")
    AIRBORNE_FOOT_FORCE = (8.0, 10.0, 12.0, 9.0)  # FR, FL, RR, RL, from db_collector recordings
    RAW_FIELDS = (("acc", 3), ("tau", 12), ("ddq", 12), ("q_target", 12), ("base_pos_w", 3), ("base_lin_vel_w", 3),
                  ("yaw_rate_w", 1), ("foot_pos_b", 12), ("foot_vel_b", 12), ("foot_force_vec", 12),
                  ("command", 3), ("action", 12))

    def __init__(self, env, port):
        import torch
        self.torch = torch
        self.env = env
        self.address = ("127.0.0.1", port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session = uuid.uuid4().hex
        self.robot = env.scene["robot"]
        names = list(self.robot.joint_names)
        self.joint_order = torch.tensor(
            [names.index(f"{leg}_{joint}_joint") for leg in self.LEGS for joint in ("hip", "thigh", "calf")],
            device=env.device)
        self.foot_sensors = [env.scene.sensors[f"feet_ground_contact_{leg}_foot_collision_force"]
                             for leg in self.LEGS]
        self.offset = torch.tensor(self.AIRBORNE_FOOT_FORCE, device=env.device)
        self.imu_acc = env.scene.sensors["robot/imu_lin_acc"]
        sites = list(self.robot.site_names)
        self.foot_sites = torch.tensor([sites.index(leg) for leg in self.LEGS], device=env.device)
        # PD gains per joint in FR, FL, RR, RL order (constant, like the real LowCmd kp/kd).
        model = env.sim.mj_model
        actuators = [model.actuator(i).name.split("/")[-1] for i in range(model.nu)]
        order = [actuators.index(f"{leg}_{joint}_joint") for leg in self.LEGS for joint in ("hip", "thigh", "calf")]
        self.kp = [float(model.actuator_gainprm[i, 0]) for i in order]
        self.kd = [float(-model.actuator_biasprm[i, 2]) for i in order]
        # One direct-drive actuator per joint, so actuator force = joint torque (real ls_tau).
        self.actuator_order = torch.tensor(order, device=env.device)
        self.physics_dt = env.physics_dt
        self.tick = 0
        update = env.scene.update

        def update_and_send(*args, **kwargs):
            update(*args, **kwargs)
            self.send()
        env.scene.update = update_and_send

    def send(self):
        torch, data, j = self.torch, self.robot.data, self.joint_order
        force_vec = torch.stack([sensor.data[0] for sensor in self.foot_sensors])  # (4, 3)
        force = force_vec.norm(dim=-1) + self.offset
        quat, base_pos, base_vel = data.root_link_quat_w[0], data.root_link_pos_w[0], data.root_link_lin_vel_w[0]
        foot_pos = data.site_pos_w[0, self.foot_sites] - base_pos
        foot_vel = data.site_lin_vel_w[0, self.foot_sites] - base_vel
        raw = [
            self.imu_acc.data[0], data.actuator_force[0, self.actuator_order], data.joint_acc[0, j],
            data.joint_pos_target[0, j],
            base_pos, base_vel,
            data.root_link_ang_vel_w[0, 2:3], quat_rotate_inverse(quat, foot_pos).reshape(-1),
            quat_rotate_inverse(quat, foot_vel).reshape(-1), force_vec.reshape(-1),
            self.env.command_manager.get_command("twist")[0], self.env.action_manager.action[0],
        ]
        values = torch.cat([force, quat, data.root_link_ang_vel_b[0], data.joint_pos[0, j],
                            data.joint_vel[0, j], *raw]).cpu().tolist()  # one GPU->CPU copy
        fields, start = {}, 35
        for name, size in self.RAW_FIELDS:
            fields[name], start = values[start:start + size], start + size
        fields.update(kp=self.kp, kd=self.kd)
        packet = {"session": self.session, "tick": self.tick,
                  "sim_time": self.tick * self.physics_dt, "low": values[:35], "raw": fields}
        self.tick += 1
        self.socket.sendto(json.dumps(packet).encode(), self.address)

    def close(self):
        self.socket.close()


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
        # Optional: real robot R2's per-step policy command (config command_mirror_ports),
        # used instead of the ROS bridge command while fresh, so both policies get the same
        # command within one policy step (the keyboard mirror path samples at only 20 Hz).
        self.real_receiver, self.real_watchdog = None, None
        if config.get("real_command_port"):
            self.real_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.real_receiver.bind(("127.0.0.1", config["real_command_port"]))
            self.real_receiver.setblocking(False)
            self.real_watchdog = CommandWatchdog(
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
        if self.real_receiver is not None:
            for _ in range(32):
                try:
                    packet, _ = self.real_receiver.recvfrom(4096)
                except BlockingIOError:
                    break
                self.real_watchdog.receive(packet, now)
            if 0 <= now - self.real_watchdog.stamp <= self.real_watchdog.timeout:
                return self.real_watchdog.command
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
        if self.real_receiver is not None:
            self.real_receiver.close()


def quietpaw_policy(state, family, condition_value, device):
    """QuietPaw/QPace actor (with its saved obs normalization) fed 47 Flat inputs + the fixed condition."""
    from copy import deepcopy
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    agent = state["infos"]["quietpaw_agent" if family == "quietpaw" else "go2_qpace_agent"]
    actor_cfg = deepcopy(agent["actor"])
    if actor_cfg.pop("class_name") != "MLPModel" or tuple(agent["obs_groups"]["actor"]) != ("actor",):
        raise ValueError("Only the saved QuietPaw/QPace Flat MLPModel actor is supported")
    dummy = TensorDict({"actor": torch.zeros(1, 48)}, batch_size=[1])
    actor = MLPModel(dummy, {"actor": ["actor"]}, "actor", 12, **actor_cfg)
    actor.load_state_dict(state["actor_state_dict"], strict=True)
    actor.to(device).eval()
    condition = torch.full((1, 1), condition_value, device=device)

    def policy(obs):
        base = obs["actor"]
        if base.shape[-1] != 47:
            raise ValueError(f"Expected 47 Flat observations before the noise condition, got {base.shape}")
        full = torch.cat((base, condition.expand(base.shape[0], 1)), dim=-1)
        return actor(TensorDict({"actor": full}, batch_size=[base.shape[0]]))
    return policy, agent["clip_actions"]


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
    import src.locomotion_rl  # noqa: F401

    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config/simulation.json"
        if not config_path.is_file():
            from ament_index_python.packages import get_package_share_directory
            config_path = Path(get_package_share_directory("go2_autonomous_driving")) / "config/simulation.json"
    config = json.loads(config_path.read_text())
    if args.speed is not None:
        config["max_forward_speed"] = args.speed
    if args.lateral_speed is not None:
        config["max_lateral_speed"] = args.lateral_speed
    if args.yaw_rate is not None:
        config["max_yaw_rate"] = args.yaw_rate
    if args.real_command_port is not None:
        config["real_command_port"] = args.real_command_port
    # Config paths are relative to the config file; explicit CLI paths use cwd.
    # "checkpoint": "../model_pt/model_10000.pt",  <- torch.load로 PT을 읽고 구조를 확인
    checkpoint = args.checkpoint or (config_path.parent / config["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    actor = state["actor_state_dict"]
    inputs = actor["mlp.0.weight"].shape[1]
    # 48-input noise-conditioned Flat models: the same 47 Flat inputs + one condition scalar computed
    # from the PT's saved settings, with the same rule as the real robot (real_quietpaw.py):
    #   QuietPaw (e.g. model_9999): cost limit from --quietness;  QPace (e.g. model_1000): normalized --target-db.
    infos = state.get("infos") or {}
    family = ("quietpaw" if "quietpaw" in infos else "qpace" if "go2_qpace" in infos else None) if inputs == 48 else None
    quietpaw = family is not None
    if inputs not in (47, 48) or (inputs == 48 and not quietpaw) or tuple(actor["mlp.6.weight"].shape)[0] != 12:
        raise ValueError("This simulation requires a Go2 Flat 47-input or QuietPaw/QPace Flat 48-input, 12-action policy")
    condition_value, condition_text = None, ""
    if family == "quietpaw":
        from ..real.real_quietpaw import cost_limit_from_quietness
        condition_value = cost_limit_from_quietness(args.quietness, infos["quietpaw"]["noise_condition"])
        condition_text = f"quietness={args.quietness:g} cost_limit={condition_value:g} (same as R2 --quietness)"
    elif family == "qpace":
        from ..real.real_quietpaw import default_target_db, qpace_condition_from_target_db
        saved = infos["go2_qpace"]["noise_condition"]
        target_db = default_target_db(saved) if args.target_db is None else args.target_db
        condition_value = qpace_condition_from_target_db(target_db, saved)
        condition_text = (f"target_db={target_db:g} (trained {saved['min_db']:g}..{saved['max_db']:g}) "
                          f"condition={condition_value:.4f} (same as R2 --target-db)")
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
    window = noise = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
        if quietpaw:
            policy, clip_actions = quietpaw_policy(state, family, condition_value, device)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
        else:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = load_runner_cls("Unitree-Go2-Flat")(wrapped, asdict(agent_cfg), device=device)
            runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
        wrapped.reset()
        print(f"Loaded {checkpoint}, iter={state.get('iter')}; navigation is initially stopped", flush=True)
        if quietpaw:
            print(f"{family.upper()}: 48 inputs, {condition_text}", flush=True)
        if config.get("real_command_port"):
            print(f"Real command mirror: UDP {config['real_command_port']} (R2 per-step command overrides "
                  f"the bridge while fresh)", flush=True)

        window = NativeWindow(env) if args.native else None
        noise = NoiseStateSender(env, args.noise_port) if args.noise_port else None

        class TelemetryPolicy:
            def __call__(self, obs):
                transport.telemetry(env)
                if window is not None:
                    window.sync()
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
        if window is not None:
            window.close()
        if noise is not None:
            noise.close()
        if env is not None:
            env.close()
        transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--speed", type=float,
                        help="Max forward speed [m/s], overrides config; match the launch speed:=")
    parser.add_argument("--lateral-speed", type=float,
                        help="Max lateral speed [m/s], overrides config; match the launch lateral_speed:=")
    parser.add_argument("--yaw-rate", type=float,
                        help="Max yaw rate [rad/s], overrides config; match the launch yaw_rate:=")
    parser.add_argument("--device")
    parser.add_argument("--viewer", choices=("viser", "none"), default="viser")
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--state-port", "--noise-port", dest="noise_port", type=int,
                        help="Stream real-robot-like physics states (200 Hz) to sim_record.sh or the "
                             "noise predictor, e.g. 9881")
    parser.add_argument("--quietness", type=float, default=0.5,
                        help="QuietPaw (48-input) models only: 0 loose .. 1 strict noise condition; "
                             "use the same value as the real robot (R2 --quietness)")
    parser.add_argument("--target-db", type=float,
                        help="QPace (48-input) models only: target robot noise [dB] in the trained range "
                             "(default reference_db); use the same value as the real robot (R2 --target-db)")
    parser.add_argument("--real-command-port", type=int,
                        help="Follow the real robot's per-step policy command (R2 command_mirror_ports), "
                             "e.g. 9905 for real+sim recording; falls back to the bridge command when stale")
    parser.add_argument("--native", action="store_true",
                        help="Also open a MuJoCo window that follows the robot (needs a host display)")
    parser.add_argument("--steps", type=int, help="Stop after this many policy steps")
    parser.add_argument("--fast", action="store_true", help="Headless benchmark without real-time pacing")
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
