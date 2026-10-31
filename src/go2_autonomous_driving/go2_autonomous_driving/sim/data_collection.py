"""Collect Go2 walking data in MuJoCo with direct keyboard velocity commands.

Rows are written in the column layout of the real-robot trial CSV
(sim_to_real_data/real/*.csv), one row per physics step (200 Hz). Audio columns
stay empty because the simulation has no microphone.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np
from scipy.spatial.transform import Rotation

SOURCE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE_ROOT / "low_noise_prediction/preprocessing"))
from go2_kinematics import Go2Kinematics, LEGS, world_foot_velocity  # noqa: E402

from ..real.control_transitions import VelocityRamp  # noqa: E402

JOINTS = ("hip", "thigh", "calf")
SDK_JOINT_NAMES = [f"{leg.upper()}_{joint}_joint" for leg in LEGS for joint in JOINTS]
FOOT_GEOMS = [f"{leg.upper()}_foot_collision" for leg in LEGS]
CONTACT_THRESHOLD = 1.0  # [N] vertical foot force counted as stance.


def csv_columns():
    """Header identical to the real-robot trial CSV."""
    axes = ("x", "y", "z")
    joint = [f"{leg}_{name}" for leg in LEGS for name in JOINTS]
    columns = ["trial", "command_vx", "command_vy", "command_vyaw", "time_abs", "time_from_move_start",
               "audio_rms_20ms", "audio_dbfs_20ms", "audio_peak_2ms", "audio_delay_s"]
    columns += [f"ls_acc_{a}" for a in axes] + [f"ls_dq_{j}" for j in joint]
    columns += [f"ls_ff_{leg}" for leg in LEGS] + [f"ls_gyro_{a}" for a in axes]
    columns += [f"ls_q_{j}" for j in joint] + [f"ls_quaternion_{a}" for a in "wxyz"]
    columns += [f"ls_rpy_{a}" for a in ("roll", "pitch", "yaw")] + [f"ls_tau_{j}" for j in joint]
    columns += ["ls_tick_value"]
    columns += [f"knee_{group}_{leg}" for group in ("q", "dq", "tau") for leg in LEGS]
    columns += [f"sp_vel_{a}" for a in axes]
    columns += [f"sp_foot_vel_{leg}_{a}" for leg in LEGS for a in axes]
    columns += ["sp_vel_x_smooth", "sp_acc_x", "is_steady", "segment", "lowstate_valid", "audio_valid",
                "source_data_valid", "body_velocity_source", "command_source", "synchronization_status",
                "body_velocity_estimate_valid", "body_velocity_contact_count",
                "body_velocity_consensus_residual_mps", "body_velocity_invalid_reason"]
    for a in axes:
        columns += [f"body_velocity_est_body_{a}_mps", f"body_velocity_est_world_{a}_mps"]
    columns += [f"body_velocity_contact_{leg}" for leg in LEGS]
    for leg in LEGS:
        for a in axes:
            columns += [f"foot_position_body_{leg}_{a}_m", f"foot_velocity_relative_{leg}_{a}_mps",
                        f"foot_velocity_world_{leg}_{a}_mps"]
        columns += [f"foot_speed_world_{leg}_mps", f"foot_downward_speed_{leg}_mps"]
    return columns


class KeyboardVelocity:
    """Same keys as the real keyboard: a direction stays active until Space."""
    directions = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "j": (2, 1), "l": (2, -1)}

    def __init__(self, speeds):
        self.speeds = speeds
        self.requested = np.zeros(3)
        self.buffer = ""

    def keys(self):
        if not select.select([sys.stdin], [], [], 0)[0]:
            return []
        data = os.read(sys.stdin.fileno(), 64).decode(errors="ignore")
        if not data:
            return []
        self.buffer += data
        keys = []
        while self.buffer:
            if self.buffer[0] == "\x1b":
                if len(self.buffer) < 3:
                    break
                key = {"A": "w", "B": "s", "C": "d", "D": "a"}.get(self.buffer[2]) if self.buffer[1] == "[" else None
                if key:
                    keys.append(key)
                self.buffer = self.buffer[3:] if self.buffer[1] == "[" else self.buffer[2:]
            else:
                keys.append(self.buffer[0].lower())
                self.buffer = self.buffer[1:]
        return keys

    def handle(self, key):
        if key == " ":
            self.requested[:] = 0
        elif key in self.directions:
            axis, sign = self.directions[key]
            self.requested[:] = 0
            self.requested[axis] = sign * self.speeds[axis]


class TrialRecorder:
    """Buffers physics-step samples and writes one CSV per R-to-R recording."""

    def __init__(self, env, output_dir, label, command):
        self.env = env
        self.output_dir = output_dir
        self.label = label
        self.kinematics = Go2Kinematics()
        robot = env.scene["robot"]
        self.robot = robot.data
        self.joint_ids = [robot.joint_names.index(name) for name in SDK_JOINT_NAMES]
        self.actuator_ids = [robot.actuator_names.index(name) for name in SDK_JOINT_NAMES]
        self.gyro = env.scene["robot/imu_ang_vel"]
        self.acc = env.scene["robot/imu_lin_acc"]
        self.contact = env.scene["feet_ground_contact"]
        contact_names = list(dict.fromkeys(slot.primary_name for slot in self.contact._slots))
        self.contact_ids = [next(i for i, name in enumerate(contact_names) if name.endswith(geom))
                            for geom in FOOT_GEOMS]
        self.command = command  # Applied (ramped) command, updated in place.
        self.samples = None
        self.trial = None
        self.segment = 1
        self.last_episode_step = None

    @property
    def active(self):
        return self.samples is not None

    def start(self):
        self.trial = f"{self.label}_{datetime.now():%Y%m%d_%H%M%S}"
        self.samples = []
        self.segment = 1
        self.last_episode_step = None
        self.start_time = self.sim_time()
        print(f"REC START {self.trial}", flush=True)

    def sim_time(self):
        return self.env._sim_step_counter * self.env.physics_dt

    def sample(self):
        import torch
        episode_step = int(self.env.episode_length_buf[0])
        if self.last_episode_step is not None and episode_step < self.last_episode_step:
            self.segment += 1  # The robot fell and the environment was reset.
        self.last_episode_step = episode_step
        data = self.robot
        force = self.contact.data.force[0, self.contact_ids]
        values = torch.cat((
            data.joint_pos[0, self.joint_ids], data.joint_vel[0, self.joint_ids],
            data.actuator_force[0, self.actuator_ids], data.root_link_quat_w[0],
            self.gyro.data[0], self.acc.data[0], data.root_link_lin_vel_b[0],
            data.root_link_lin_vel_w[0], force.reshape(-1),
        )).detach().cpu().numpy().astype(float)
        self.samples.append(np.concatenate(([self.sim_time(), self.segment], self.command, values)))

    def stop(self):
        if not self.active:
            return
        samples, self.samples = np.asarray(self.samples), None
        if len(samples) == 0:
            print("REC STOP: no samples, nothing written", flush=True)
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.trial}.csv"
        rows = self.rows(samples)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(csv_columns())
            writer.writerows(rows)
        meta = path.with_suffix(".json")
        meta.write_text(json.dumps({
            "source": "mjlab Unitree-Go2-Flat play", "rows": len(rows),
            "sample_rate_hz": 1 / self.env.physics_dt, "policy_rate_hz": 1 / self.env.step_dt,
            "segments": int(samples[-1, 1]),
            "ls_ff": "|vertical foot contact force| [N] (real robot: raw foot sensor counts)",
            "ls_tau": "simulated actuator force [N m]",
            "ls_tick_value": "simulation time [ms]",
            "body_velocity": "simulation ground truth (real robot: stance kinematics estimate)",
            "body_velocity_contact": f"vertical foot force > {CONTACT_THRESHOLD} N",
            "audio": "not simulated; audio columns are empty",
        }, indent=2))
        # The container runs as root; give the files to the owner of the output folder.
        owner = self.output_dir.stat()
        for file in (path, meta):
            try:
                os.chown(file, owner.st_uid, owner.st_gid)
            except PermissionError:
                pass
        print(f"REC STOP {len(rows)} rows ({len(rows) * self.env.physics_dt:.1f} s) -> {path}", flush=True)

    def rows(self, samples):
        t, segment, command = samples[:, 0], samples[:, 1], samples[:, 2:5]
        q, dq, tau = samples[:, 5:17], samples[:, 17:29], samples[:, 29:41]
        quat, gyro, acc = samples[:, 41:45], samples[:, 45:48], samples[:, 48:51]
        velocity_body, velocity_world = samples[:, 51:54], samples[:, 54:57]
        foot_force = samples[:, 57:69].reshape(-1, 4, 3)
        foot_load = np.abs(foot_force[:, :, 2])
        contact = foot_load > CONTACT_THRESHOLD
        rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
        rpy = rotation.as_euler("xyz")
        feet = []
        for i, leg in enumerate(LEGS):
            position, jacobian = self.kinematics.position_jacobian(leg, q[:, 3 * i:3 * i + 3])
            relative = np.einsum("nij,nj->ni", jacobian, dq[:, 3 * i:3 * i + 3])
            world = world_foot_velocity(position, relative, gyro, rotation, velocity_body, "body")
            feet.append((position, relative, world))
        empty = [""] * 12
        rows = []
        for n in range(len(samples)):
            row = [self.trial, *command[n], t[n], t[n] - self.start_time, "", "", "", ""]
            row += [*acc[n], *dq[n], *foot_load[n], *gyro[n], *q[n], *quat[n], *rpy[n], *tau[n]]
            row += [round(t[n] * 1000)]
            row += [*q[n, 2::3], *dq[n, 2::3], *tau[n, 2::3]]
            row += [*velocity_body[n], *empty, "", "", "", int(segment[n]), 1, 0, 1,
                    "simulation_ground_truth", "simulation_keyboard", "SIMULATION_EXACT",
                    1, int(contact[n].sum()), "", "valid"]
            for a in range(3):
                row += [velocity_body[n, a], velocity_world[n, a]]
            row += [int(c) for c in contact[n]]
            for position, relative, world in feet:
                for a in range(3):
                    row += [position[n, a], relative[n, a], world[n, a]]
                row += [np.linalg.norm(world[n]), max(-world[n, 2], 0.0)]
            rows.append([repr(float(v)) if isinstance(v, (float, np.floating)) else v for v in row])
        return rows


def run(args):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
    from mjlab.utils.torch import configure_torch_backends
    import src.locomotion_rl  # noqa: F401

    config_path = args.config or Path(__file__).resolve().parents[2] / "config/real_closed_loop.json"
    config = json.loads(config_path.read_text())
    speeds = [args.speed or config["keyboard_forward_speed"],
              args.lateral_speed or config["keyboard_lateral_speed"],
              args.yaw_rate or config["keyboard_yaw_rate"]]
    checkpoint = args.checkpoint or Path(__file__).resolve().parents[2] / "model_pt/model_10000.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    configure_torch_backends()
    keyboard = KeyboardVelocity(speeds)
    recorder = None
    command = np.zeros(3)
    ramp = None
    stop_requested = False

    def handle_keys():
        nonlocal stop_requested
        for key in keyboard.keys():
            if key in ("q", "\x03", "\x04"):
                # The viewers catch exceptions raised inside a step, so quit through a flag.
                stop_requested = True
                return
            if key == "r" and recorder is not None:
                recorder.stop() if recorder.active else recorder.start()
            elif key in KeyboardVelocity.directions or key == " ":
                keyboard.handle(key)
                c = keyboard.requested
                print(f"command=[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]", flush=True)

    class KeyboardCommand(UniformVelocityCommand):
        def compute(self, dt):
            super().compute(dt)
            handle_keys()
            if ramp is not None:
                # Same acceleration limits as the real closed loop.
                command[:] = ramp.step(keyboard.requested)
            self.vel_command_b[:] = torch.tensor(command, device=self.device)

        def create_gui(self, *unused_args, **unused_kwargs):
            pass

    class KeyboardCommandCfg(UniformVelocityCommandCfg):
        def build(self, env):
            return KeyboardCommand(self, env)

    cfg = load_env_cfg("Unitree-Go2-Flat", play=True)
    cfg.scene.num_envs = 1
    cfg.commands["twist"] = KeyboardCommandCfg(
        entity_name="robot", resampling_time_range=(1e6, 1e6),
        heading_command=False, rel_standing_envs=0.0, debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-config["max_forward_speed"], config["max_forward_speed"]),
            lin_vel_y=(-config["max_lateral_speed"], config["max_lateral_speed"]),
            ang_vel_z=(-config["max_yaw_rate"], config["max_yaw_rate"]), heading=None,
        ),
    )
    env = None
    original_terminal = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        recorder = TrialRecorder(env, args.output_dir, args.label, command)
        ramp = VelocityRamp(config["command_acceleration"], config["command_deceleration"], env.step_dt)
        scene_update = env.scene.update

        def update_and_record(dt):
            # mj_step leaves sensors/contacts one substep behind qpos; refresh them
            # so every row is one consistent physical state.
            env.sim.forward()
            scene_update(dt)
            if recorder.active:
                recorder.sample()

        env.scene.update = update_and_record
        agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = load_runner_cls("Unitree-Go2-Flat")(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        wrapped.reset()
        print(f"Loaded {checkpoint}\n"
              f"Keys (this terminal): W/S forward/backward {speeds[0]:.2f} m/s, A/D lateral {speeds[1]:.2f} m/s, "
              f"J/L turn {speeds[2]:.2f} rad/s; Space stop; R stop/start a new trial file; Q save and quit.\n"
              f"Recording {1 / env.physics_dt:.0f} Hz rows to {args.output_dir}", flush=True)
        if original_terminal is not None:
            tty.setcbreak(sys.stdin.fileno())
        if args.record:
            recorder.start()
        try:
            if args.viewer in ("native", "viser"):
                if args.viewer == "native":
                    from mjlab.viewer import NativeMujocoViewer
                    viewer = NativeMujocoViewer(wrapped, policy)
                else:
                    import viser
                    from mjlab.viewer import ViserPlayViewer
                    viewer = ViserPlayViewer(wrapped, policy, viser_server=viser.ViserServer(port=args.viewer_port))
                viewer_running = viewer.is_running

                def is_running():
                    handle_keys()  # Also while the viewer is paused.
                    return viewer_running() and not stop_requested

                viewer.is_running = is_running
                viewer.run(num_steps=args.steps)
            else:
                deadline = time.monotonic()
                index = 0
                with torch.inference_mode():
                    while (args.steps is None or index < args.steps) and not stop_requested:
                        wrapped.step(policy(wrapped.get_observations()))
                        index += 1
                        deadline += env.step_dt
                        if not args.fast:
                            time.sleep(max(0.0, deadline - time.monotonic()))
        except KeyboardInterrupt:
            pass
    finally:
        if original_terminal is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)
        if recorder is not None:
            recorder.stop()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="Keyboard speeds/ramps (default config/real_closed_loop.json)")
    parser.add_argument("--output-dir", type=Path, default=SOURCE_ROOT / "sim_to_real_data/simulator")
    parser.add_argument("--label", default="SIM", help="Trial file prefix, e.g. NORMAL")
    parser.add_argument("--speed", type=float, help="W/S speed in m/s")
    parser.add_argument("--lateral-speed", type=float, help="A/D speed in m/s")
    parser.add_argument("--yaw-rate", type=float, help="J/L yaw rate in rad/s")
    parser.add_argument("--device")
    parser.add_argument("--viewer", choices=("native", "viser", "none"), default="native")
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True,
                        help="Record from launch (default); --no-record waits for R")
    parser.add_argument("--steps", type=int, help="Stop after this many policy steps")
    parser.add_argument("--fast", action="store_true", help="With --viewer none, run without real-time pacing")
    args = parser.parse_args()
    for value in (args.speed, args.lateral_speed, args.yaw_rate):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error("speeds must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
