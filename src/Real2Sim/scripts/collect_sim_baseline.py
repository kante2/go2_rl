#!/usr/bin/env python3
"""Collect baseline Go2 policy rollouts; no ROS, UDP or robot connection.

Default: replay 0.2_1.npz's applied velocity commands with the identical PT.
Run through run_collect_sim_baseline.sh in the existing mjlab container.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/go2_autonomous_driving"))
sys.path.insert(0, str(ROOT / "src/go2_autodrive"))

LEGS = ("FR", "FL", "RR", "RL")
JOINTS = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")]
DEFAULT_REFERENCE = ROOT / "src/db_collector/data/0.2_1.npz"
DEFAULT_CHECKPOINT = ROOT / "src/go2_autodrive/config/model_pt/baseline/model_34997.pt"
DEFAULT_CONFIG = ROOT / "src/go2_autodrive/config/baseline.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CommandSchedule:
    """Causal zero-order hold of the recorded, already-ramped command."""

    def __init__(self, t, command, duration, events=None):
        self.t = np.asarray(t, dtype=float)
        self.command = np.asarray(command, dtype=float)
        self.duration = float(duration)
        self.events = events or []
        if (self.t.ndim != 1 or len(self.t) == 0
                or self.command.shape != (len(self.t), 3)
                or not np.isfinite(self.t).all() or not np.isfinite(self.command).all()
                or np.any(np.diff(self.t) <= 0)
                or not math.isfinite(self.duration) or self.duration <= 0):
            raise ValueError("Invalid command schedule")

    def at(self, t):
        index = np.searchsorted(self.t, t, side="right") - 1
        return np.zeros(3) if index < 0 else self.command[index].copy()

    @classmethod
    def from_reference(cls, path, checkpoint_hash):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(z["meta"].item())
            if meta["control_config"]["model"] != "baseline":
                raise ValueError("Reference must use baseline")
            if meta["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Checkpoint SHA256 does not match the real recording")
            if not meta.get("stop_confirmed") or meta.get("stop_reason") != "completed":
                raise ValueError("Reference trial did not complete normally")
            labels, stamps = z["ev_label"], z["ev_t"]
            start = float(stamps[np.flatnonzero(labels == "record_start")[0]])
            end = float(stamps[np.flatnonzero(labels == "record_stop")[0]])
            events = [(float(t - start), str(label)) for t, label in zip(stamps, labels)]
            schedule = cls(z["rl_t"] - start, z["rl_command"], end - start, events)
        return schedule, meta

    @classmethod
    def generated(cls, vx, pre, duration, post, dt, config):
        from go2_autonomous_driving.real.control_transitions import VelocityRamp

        ramp = VelocityRamp(config["command_acceleration"], config["command_deceleration"], dt)
        # Include deceleration before the post-stop recording interval.
        stop = pre + duration
        total = stop + abs(vx) / config["command_deceleration"][0] + post
        t = np.arange(math.ceil(total / dt) + 1) * dt
        commands = [ramp.step([vx if pre <= stamp < stop else 0., 0., 0.]) for stamp in t]
        return cls(t, commands, total, [(0., "record_start"), (pre, "move_start"),
                                       (stop, "move_stop"), (total, "record_stop")])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    source.add_argument("--no-reference", action="store_true", help="Generate a velocity ramp instead")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vx", type=float, default=.2, help="Forward command [m/s]")
    p.add_argument("--trial", type=int, default=1, help="Generated-mode filename trial number")
    p.add_argument("--pre", type=float, default=5.)
    p.add_argument("--duration", type=float, default=15.)
    p.add_argument("--post", type=float, default=5.)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=ROOT / "src/Real2Sim/simulator/baseline")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if (not np.isfinite([args.vx, args.pre, args.duration, args.post]).all()
            or args.vx < 0 or min(args.pre, args.post) < 0 or args.duration <= 0
            or args.seed < 0 or args.trial < 1):
        p.error("Invalid velocity, durations, seed or trial")
    return args


class Recorder:
    """Post-integration physics states and the commands that produced them."""

    def __init__(self, env, mapping):
        self.env, self.mapping = env, mapping
        robot = env.scene["robot"]
        self.robot = robot.data
        self.jids = [robot.joint_names.index(name) for name in JOINTS]
        self.aids = [robot.actuator_names.index(name) for name in JOINTS]
        self.rows = []
        self.control = []
        self.enabled = False
        self.original_update = env.scene.update
        env.scene.update = self.update

    def update(self, dt):
        # Refresh derived sensors to the same post-integration state as q/dq.
        self.env.sim.forward()
        self.original_update(dt)
        if self.enabled:
            self.sample()

    def sample(self):
        import torch

        d, env = self.robot, self.env
        # Sensor force acts on the terrain. Store its reaction on the robot.
        force = -torch.stack([env.scene[f"gait_monitor_{leg}"].data.force[0, 0] for leg in LEGS])
        values = torch.cat((
            d.joint_pos[0, self.jids], d.joint_vel[0, self.jids],
            d.actuator_force[0, self.aids], d.root_link_quat_w[0],
            env.scene["robot/imu_ang_vel"].data[0],
            env.scene["robot/imu_lin_acc"].data[0],
            d.root_link_pos_w[0], d.root_link_lin_vel_b[0], d.root_link_lin_vel_w[0],
            force.reshape(-1),
        )).detach().cpu().numpy().astype(float)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite simulated state")
        self.rows.append(np.r_[env._sim_step_counter * env.physics_dt, values])

    def arrays(self):
        from scipy.spatial.transform import Rotation

        a, c = np.asarray(self.rows), np.asarray(self.control)
        quat = a[:, 37:41]
        return dict(
            ls_t=a[:, 0], ls_tick=a[:, 0] * 1000.,
            ls_q=a[:, 1:13], ls_dq=a[:, 13:25], ls_tau=a[:, 25:37],
            ls_quaternion=quat, ls_rpy=Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_euler("xyz"),
            ls_gyro=a[:, 41:44], ls_acc=a[:, 44:47],
            sim_base_pos_world=a[:, 47:50], sim_base_vel_body=a[:, 50:53],
            sim_base_vel_world=a[:, 53:56], sim_foot_force_world_N=a[:, 56:68].reshape(-1, 4, 3),
            sim_foot_fz_N=a[:, 56:68].reshape(-1, 4, 3)[:, :, 2],
            lc_t=c[:, 0], rl_t=c[:, 0], rl_command=c[:, 1:4],
            policy_observation=c[:, 4:51], policy_action_raw=c[:, 51:63],
            policy_action_applied=c[:, 63:75], lc_q=c[:, 75:87],
            lc_dq=np.zeros((len(c), 12)), lc_tau=np.zeros((len(c), 12)),
            lc_kp=np.tile(self.mapping.to_sdk(self.mapping.meta["joint_stiffness"]), (len(c), 1)),
            lc_kd=np.tile(self.mapping.to_sdk(self.mapping.meta["joint_damping"]), (len(c), 1)),
        )


def run(args):
    checkpoint_hash = sha256(args.checkpoint)
    config = json.loads(args.config.read_text())
    real_meta = None
    schedule = None
    if not args.no_reference:
        schedule, real_meta = CommandSchedule.from_reference(args.reference, checkpoint_hash)
        if not np.isclose(real_meta["vx"], args.vx):
            raise ValueError("--vx must match the reference's requested forward velocity")
        config = real_meta["control_config"]
    name = f"{args.vx:.1f}_{args.trial}" if args.no_reference else args.reference.stem
    output = args.output_dir / f"{name}.npz"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {output}")
    # Heavy dependencies are needed only after validating inputs and hashes.
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import get_base_metadata
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from go2_autodrive.simulation.go2_mujuco_udp_runner import check_policy_shape
    from go2_autonomous_driving.real.real_control import PolicyMapping
    from go2_autonomous_driving.real.control_transitions import JointTargetRamp
    from src.go2_gait_monitor.go2_gait_monitor.mujoco import add_contact_sensors
    import src.locomotion_rl  # noqa: F401 -- task registration

    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    check_policy_shape(state, "baseline")
    configure_torch_backends()
    cfg = load_env_cfg("Unitree-Go2-Flat", play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    # Nominal model: retain deterministic resets, remove startup randomization.
    cfg.events = {key: value for key, value in cfg.events.items()
                  if key in ("reset_base", "reset_robot_joints")}
    cfg.events["reset_base"].params["pose_range"] = {}
    cfg.observations["actor"].enable_corruption = False
    add_contact_sensors(cfg)
    env = None
    recorder = None
    completed, failure = False, None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        dt = env.step_dt
        if schedule is None:
            schedule = CommandSchedule.generated(args.vx, args.pre, args.duration, args.post, dt, config)
        agent = load_rl_cfg("Unitree-Go2-Flat")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls("Unitree-Go2-Flat")(wrapped, asdict(agent), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        metadata = get_base_metadata(env, str(args.checkpoint))
        model = env.sim.mj_model
        metadata.update(
            policy_to_sdk=[JOINTS.index(n) for n in metadata["joint_names"]],
            joint_limits=[model.jnt_range[model.joint("robot/" + n).id].tolist()
                          for n in metadata["joint_names"]],
            clip_actions=agent.clip_actions, step_dt=dt,
            phase_period=cfg.observations["actor"].terms["phase"].params["period"],
        )
        mapping = PolicyMapping(metadata)
        ramp = JointTargetRamp(config["max_joint_target_rate"], dt,
            transition_rate=config.get("phase_transition_joint_rate"),
            transition_seconds=config.get("phase_transition_seconds", 0.))
        env.reset(seed=args.seed)
        recorder = Recorder(env, mapping)
        recorder.enabled = True
        recorder.sample()
        previous = np.zeros(12, dtype=np.float32)
        nsteps = math.ceil(schedule.duration / dt)
        print(f"baseline {args.vx:g} m/s, {nsteps} policy steps, "
              f"physics {1 / env.physics_dt:g} Hz, policy {1 / dt:g} Hz -> {output}", flush=True)
        with torch.inference_mode():
            for step in range(nsteps):
                stamp = step * dt
                command = schedule.at(stamp)
                env.command_manager.get_command("twist")[:] = torch.tensor(command, device=args.device)
                # Use the real deployment observation and target-ramp contracts.
                d = recorder.robot
                q = d.joint_pos[0, recorder.jids].cpu().numpy()
                obs = mapping.observation(q, d.joint_vel[0, recorder.jids].cpu().numpy(),
                    d.root_link_quat_w[0].cpu().numpy(),
                    env.scene["robot/imu_ang_vel"].data[0].cpu().numpy(), command, step, previous)
                inputs = wrapped.get_observations()
                inputs["actor"][:] = torch.as_tensor(obs, device=args.device)
                raw = policy(inputs).detach().cpu().numpy()[0]
                _, target = mapping.targets(raw)
                target = ramp.step(target, initial=q if step == 0 else None,
                                   moving=bool(np.any(obs[0, 9:11])))
                previous = ((target[mapping.order] - mapping.default) / mapping.scale).astype(np.float32)
                recorder.control.append(np.r_[stamp, command, obs[0], raw, previous, target])
                _, _, done, _ = wrapped.step(torch.as_tensor(previous[None], device=args.device))
                if bool(done[0]):
                    raise RuntimeError(f"Environment terminated at {(step + 1) * dt:.3f}s; trial stopped")
                if (step + 1) % 250 == 0:
                    print(f"  {(step + 1) * dt:.1f}/{schedule.duration:.1f} sim seconds", flush=True)
        completed = True
    except (KeyboardInterrupt, Exception) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if recorder is not None and recorder.control and recorder.rows:
                arrays = recorder.arrays()
                meta = dict(
                    schema="go2-baseline-sim-v1", source="mjlab/MuJoCo", model="baseline",
                    checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                    reference=str(args.reference) if real_meta else None,
                    reference_sha256=sha256(args.reference) if real_meta else None,
                    command_source="real rl_command zero-order hold" if real_meta else "generated velocity ramp",
                    requested_vx=args.vx, seed=args.seed, completed=completed, failure=failure,
                    requested_recording_duration_s=schedule.duration,
                    recorded_duration_s=float(arrays["ls_t"][-1]),
                    physics_dt=env.physics_dt, policy_dt=env.step_dt,
                    randomization=False, initial_state="nominal model pose, zero velocities",
                    joint_order=JOINTS, foot_order=list(LEGS), policy_mapping=metadata,
                    control_config=config, target_ramp=True,
                    time_semantics="simulation seconds from recording start; lc_t starts an action interval",
                    state_semantics="post-integration; initial state at t=0 included",
                    force_semantics="World-frame GRF on robot; negated primary-foot sensor force, upward Fz positive. Not real raw counts.",
                    torque_semantics="ls_tau: simulated actuator torque; lc_tau: feedforward only",
                    limitations=["Actual base position/velocity absent from real recordings.",
                                 "Recorded command receive times are not exact policy application times.",
                                 "Nominal initialization does not reproduce the real initial state or gait phase.",
                                 "Real watchdog/fault state machine is not reproduced.",
                                 "Physics samples are 200 Hz; real LowState is approximately 500 Hz."],
                )
                arrays.update(meta=np.array(json.dumps(meta, ensure_ascii=False)),
                              ev_t=np.array([e[0] for e in schedule.events]),
                              ev_label=np.array([e[1] for e in schedule.events]))
                save_npz(output, arrays, args.overwrite)
                print(f"Saved {len(arrays['ls_t'])} states; completed={completed}: {output}", flush=True)
        finally:
            if env is not None:
                env.close()


def save_npz(path, arrays, overwrite=False):
    """Exclusive by default; never silently replace a trial."""
    owner_dir = path.parent
    while not owner_dir.exists():
        owner_dir = owner_dir.parent
    owner = owner_dir.stat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb" if overwrite else "xb") as stream:
        np.savez_compressed(stream, **arrays)
    try:
        os.chown(path.parent, owner.st_uid, owner.st_gid)
        os.chown(path, owner.st_uid, owner.st_gid)
    except PermissionError:
        pass


if __name__ == "__main__":
    run(parse_args())
