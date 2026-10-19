"""Read-only instrumentation for all three MuJoCo policy families.

    Contact sensors sample each physics substep, before episode reset.
    Transfer their small buffers to CPU once per policy step. Send UI data
    separately from navigation so either receiver can start/stop independently.
"""

import json
import socket
import time
import uuid

from .metrics import FEET, FootMetrics


def add_contact_sensors(cfg):
    from mjlab.sensor import ContactSensorCfg, ContactMatch

    # One explicitly named geom per sensor avoids implicit joint/geom ordering.
    sensors = [ContactSensorCfg(
        name=f"gait_monitor_{foot}",
        primary=ContactMatch(mode="geom", pattern=f"{foot}_foot_collision", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",), reduce="netforce", num_slots=1,
    ) for foot in FEET]
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(sensors)


class MujocoMonitor:
    def __init__(self, env, config, checkpoint, port):
        self.env, self.config = env, config
        self.checkpoint = checkpoint.name
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.session = uuid.uuid4().hex
        self.reset_counter = 0
        self.sequence = 0
        self.next_send = 0.0
        self.metrics = FootMetrics()
        self.samples = []
        self.original_update = env.scene.update
        self.original_reset = env._reset_idx
        env.scene.update = self.update
        env._reset_idx = self.reset

    def update(self, dt):
        import torch

        self.original_update(dt)
        forces = torch.stack([
            self.env.scene[f"gait_monitor_{foot}"].data.force[0, 0, 2]
            for foot in FEET
        ]).detach().clone()
        self.samples.append(forces)

    def reset(self, *args, **kwargs):
        result = self.original_reset(*args, **kwargs)
        self.metrics = FootMetrics()
        self.samples.clear()
        self.reset_counter += 1
        return result

    def publish(self, force=False):
        import torch

        if self.samples:
            forces = torch.stack(self.samples).cpu().tolist()
            self.samples.clear()
            for sample in forces:
                self.metrics.sample(sample, self.env.physics_dt)
        now = time.monotonic()
        if not force and now < self.next_send:
            return
        self.next_send = now + .05
        robot = self.env.scene["robot"].data
        command = self.env.command_manager.get_command("twist")[0].detach().cpu().tolist()
        packet = dict(
            version=1, source="mujoco", session=self.session, sequence=self.sequence,
            reset_counter=self.reset_counter, sim_time=self.metrics.time,
            model=self.config["model"], checkpoint=self.checkpoint,
            quietness=self.config["quietness"] if self.config["model"] != "baseline" else None,
            physics_dt=self.env.physics_dt, feet=list(FEET), command=command,
            linear_velocity=robot.root_link_lin_vel_b[0].detach().cpu().tolist(),
            angular_velocity=robot.root_link_ang_vel_b[0].detach().cpu().tolist(),
            **self.metrics.snapshot(),
        )
        try:
            self.socket.sendto(json.dumps(packet, allow_nan=False).encode(), ("127.0.0.1", self.port))
        except (BlockingIOError, OSError):
            # UI availability must never stop the simulation/controller.
            pass
        self.sequence += 1
        self.metrics.clear_interval_peak()

    def close(self):
        self.env.scene.update = self.original_update
        self.env._reset_idx = self.original_reset
        self.socket.close()
