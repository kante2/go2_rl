"""Pure policy mapping and Unitree wire CRC; no ROS publishers or hardware I/O."""
import math
import struct

import numpy as np


def _crc_table():
    table = []
    for byte in range(256):
        value = byte << 24
        for _ in range(8):
            value = ((value << 1) ^ (0x04C11DB7 if value & 0x80000000 else 0)) & 0xFFFFFFFF
        table.append(value)
    return table


_TABLE = _crc_table()


def next_period(deadline, period, now):
    """Skip elapsed slots instead of accumulating lateness or running catch-up bursts."""
    deadline += period
    missed = max(0, math.floor((now - deadline) / period) + 1) if now >= deadline else 0
    return deadline + missed * period, missed


def require_independent_navigation(config):
    """Known Go2 sport-derived localization disappeared when normal was released."""
    dependent = {'/utlidar/robot_odom', '/utlidar/cloud_deskewed', '/utlidar/cloud_base'}
    selected = {config['odom_topic'], config['cloud_topic']}
    if selected & dependent:
        raise RuntimeError(
            'LIVE BLOCKED before releasing normal mode: the configured Unitree odometry/cloud '
            'stop with the built-in controller on this Go2. Configure independently running '
            'localization (Odometry + world-frame PointCloud2) first. Dry-run is still available.'
        )


def wire_crc(packed):
    """Unitree CRC: little-endian uint32 words, MSB-first bits, omit final CRC.

    Wire layouts follow unitreerobotics/unitree_sdk2_python utils/crc.py.
    """
    crc = 0xFFFFFFFF
    for offset in range(0, len(packed) - 4, 4):
        for index in (3, 2, 1, 0):
            crc = ((crc << 8) ^ _TABLE[((crc >> 24) ^ packed[offset + index]) & 255]) & 0xFFFFFFFF
    return crc


def pack_low_command(cmd):
    values = [*cmd.head, cmd.level_flag, cmd.frame_reserve, *cmd.sn, *cmd.version, cmd.bandwidth]
    for motor in cmd.motor_cmd:
        values.extend([motor.mode, motor.q, motor.dq, motor.tau, motor.kp, motor.kd, *motor.reserve])
    values.extend([cmd.bms_cmd.off, *cmd.bms_cmd.reserve, *cmd.wireless_remote,
                   *cmd.led, *cmd.fan, cmd.gpio, cmd.reserve, cmd.crc])
    return struct.pack('<4B4IH2x' + 'B3x5f3I' * 20 + '4B55Bx2I', *values)


def pack_low_state(msg):
    imu, bms = msg.imu_state, msg.bms_state
    values = [*msg.head, msg.level_flag, msg.frame_reserve, *msg.sn, *msg.version, msg.bandwidth,
              *imu.quaternion, *imu.gyroscope, *imu.accelerometer, *imu.rpy, imu.temperature]
    for motor in msg.motor_state:
        values.extend([motor.mode, motor.q, motor.dq, motor.ddq, motor.tau_est,
                       motor.q_raw, motor.dq_raw, motor.ddq_raw, motor.temperature,
                       motor.lost, *motor.reserve])
    values.extend([bms.version_high, bms.version_low, bms.status, bms.soc, bms.current,
                   bms.cycle, *bms.bq_ntc, *bms.mcu_ntc, *bms.cell_vol,
                   *msg.foot_force, *msg.foot_force_est, msg.tick, *msg.wireless_remote,
                   msg.bit_flag, msg.adc_reel, msg.temperature_ntc1, msg.temperature_ntc2,
                   msg.power_v, msg.power_a, *msg.fan_frequency, msg.reserve, msg.crc])
    return struct.pack('<4B4IH2x13fb3x' + 'B3x7fb3x3I' * 20 + '4BiH4b15H8hI41B3xf2b2x2f4H2I', *values)


def gravity_body(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not 0.95 < norm < 1.05:
        raise ValueError('Invalid IMU quaternion')
    w, x, y, z = quaternion / norm
    return np.array([2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x*x + y*y) - 1])


def remote_estop(remote):
    return bool(int(remote[2]) & 0x20 and int(remote[3]) & 0x02)  # L2 + B


class PolicyMapping:
    def __init__(self, metadata):
        self.meta = metadata
        self.order = np.asarray(metadata['policy_to_sdk'], dtype=int)
        self.default = np.asarray(metadata['default_joint_pos'], dtype=np.float32)
        self.scale = np.broadcast_to(np.asarray(metadata['action_scale'], dtype=np.float32), (12,))
        self.limits = np.asarray(metadata['joint_limits'], dtype=np.float32)
        if sorted(self.order.tolist()) != list(range(12)) or self.default.shape != (12,):
            raise ValueError('Invalid joint mapping')
        if metadata['observation_names'] != ['base_ang_vel', 'projected_gravity', 'command',
                                               'phase', 'joint_pos', 'joint_vel', 'actions']:
            raise ValueError('Unsupported actor observations')

    def observation(self, q_sdk, dq_sdk, quaternion, gyro, command, step, previous_action):
        phase_angle = 2 * math.pi * ((step * self.meta['step_dt']) % self.meta['phase_period']) / self.meta['phase_period']
        phase = [math.sin(phase_angle), math.cos(phase_angle)] if np.linalg.norm(command) >= 0.1 else [0., 0.]
        obs = np.concatenate((gyro, gravity_body(quaternion), command, phase,
                              np.asarray(q_sdk)[self.order] - self.default,
                              np.asarray(dq_sdk)[self.order], previous_action)).astype(np.float32)
        if obs.shape != (47,) or not np.isfinite(obs).all():
            raise ValueError('Invalid policy observation')
        return obs[None, :]

    def targets(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Invalid policy action')
        clip = self.meta['clip_actions']
        if clip is not None:
            action = np.clip(action, -clip, clip)
        targets = self.default + self.scale * action
        if np.any(targets < self.limits[:, 0]) or np.any(targets > self.limits[:, 1]):
            raise ValueError('Policy target outside joint limits')
        return action, self.to_sdk(targets)

    def to_sdk(self, values):
        result = np.empty(12, dtype=np.float32)
        result[self.order] = values
        return result
