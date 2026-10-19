"""Learned Go2 control from actual LowState, plus one-way state visualization."""
import argparse
from collections import deque
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

from .direct_control import DirectCommandGate, keyboard_speed_settings
from .control_transitions import JointTargetRamp, VelocityRamp
from .real_robot import Backend, run


class ClosedLoopBackend(Backend):
    ready_label = 'REAL POLICY READY'
    ready_instruction = 'real sensor feedback active; terminal 3 W to move, Space to stop.'

    def __init__(self, robot, session, input_name):
        super().__init__(robot, session, input_name)
        self.direct = DirectCommandGate(self.config)
        self.command_ramp = VelocityRamp(self.config['command_acceleration'], self.config['command_deceleration'],
                                         self.mapping.meta['step_dt'])
        self.target_ramp = JointTargetRamp(
            self.config['max_joint_target_rate'], self.mapping.meta['step_dt'],
            transition_rate=self.config.get('phase_transition_joint_rate'),
            transition_seconds=self.config.get('phase_transition_seconds', 0.))
        self.requested_command = np.zeros(3)
        self.trace = deque(maxlen=100)
        self.tracking_failure = None
        self.telemetry_sequence = 0
        print('CLOSED LOOP: measured IMU/joints -> trained policy -> real joint targets.\n'
              'MuJoCo displays measured state only; viewer startup/exit does not control motors.', flush=True)

    def navigation_ready(self):
        # No odometry or cloud dependency. checked_state() supplies the fault guards.
        return self.robot.state is not None and time.monotonic() - self.robot.state_time < self.config['state_timeout']

    def command_sample(self):
        now = time.monotonic()
        if self.direct.session != self.session_id:
            self.command_ramp.reset()
        self.direct.set_session(self.session_id)
        enabled = self.navigation_ready() and (self.live_policy if self.execute else self.phase == 'DRY READY')
        for _ in range(32):
            try:
                payload, _ = self.receiver.recvfrom(4096)
            except BlockingIOError:
                break
            if enabled:
                self.direct.receive(payload, now)
        command = self.direct.sample(now)
        self.requested_command = np.array([command.vx, command.vy, command.wz]) if enabled else np.zeros(3)
        if not enabled or self.direct.intent is None or self.direct.intent == self.direct.expired_intent:
            # Sensor/keyboard loss still requests zero immediately; never ramp a
            # stale travel command. E-stop/faults go directly to damping in run().
            self.command_ramp.reset()
            return np.zeros(3)
        return self.command_ramp.step(self.requested_command)

    def policy_targets(self, output, obs, measured_q, stamp, reset):
        # Validate the raw model output before smoothing: out-of-range actions
        # still fault rather than being hidden by a slew limit.
        _, raw = self.mapping.targets(output)
        targets = self.target_ramp.step(raw, initial=measured_q if reset else None,
                                       moving=bool(np.any(obs[0, 9:11])))
        applied_action = (targets[self.mapping.order] - self.mapping.default) / self.mapping.scale
        # Previous-action observation must describe the filtered command sent to
        # the joints, not the unfiltered actor output.
        self.trace.append(dict(monotonic=stamp, observation=obs[0].tolist(),
                               requested_command=self.requested_command.tolist(), policy_command=self.command.tolist(),
                               phase_transition_remaining=self.target_ramp.transition_remaining,
                               measured_q_sdk=measured_q.tolist(), raw_target_sdk=raw.tolist(),
                               commanded_q_sdk=targets.tolist()))
        return applied_action.astype(np.float32), targets

    def check_tracking(self, targets, actual_q):
        try:
            super().check_tracking(targets, actual_q)
        except RuntimeError:
            self.tracking_failure = dict(monotonic=time.monotonic(), actual_q_sdk=actual_q.tolist(),
                                         target_q_sdk=targets.tolist(), error_sdk=np.abs(targets - actual_q).tolist())
            raise

    def close(self):
        super().close()
        if self.fault and self.config.get('fault_log'):
            try:
                path = Path(self.config['fault_log'])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(fault=self.fault, checkpoint_sha256=self.mapping.meta['checkpoint_sha256'],
                                                config=self.config, tracking_failure=self.tracking_failure,
                                                motor_timing_failure=self.motor_timing_failure,
                                                peak_publish_timing=getattr(self.robot, 'peak_publish_timing', None),
                                                recent_motor_timing=list(self.motor_timing),
                                                recent_policy_frames=list(self.trace)), indent=2, allow_nan=False) + '\n')
                print(f'Fault record: {path}', flush=True)
            except (OSError, ValueError) as error:
                print(f'Could not save fault record: {error}', file=sys.stderr, flush=True)

    def status(self):
        age = (time.monotonic() - self.robot.state_time) * 1000
        return (f'phase=[{self.phase}] state_age_ms={age:.1f} '
                f'requested_cmd={np.round(self.requested_command, 3).tolist()} '
                f'keyboard=[{self.direct.reason}] viewer_error={self.navigation_fault}')

    def telemetry(self):
        now = time.monotonic()
        if now < self.next_telemetry:
            return
        self.next_telemetry = now + 1 / 30
        # Capture before reading the message: a concurrently arriving newer state
        # is conservatively stamped older, never made artificially fresh.
        stamp = self.robot.state_time
        state, q, dq = self.robot.checked_state()
        data = dict(version=1, kind='go2_measured_state', monotonic=stamp,
                    session=self.session_id, sequence=self.telemetry_sequence,
                    checkpoint_sha256=self.mapping.meta['checkpoint_sha256'],
                    q_policy=q[self.mapping.order].tolist(), dq_policy=dq[self.mapping.order].tolist(),
                    quaternion_wxyz=[float(v) for v in state.imu_state.quaternion],
                    gyro=[float(v) for v in state.imu_state.gyroscope],
                    command=self.command.tolist(), requested_command=self.requested_command.tolist(),
                    execute=self.execute, phase=self.phase,
                    ready=self.phase in ('DRY READY', self.ready_label) and self.navigation_ready(),
                    position_mode='fixed_for_display', policy_steps=self.policy_steps)
        data['peak_publish_timing'] = getattr(self.robot, 'peak_publish_timing', None)
        data.update(keyboard_speed_settings(self.config))
        payload = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
        for port in (self.config['telemetry_port'], self.config['keyboard_status_port']):
            self.sender.sendto(payload, ('127.0.0.1', port))
        self.telemetry_sequence += 1
        self.telemetry_time = stamp
        self.telemetry_ms = (time.monotonic() - now) * 1000
        self.max_telemetry_ms = max(self.max_telemetry_ms, self.telemetry_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Motor output after interactive ARM')
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--fault-log', type=Path, help='Write fault details after damping/disconnecting')
    args = parser.parse_args()
    def interrupted(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        run(args, backend_type=ClosedLoopBackend, requires_navigation=False, send_telemetry=True)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'STOPPED: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
