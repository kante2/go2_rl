"""Host Go2 DDS backend. Default is receive/infer only; --execute arms motors."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import socket
import signal
import sys
import threading
import time
import uuid

import numpy as np
import onnxruntime as ort
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from unitree_api.msg import Request, Response
from unitree_go.msg import LowCmd, LowState

from .core import CommandWatchdog
from .real_control import (PolicyMapping, gravity_body, next_period, pack_low_command,
                           pack_low_state, remote_estop, require_independent_navigation, wire_crc)


class Robot(Node):
    def __init__(self, config, mapping, *, subscribe_navigation=True):
        super().__init__('go2_learned_robot_' + uuid.uuid4().hex[:8])
        self.config, self.mapping = config, mapping
        self.state = self.odom = self.cloud = None
        self.state_time = self.odom_time = self.cloud_time = -math.inf
        self.last_tick = None
        self.bad_crc = self.state_count = 0
        self.last_other_command = -math.inf
        self.own_packets = deque(maxlen=128)
        self.responses = {}
        self.create_subscription(LowState, '/lowstate', self.on_state, qos_profile_sensor_data)
        self.create_subscription(LowCmd, '/lowcmd', self.on_command, qos_profile_sensor_data)
        if subscribe_navigation:
            self.create_subscription(Odometry, config['odom_topic'], self.on_odom, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, config['cloud_topic'], self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(Response, '/api/motion_switcher/response', self.on_response, 10)
        # CheckMode requests are read-only. LowCmd publisher is created only after live arming.
        self.api = self.create_publisher(Request, '/api/motion_switcher/request', 10)
        self.motor_publisher = None

    def on_state(self, msg):
        if wire_crc(pack_low_state(msg)) != msg.crc:
            self.bad_crc += 1
            return
        if self.last_tick is not None and not 0 < ((msg.tick - self.last_tick) & 0xFFFFFFFF) < 0x80000000:
            return
        self.last_tick = msg.tick
        self.state, self.state_time = msg, time.monotonic()
        self.state_count += 1

    def on_command(self, msg):
        # This host's rclpy does not expose the publisher GID in callbacks.
        # Compare complete wire payloads against our recent transmitted packets.
        if pack_low_command(msg) not in self.own_packets:
            self.last_other_command = time.monotonic()

    def on_odom(self, msg):
        if msg.header.frame_id.lstrip('/') == 'odom' and msg.child_frame_id.lstrip('/') == 'base_link':
            if self.odom is None or msg.header.stamp != self.odom.header.stamp:
                self.odom, self.odom_time = msg, time.monotonic()

    def on_cloud(self, msg):
        if msg.header.frame_id.lstrip('/') == 'odom':
            if self.cloud is None or msg.header.stamp != self.cloud.header.stamp:
                self.cloud, self.cloud_time = msg, time.monotonic()

    def on_response(self, msg):
        key = msg.header.identity.id
        if key in self.responses:
            self.responses[key] = msg

    def motion_api(self, api_id):
        request = Request()
        request.header.identity.id = time.time_ns()
        request.header.identity.api_id = api_id
        request.parameter = '{}'
        key = request.header.identity.id
        self.responses[key] = None
        try:
            self.api.publish(request)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                response = self.responses[key]
                if response is not None:
                    if response.header.status.code != 0:
                        raise RuntimeError(f'MotionSwitcher {api_id}: code {response.header.status.code}')
                    data = json.loads(response.data or '{}')
                    if api_id == 1001 and (not isinstance(data, dict) or not isinstance(data.get('name'), str)):
                        raise RuntimeError('Invalid CheckMode response; control ownership is unknown')
                    return data
                time.sleep(.01)
            raise RuntimeError(f'MotionSwitcher {api_id} timed out')
        finally:
            self.responses.pop(key, None)

    def sensor_fresh(self):
        now = time.monotonic()
        return (self.odom is not None and self.cloud is not None
                and now - min(self.odom_time, self.cloud_time) < self.config['sensor_timeout'])

    def sensor_status(self):
        now = time.monotonic()
        def age(stamp):
            return 'never' if not math.isfinite(stamp) else f'{now - stamp:.3f}s'
        return f'odom_age={age(self.odom_time)} cloud_age={age(self.cloud_time)}'

    def checked_state(self):
        state = self.state
        if state is None or time.monotonic() - self.state_time > self.config['state_timeout']:
            raise RuntimeError('LowState missing/stale')
        if remote_estop(state.wireless_remote):
            raise RuntimeError('Remote L2+B emergency stop')
        q = np.array([motor.q for motor in state.motor_state[:12]], dtype=np.float32)
        dq = np.array([motor.dq for motor in state.motor_state[:12]], dtype=np.float32)
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise RuntimeError('Non-finite motor state')
        gravity = gravity_body(state.imu_state.quaternion)
        if -gravity[2] < math.cos(math.radians(self.config['max_tilt_degrees'])):
            raise RuntimeError('Robot tilt exceeds limit')
        if max(m.temperature for m in state.motor_state[:12]) >= 75:
            raise RuntimeError('Motor temperature >= 75 C')
        return state, q, dq

    def create_motor_publisher(self):
        self.motor_publisher = self.create_publisher(LowCmd, '/lowcmd', 1)

    def publish_motors(self, targets, damping=False):
        msg = LowCmd()
        msg.head = [0xFE, 0xEF]
        msg.level_flag = 0xFF
        kp = self.mapping.to_sdk(self.mapping.meta['joint_stiffness'])
        kd = self.mapping.to_sdk(self.mapping.meta['joint_damping'])
        for index, motor in enumerate(msg.motor_cmd):
            motor.mode = 1
            motor.q, motor.dq = 2.146e9, 16000.0
            if index < 12:
                motor.q = float(targets[index]) if not damping else 2.146e9
                motor.dq = 0.0
                motor.kp = float(kp[index]) if not damping else 0.0
                motor.kd = float(kd[index]) if not damping else 2.0
        msg.crc = wire_crc(pack_low_command(msg))
        self.own_packets.append(pack_low_command(msg))
        self.motor_publisher.publish(msg)


class Backend:
    ready_label = 'POLICY READY'
    ready_instruction = 'start terminal 3, then W to move; Space to stop.'
    def __init__(self, robot, session, input_name):
        self.robot, self.config, self.mapping = robot, robot.config, robot.mapping
        self.session, self.input_name = session, input_name
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(('127.0.0.1', self.config['command_port']))
        self.receiver.setblocking(False)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.watchdog = CommandWatchdog(self.config['command_timeout'], self.config['max_forward_speed'],
                                       self.config['max_lateral_speed'], self.config['max_yaw_rate'])
        self.session_id = uuid.uuid4().hex
        self.reset_counter = 0
        self.stop = threading.Event()
        self.fault = None
        self.target = None
        self.command = np.zeros(3)
        self.policy_steps = 0
        self.live_policy = False
        self.execute = False
        self.output_started = False
        self.phase = 'PREFLIGHT'
        self.policy_ms = 0.0
        self.max_policy_ms = 0.0
        self.skipped_slots = 0
        self.last_skip = -math.inf
        self.telemetry_time = -math.inf
        self.telemetry_ms = 0.0
        self.max_telemetry_ms = 0.0
        self.navigation_fault = None
        self.next_telemetry = 0.0
        self.previous_position = None
        self.had_sensors = False

    def navigation_ready(self):
        return (self.robot.sensor_fresh() and self.navigation_fault is None
                and time.monotonic() - self.telemetry_time < self.config['sensor_timeout'])

    def ready_to_arm(self):
        return self.navigation_ready()

    def status(self):
        return (f'phase=[{self.phase}] nav={self.navigation_ready()} '
                f'{self.robot.sensor_status()} nav_fault={self.navigation_fault}')

    def policy_targets(self, output, obs, measured_q, stamp, reset):
        return self.mapping.targets(output)

    def check_tracking(self, targets, actual_q):
        limit = self.config.get('max_tracking_error', math.inf)
        error = np.abs(targets - actual_q)
        index = int(np.argmax(error))
        if error[index] > limit:
            policy_index = int(np.flatnonzero(self.mapping.order == index)[0])
            name = self.mapping.meta['joint_names'][policy_index]
            requested = getattr(self, 'requested_command', self.command)
            raise RuntimeError(
                f'Real joints deviate from commanded targets: joint={name} sdk_index={index} '
                f'error={error[index]:.3f}rad limit={limit:.3f}rad '
                f'actual={actual_q[index]:.3f} target={targets[index]:.3f} '
                f'requested_cmd={np.round(requested, 3).tolist()} '
                f'policy_cmd={np.round(self.command, 3).tolist()}; stop and inspect posture')

    def check_target(self):
        if self.fault:
            raise RuntimeError(self.fault)
        if self.stop.is_set():
            raise RuntimeError('Policy worker stopped')
        if self.target is None or time.monotonic() - self.target[0] > self.config['policy_timeout']:
            raise RuntimeError('Policy observation/output older than 100 ms')
        return self.target

    def preflight(self):
        """Exercise the warmed policy and telemetry BEFORE any controller release."""
        stable_since = None
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if self.fault or self.stop.is_set():
                raise RuntimeError(f'Preflight failed before release: {self.fault or "worker stopped"}')
            now = time.monotonic()
            valid = (self.target is not None and now - self.target[0] < 2 * self.mapping.meta['step_dt']
                     and self.ready_to_arm() and now - self.last_skip > .1)
            stable_since = (stable_since or now) if valid else None
            if stable_since is not None and now - stable_since >= 2:
                print(f'PREFLIGHT OK: policy_steps={self.policy_steps} max_infer_ms={self.max_policy_ms:.2f} '
                      f'skipped_slots={self.skipped_slots} {self.status()}; no motor output.', flush=True)
                return
            self.stop.wait(.02)
        raise RuntimeError(f'Preflight not stable; no controller release. {self.status()} '
                           f'skipped_slots={self.skipped_slots}')

    def command_sample(self):
        now = time.monotonic()
        fresh = self.navigation_ready()
        if self.had_sensors and not fresh:
            self.session_id = uuid.uuid4().hex
            self.watchdog = CommandWatchdog(self.config['command_timeout'], self.config['max_forward_speed'],
                                           self.config['max_lateral_speed'], self.config['max_yaw_rate'])
        self.had_sensors = fresh
        for _ in range(32):
            try:
                packet, _ = self.receiver.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                if json.loads(packet).get('session') != self.session_id:
                    continue
            except (ValueError, AttributeError):
                continue
            self.watchdog.receive(packet, now)
        cmd = self.watchdog.sample(now)
        enabled = fresh and (not self.execute or self.live_policy)
        return np.array([cmd.vx, cmd.vy, cmd.wz]) if enabled else np.zeros(3)

    def telemetry(self):
        now = time.monotonic()
        if now < self.next_telemetry or not self.robot.sensor_fresh():
            return
        self.next_telemetry = now + .1
        odom, cloud = self.robot.odom, self.robot.cloud
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        position = np.array([p.x, p.y, p.z])
        if self.previous_position is not None and np.linalg.norm(position - self.previous_position) > .5:
            raise RuntimeError('Odometry jumped by > 0.5 m')
        self.previous_position = position
        # Both topics already use odom: the planner's map is an alias of that
        # local frame. Do not transform the deskewed world cloud a second time.
        xyz = point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z'), skip_nans=True)
        xyz = np.asarray(xyz).reshape(-1, 3)
        distance = np.linalg.norm(xyz[:, :2] - position[:2], axis=1)
        xyz = xyz[(distance < self.config['scan_range']) & (distance > .35)
                  & (xyz[:, 2] > p.z - .22) & (xyz[:, 2] < p.z + .5)]
        # Voxel reduce and select nearest occupied cells if the scan is large.
        # Round coordinates also keeps UDP datagrams below 65 kB.
        if len(xyz):
            _, indices = np.unique(np.floor(xyz / .06).astype(np.int32), axis=0, return_index=True)
            xyz = xyz[indices]
            xyz = xyz[np.argsort(np.linalg.norm(xyz[:, :2] - position[:2], axis=1))[:1800]]
        v, w = odom.twist.twist.linear, odom.twist.twist.angular
        data = dict(version=1, monotonic=now, session=self.session_id, reset_counter=self.reset_counter,
                    position=position.tolist(), quaternion_wxyz=[q.w, q.x, q.y, q.z],
                    linear_velocity_body=[v.x, v.y, v.z], angular_velocity_body=[w.x, w.y, w.z],
                    points=[[round(float(value), 3) for value in row] for row in xyz])
        self.sender.sendto(json.dumps(data, separators=(',', ':'), allow_nan=False).encode(),
                           ('127.0.0.1', self.config['telemetry_port']))
        self.telemetry_time = now
        self.telemetry_ms = (time.monotonic() - now) * 1000
        self.max_telemetry_ms = max(self.max_telemetry_ms, self.telemetry_ms)

    def telemetry_loop(self):
        # Point-cloud decoding/voxel reduction never runs in the policy loop.
        while not self.stop.is_set():
            try:
                self.telemetry()
            except Exception as error:
                # Losing localization stops navigation, not the standing controller.
                self.navigation_fault = str(error)
                self.telemetry_time = -math.inf
                return
            self.stop.wait(.01)

    def infer_loop(self):
        previous = np.zeros(12, dtype=np.float32)
        previous_live = False
        deadline = phase_origin = time.monotonic()
        try:
            while not self.stop.is_set():
                state, q, dq = self.robot.checked_state()
                observation_time = self.robot.state_time
                started = time.monotonic()
                reset_targets = self.policy_steps == 0 or self.live_policy != previous_live
                if self.live_policy != previous_live:
                    phase_origin, previous = started, np.zeros(12, dtype=np.float32)
                    previous_live = self.live_policy
                step = round((started - phase_origin) / self.mapping.meta['step_dt'])
                command = self.command_sample()
                self.command = command
                obs = self.mapping.observation(q, dq, state.imu_state.quaternion,
                                               state.imu_state.gyroscope, command, step, previous)
                start = time.monotonic()
                output = self.session.run(None, {self.input_name: obs})[0]
                self.policy_ms = (time.monotonic() - start) * 1000
                self.max_policy_ms = max(self.max_policy_ms, self.policy_ms)
                previous, targets = self.policy_targets(output, obs, q, observation_time, reset_targets)
                if time.monotonic() - observation_time > self.config['policy_timeout']:
                    raise RuntimeError(f'Policy result expired before use: '
                                       f'{(time.monotonic() - observation_time) * 1000:.1f} ms')
                self.target = (observation_time, targets, previous_live)
                self.policy_steps += 1
                deadline, skipped = next_period(deadline, self.mapping.meta['step_dt'], time.monotonic())
                self.skipped_slots += skipped
                if skipped:
                    self.last_skip = time.monotonic()
                self.stop.wait(max(0, deadline - time.monotonic()))
        except Exception as error:
            self.fault = str(error)
            self.stop.set()

    def close(self):
        self.stop.set()
        self.receiver.close()
        self.sender.close()


def run(args, *, backend_type=Backend, requires_navigation=True, run_policy=True,
        send_telemetry=None):
    config = json.loads(args.config.read_text())
    if getattr(args, 'fault_log', None) is not None:
        config['fault_log'] = str(args.fault_log)
    if args.execute and requires_navigation:
        require_independent_navigation(config)
    metadata = json.loads((args.policy_dir / 'policy.json').read_text())
    policy_path = args.policy_dir / 'policy.onnx'
    if hashlib.sha256(policy_path.read_bytes()).hexdigest() != metadata['onnx_sha256']:
        raise RuntimeError('ONNX hash does not match policy metadata; export again')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != metadata['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed; export this checkpoint again')
    mapping = PolicyMapping(metadata)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = None
    if run_policy:
        session = ort.InferenceSession(str(policy_path), sess_options=options, providers=['CPUExecutionProvider'])
        if session.get_inputs()[0].shape != [1, 47] or session.get_outputs()[0].shape != [1, 12]:
            raise RuntimeError('Expected 47 observations and 12 actions')
        # Initialize ONNX kernels/allocations while no ROS motor publisher can exist.
        for _ in range(20):
            session.run(None, {session.get_inputs()[0].name: np.zeros((1, 47), dtype=np.float32)})
    # Keep the DDS context alive until the final damping packets are sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    robot = Robot(config, mapping, subscribe_navigation=requires_navigation)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(robot)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    backend = None
    inference = None
    telemetry = None
    motors_started = False
    released_mode = False
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and robot.state is None:
            time.sleep(.05)
        robot.checked_state()
        mode = robot.motion_api(1001)
        print(f'Connected: firmware={mode}, state={robot.state_count}, bad_crc={robot.bad_crc}', flush=True)
        while requires_navigation and time.monotonic() < deadline and not robot.sensor_fresh():
            time.sleep(.05)
        if requires_navigation and not robot.sensor_fresh():
            raise RuntimeError(f'Localization missing: {robot.sensor_status()}. '
                               f'Firmware mode={mode.get("name")!r}. Restore normal for default-topic dry-run; '
                               'learned walking needs independent localization.')
        backend = backend_type(robot, session, session.get_inputs()[0].name if session else None)
        backend.execute = args.execute
        if send_telemetry is None:
            send_telemetry = requires_navigation
        telemetry = threading.Thread(target=backend.telemetry_loop, daemon=True) if send_telemetry else None
        inference = threading.Thread(target=backend.infer_loop, daemon=True)
        if telemetry is not None:
            telemetry.start()
        inference.start()
        backend.preflight()
        if args.execute:
            backend.phase = 'WAITING ARM'
            if not sys.stdin.isatty():
                raise RuntimeError('--execute requires an interactive terminal for arming')
            print('LIVE: support the standing robot; keep remote L2+B accessible.\n'
                  'Typing ARM releases the built-in controller and moves to the learned standing pose\n'
                  f'over 3 seconds. Start keyboard control only after {backend.ready_label}.\n'
                  'Space in keyboard = zero velocity. Ctrl+C here = damping, then disconnect.', flush=True)
            if input('Type ARM to enable motors: ').strip() != 'ARM':
                print('Not armed; no motor commands sent.')
                return
            backend.preflight()  # Recheck after an arbitrarily long operator wait.
            state, initial_q, dq = robot.checked_state()
            if not backend.ready_to_arm() or np.max(np.abs(dq)) > 1.0:
                raise RuntimeError('Arming requires fresh sensors and a stationary robot')
            if np.max(np.abs(initial_q - mapping.to_sdk(mapping.default))) > .6:
                raise RuntimeError('Arming requires a supported standing pose near the trained pose')
            backend.check_target()
            mode = robot.motion_api(1001)
            if mode.get('name'):
                robot.motion_api(1003)  # ReleaseMode: live operator arming only.
                released_mode = True
            if robot.motion_api(1001).get('name'):
                raise RuntimeError('Built-in controller did not release control')
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and time.monotonic() - robot.last_other_command < .3:
                time.sleep(.01)
            if time.monotonic() - robot.last_other_command < .3:
                raise RuntimeError('Another LowCmd stream is still active; refusing motor output')
            _, initial_q, _ = robot.checked_state()
            robot.create_motor_publisher()
            backend.reset_counter += 1
            backend.phase = 'STANDING'
        else:
            backend.phase = 'DRY READY'
            print('DRY RUN: no LowCmd publisher; inspect outputs only.', flush=True)
        start = deadline = time.monotonic()
        next_report = start
        while not backend.stop.is_set() and (args.seconds is None or time.monotonic() - start < args.seconds):
            now = time.monotonic()
            _, actual_q, _ = robot.checked_state()
            if args.execute:
                if now - robot.last_other_command < .1:
                    motors_started = False  # Do not fight another controller, even with damping packets.
                    raise RuntimeError('Competing LowCmd stream detected')
                target = backend.check_target()
                alpha = min(1.0, (now - start) / config['stand_seconds'])
                if alpha < 1:
                    targets = initial_q + alpha * (mapping.to_sdk(mapping.default) - initial_q)
                else:
                    if not backend.live_policy and backend.ready_to_arm():
                        backend.live_policy = True
                        backend.session_id = uuid.uuid4().hex  # Clear any pre-arming waypoint.
                        backend.watchdog = CommandWatchdog(config['command_timeout'], config['max_forward_speed'],
                                                           config['max_lateral_speed'], config['max_yaw_rate'])
                        backend.phase = backend.ready_label
                        print(f'{backend.ready_label}: {backend.ready_instruction}', flush=True)
                    targets = target[1] if backend.live_policy and target[2] else mapping.to_sdk(mapping.default)
                backend.check_tracking(targets, actual_q)
                robot.publish_motors(targets)
                motors_started = True
                backend.output_started = True
            if now >= next_report:
                print(f'{"LIVE" if args.execute else "DRY"} states={robot.state_count} crc_errors={robot.bad_crc} '
                      f'inferences={backend.policy_steps} infer_ms={backend.policy_ms:.2f} '
                      f'max_infer_ms={backend.max_policy_ms:.2f} skipped={backend.skipped_slots} '
                      f'cloud_ms={backend.telemetry_ms:.2f} max_cloud_ms={backend.max_telemetry_ms:.2f} '
                      f'cmd={np.round(backend.command, 3).tolist()} {backend.status()}', flush=True)
                next_report = now + 2
            deadline += .002
            if deadline < time.monotonic() - .01:
                if args.execute:
                    raise RuntimeError('Motor loop missed its deadline')
                deadline = time.monotonic()
            time.sleep(max(0, deadline - time.monotonic()))
        if backend.fault:
            raise RuntimeError(backend.fault)
    except Exception as error:
        if backend is not None:
            backend.fault = str(error)
        raise
    finally:
        if backend is not None:
            backend.stop.set()
        if motors_started:
            # Latch on exit/fault. Never return to walking or re-enable normal mode automatically.
            print('DAMPING for 0.5 s, then disconnecting. Support the robot.', flush=True)
            for _ in range(100):
                robot.publish_motors(np.zeros(12), damping=True)
                time.sleep(.005)
        if inference is not None:
            inference.join(timeout=2)
        if telemetry is not None:
            telemetry.join(timeout=2)
        if released_mode:
            print('The built-in controller was released. Restore normal mode with the Unitree app/remote\n'
                  'after supporting the robot; this process does not restore it automatically.', flush=True)
        if backend is not None:
            backend.close()
        executor.shutdown(timeout_sec=2)
        spin.join(timeout=2)
        robot.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--seconds', type=float, help='Duration limit, mainly for dry-run checks')
    args = parser.parse_args()
    def interrupted(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'STOPPED: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
