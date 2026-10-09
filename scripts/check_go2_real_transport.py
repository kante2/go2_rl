#!/usr/bin/env python3
"""Exercise live arming against a fake Go2 on loopback DDS domain 94 ONLY.

No real robot transport is used. Tests ReleaseMode, CRC, 500 Hz LowCmd,
standing transition, model readiness, and damping on normal process exit.
Run via scripts/run_go2_docker.sh go2-real-python.
"""
import os

os.environ['ROS_DOMAIN_ID'] = '94'
os.environ['ROS_LOCALHOST_ONLY'] = '0'  # Explicit lo below; avoid selecting lo twice in this RMW.
os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI'] = '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General></Domain></CycloneDDS>'

import json
import argparse
from pathlib import Path
import pty
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data
from unitree_api.msg import Request, Response
from unitree_go.msg import LowCmd, LowState

from go2_autonomous_driving.real.real_control import PolicyMapping, pack_low_command, pack_low_state, wire_crc
from go2_autonomous_driving.real.direct_control import MeasuredStateReceiver, velocity_packet
from go2_autonomous_driving.core import Velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fault', choices=('none', 'sigint', 'estop', 'stale_state',
                                          'keyboard_timeout', 'joint_tracking'), default='none')
    parser.add_argument('--closed-loop', action='store_true', help='Compatibility flag; closed loop is the default')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    mapping = PolicyMapping(json.loads((project / 'artifacts/go2_real/policy.json').read_text()))
    rclpy.init()
    node = rclpy.create_node('fake_go2_transport_test')
    low = node.create_publisher(LowState, '/lowstate', qos_profile_sensor_data)
    response_pub = node.create_publisher(Response, '/api/motion_switcher/response', 10)
    firmware_pub = node.create_publisher(LowCmd, '/lowcmd', qos_profile_sensor_data)
    mode = ['normal']
    calls, motor_times, crc_errors, last_cmd, last_active = [], [], [], [], []
    fault_active = False

    def request(msg):
        calls.append(msg.header.identity.api_id)
        if msg.header.identity.api_id == 1003:
            mode[0] = ''
        response = Response()
        response.header.identity = msg.header.identity
        response.data = json.dumps(dict(form='0', name=mode[0]))
        response_pub.publish(response)

    def command(msg):
        if not mode[0]:
            motor_times.append(time.monotonic())
            last_cmd[:] = [msg]
            if msg.motor_cmd[0].kp > 0:
                last_active[:] = [msg]
            if wire_crc(pack_low_command(msg)) != msg.crc:
                crc_errors.append(msg.crc)

    node.create_subscription(Request, '/api/motion_switcher/request', request, 10)
    node.create_subscription(LowCmd, '/lowcmd', command, qos_profile_sensor_data)
    state = LowState()
    state.head = [0xFE, 0xEF]
    state.imu_state.quaternion = [1., 0., 0., 0.]
    for motor, q in zip(state.motor_state, mapping.to_sdk(mapping.default)):
        motor.q = float(q)
        motor.temperature = 25
    state.bms_state.soc = 100

    def state_tick():
        if fault_active and args.fault == 'stale_state':
            return
        if fault_active and args.fault == 'estop':
            state.wireless_remote[2], state.wireless_remote[3] = 0x20, 0x02
        if fault_active and args.fault == 'joint_tracking':
            state.motor_state[0].q = float(mapping.to_sdk(mapping.default)[0]) + .9
        state.tick += 2
        state.crc = wire_crc(pack_low_state(state))
        low.publish(state)

    def sensor_tick():
        if mode[0]:
            firmware_pub.publish(LowCmd())

    node.create_timer(.002, state_tick)
    node.create_timer(.05, sensor_tick)
    master, slave = pty.openpty()
    child = None
    log = ''
    command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    status_socket = None
    status_receiver = MeasuredStateReceiver(mapping.meta['checkpoint_sha256'])
    measured_commands = []
    with tempfile.TemporaryDirectory(prefix='go2_fake_dds_') as temporary:
        config = json.loads((project / 'src/go2_autonomous_driving/config/real_closed_loop.json').read_text())
        # Pick unused loopback UDP ports rather than a real backend's fixed ports.
        sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(3)]
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        config['command_port'], config['telemetry_port'] = [sock.getsockname()[1] for sock in sockets[:2]]
        config['keyboard_status_port'] = sockets[2].getsockname()[1]
        status_socket = sockets[2]
        status_socket.setblocking(False)
        for sock in sockets[:2]:
            sock.close()
        path = Path(temporary) / 'config.json'
        path.write_text(json.dumps(config))
        sequence = 0
        def direct_tick():
            nonlocal sequence
            now = time.monotonic()
            for _ in range(64):
                try:
                    payload, _ = status_socket.recvfrom(8192)
                except BlockingIOError:
                    break
                if status_receiver.receive(payload, now):
                    measured_commands.append((now, status_receiver.packet['command']))
                    # A telemetry frame queued just before the injected jump can
                    # legitimately contain the previous measured pose.
                    if not (fault_active and args.fault == 'joint_tracking'):
                        np.testing.assert_allclose(status_receiver.packet['q_policy'],
                                                   [state.motor_state[i].q for i in mapping.order], atol=1e-6)
            if not status_receiver.fresh(now) or not status_receiver.packet['ready']:
                return
            if fault_active and args.fault == 'keyboard_timeout':
                return
            command_socket.sendto(velocity_packet(Velocity(.2, 0, 0), sequence, now,
                                                status_receiver.packet['session'], 'fake-w'),
                                 ('127.0.0.1', config['command_port']))
            sequence += 1
        node.create_timer(.05, direct_tick)
        try:
            child = subprocess.Popen([
                sys.executable, '-m', 'go2_autonomous_driving.real.real_closed_loop',
                '--checkpoint', str(project / 'src/go2_autonomous_driving/model_pt/model_10000.pt'),
                '--policy-dir', str(project / 'artifacts/go2_real'),
                '--config', str(path), '--execute', '--seconds', '8',
                '--fault-log', str(Path(temporary) / 'fault.json'),
            ], stdin=slave, stdout=slave, stderr=slave, env=os.environ.copy())
            os.close(slave)
            armed = False
            ready_at = None
            deadline = time.monotonic() + 25
            while child.poll() is None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.001)
                if select.select([master], [], [], 0)[0]:
                    try:
                        data = os.read(master, 65536).decode(errors='replace')
                    except OSError:
                        break
                    log += data
                    print(data, end='', flush=True)
                if not armed and 'Type ARM to enable motors:' in log:
                    os.write(master, b'ARM\n')
                    armed = True
                ready_label = '\nREAL POLICY READY:'
                if ready_at is None and ready_label in log:
                    ready_at = time.monotonic()
                if args.fault != 'none' and ready_at is not None and not fault_active and time.monotonic() - ready_at > 1:
                    fault_active = True
                    if args.fault == 'sigint':
                        child.send_signal(signal.SIGINT)
            if child.poll() is None:
                child.wait(timeout=2)
            while select.select([master], [], [], 0)[0]:
                try:
                    data = os.read(master, 65536).decode(errors='replace')
                except OSError:
                    break
                if not data:
                    break
                log += data
                print(data, end='', flush=True)
            for _ in range(20):
                rclpy.spin_once(node, timeout_sec=.001)
            expected_code = 1 if args.fault in ('estop', 'stale_state', 'joint_tracking') else 0
            assert child.returncode == expected_code, f'Backend exit: {child.returncode}, expected {expected_code}'
            if args.fault != 'none':
                assert fault_active, 'Backend stopped before the requested fault was injected'
            expected_fault = {
                'estop': ('Remote L2+B emergency stop',),
                'stale_state': ('LowState missing/stale',),
                'joint_tracking': ('Real joints deviate from commanded targets: joint=',),
            }.get(args.fault)
            if expected_fault:
                assert any(reason in log for reason in expected_fault), f'Wrong stop reason for {args.fault}'
            assert calls.count(1003) == 1, calls
            assert ready_label in log and 'DAMPING' in log
            assert len(motor_times) > (3000 if args.fault == 'none' else 1500), len(motor_times)
            assert not crc_errors
            assert all(m.kp == 0 and m.kd == 2 for m in last_cmd[0].motor_cmd[:12])
            assert any(c[0] > .1 for _, c in measured_commands), 'Direct keyboard command never reached policy'
            if args.fault == 'keyboard_timeout':
                # Source is never resumed; final commands must be zero while standing remains live.
                assert all(max(abs(v) for v in c) == 0 for _, c in measured_commands[-15:])
            if args.fault == 'joint_tracking':
                fault_data = json.loads((Path(temporary) / 'fault.json').read_text())
                assert fault_data['tracking_failure'] is not None
                assert fault_data['recent_policy_frames']
                assert 'limit=0.650rad' in fault_data['fault']
            print(json.dumps(dict(result='PASS', fault=args.fault, domain=94, interface='lo',
                                  motor_packets=len(motor_times), crc_errors=len(crc_errors),
                                  motion_api_calls=calls), indent=2))
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            os.close(master)
            command_socket.close()
            if status_socket is not None:
                status_socket.close()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
