"""MuJoCo/Viser displays measured Go2 posture. No physics steps or motor commands."""
import argparse
import errno
import json
from pathlib import Path
import signal
import socket
import time

import numpy as np

from .direct_control import MeasuredStateReceiver


def check_viewer_port(port):
    """Match the HTTP server's reuse behavior; TIME_WAIT is not a live listener."""
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(('0.0.0.0', port))


class MeasuredPose:
    """Write actual joints/quaternion into named model joints, never by XML order."""
    def __init__(self, model, metadata, height):
        self.model = model
        self.height = height
        self.q_addresses = [model.joint(name).qposadr[0] for name in metadata['joint_names']]
        self.dq_addresses = [model.joint(name).dofadr[0] for name in metadata['joint_names']]
        self.root_q = model.joint('floating_base_joint').qposadr[0]

    def apply(self, data, packet):
        import mujoco
        data.qvel[:] = 0
        data.qpos[self.root_q:self.root_q + 3] = [0, 0, self.height]
        quaternion = np.asarray(packet['quaternion_wxyz'])
        data.qpos[self.root_q + 3:self.root_q + 7] = quaternion / np.linalg.norm(quaternion)
        data.qpos[self.q_addresses] = packet['q_policy']
        data.qvel[self.dq_addresses] = packet['dq_policy']
        # Kinematics only: do not integrate gravity, contacts, policy, or position.
        mujoco.mj_forward(self.model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args()
    import mujoco
    import viser
    from mjlab.viewer.viser.scene import ViserMujocoScene

    config = json.loads(args.config.read_text())
    metadata = json.loads(args.metadata.read_text())
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', config['telemetry_port']))
    receiver.setblocking(False)
    stream = MeasuredStateReceiver(metadata['checkpoint_sha256'], config['sensor_timeout'])
    # Viser normally tries the next port if occupied. Fail clearly instead.
    try:
        check_viewer_port(args.port)
    except OSError as error:
        receiver.close()
        if error.errno != errno.EADDRINUSE:
            raise
        parser.exit(2, f'Viewer port {args.port} is already in use. Keep the robot controller running; '
                    f'close the previous viewer or run this viewer with --port {args.port + 1}\n')
    server = viser.ViserServer(port=args.port)
    if server.get_port() != args.port:
        server.stop()
        raise RuntimeError(f'Viewer port {args.port} is occupied; stop the previous viewer')
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    pose = MeasuredPose(model, metadata, config['display_height'])
    scene = ViserMujocoScene.create(server, model, 1)
    scene.camera_tracking_enabled = False
    # Hide the unmeasured initial model until the first valid sensor frame.
    visibility = [(handle, handle.visible) for handle in scene.mesh_handles_by_group.values()]
    for handle, _ in visibility:
        handle.visible = False
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/ground', width=4, height=4, plane='xy')
    server.gui.add_markdown('## 실로봇 상태\n실제 관절 각도와 IMU 자세를 표시합니다.\n\n'
                            f'몸체 위치는 (0, 0, {config["display_height"]}) m로 고정합니다. '
                            '실제 이동 거리·높이는 표시하지 않습니다.\n\n'
                            '**이 화면을 닫아도 모터 제어는 계속됩니다. 종료는 터미널 2에서 합니다.**')
    status = server.gui.add_text('수신 상태', initial_value='WAITING: 터미널 2를 실행하세요', disabled=True)
    phase = server.gui.add_text('제어 상태', initial_value='UNKNOWN', disabled=True)
    requested = server.gui.add_text('키보드 요청 [vx, vy, wz]', initial_value='—', disabled=True)
    command = server.gui.add_text('정책 입력 [vx, vy, wz]', initial_value='—', disabled=True)
    @server.on_client_connect
    def connected(client):
        client.camera.position = (1.3, -1.3, .85)
        client.camera.look_at = (0, 0, .25)
        client.camera.up_direction = (0, 0, 1)
    def interrupted(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    start = time.monotonic()
    frames = 0
    next_report = start
    try:
        while args.seconds is None or time.monotonic() - start < args.seconds:
            changed = False
            for _ in range(64):
                try:
                    payload, _ = receiver.recvfrom(8192)
                except BlockingIOError:
                    break
                changed = stream.receive(payload, time.monotonic()) or changed
            now = time.monotonic()
            if changed:
                pose.apply(data, stream.packet)
                for handle, visible in visibility:
                    handle.visible = visible
                scene.update_from_mjdata(data)
                frames += 1
            if stream.fresh(now):
                p = stream.packet
                status.value = f'LIVE · {(now - p["monotonic"]) * 1000:.0f} ms'
                phase.value = p['phase'] + (' · 모터 송신' if p['execute'] else ' · 모터 송신 없음')
                command.value = str([round(v, 3) for v in p['command']])
                requested.value = str([round(v, 3) for v in p['requested_command']]) if 'requested_command' in p else 'UNKNOWN'
            elif stream.packet is not None:
                status.value = 'STALE · 수신 중단, 마지막 자세 고정'
                phase.value = 'UNKNOWN · 터미널 2 확인'
                command.value = 'UNKNOWN'
                requested.value = 'UNKNOWN'
            if now >= next_report:
                print(f'VIEWER frames={frames} status={status.value} phase={phase.value}', flush=True)
                next_report = now + 2
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        server.stop()


if __name__ == '__main__':
    main()
