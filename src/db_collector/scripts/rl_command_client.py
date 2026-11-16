"""RL direct-velocity UDP client. No Sport API, ARM or motor publisher.

Uses the controller's existing protocol; port ownership excludes its keyboard.
The caller must poll frequently. If it stalls, the controller's own command
watchdog expires; a stale status/session is never used to resume travel.
"""
import json
from pathlib import Path
import socket
import sys
import time
import uuid

import numpy as np

# Also works when launched as a standalone script from the shared repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "go2_autonomous_driving"))
from go2_autonomous_driving.core import Velocity
from go2_autonomous_driving.real.direct_control import MeasuredStateReceiver, velocity_packet


class RLCommandClient:
    def __init__(self, config, expected_hash=None, on_state=None, on_send=None):
        self.config = config
        self.state = MeasuredStateReceiver(expected_hash, timeout=config['sensor_timeout'])
        self.on_state, self.on_send = on_state, on_send
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # No SO_REUSEADDR: fail if a keyboard or another collector owns it.
            self.receiver.bind(('127.0.0.1', config['keyboard_status_port']))
            self.receiver.setblocking(False)
        except OSError:
            self.receiver.close()
            raise RuntimeError('RL 상태 포트 사용 중: 키보드/다른 수집기를 먼저 종료하세요.')
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session = None
        self.checkpoint = None
        self.sequence = 0
        self.intent = uuid.uuid4().hex
        self.command = np.zeros(3)
        self.next_send = 0.
        self.changed_at = None
        self.fault = None
        self.stop_requested = False
        self.period = min(.05, config['command_timeout'] / 3)

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        for _ in range(256):
            try:
                payload, _ = self.receiver.recvfrom(65536)
            except BlockingIOError:
                break
            try:
                packet = json.loads(payload)
                requested = np.asarray(packet['requested_command'], dtype=float)
                if requested.shape != (3,) or not np.isfinite(requested).all():
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            if self.state.receive(payload, now):
                if self.on_state:
                    self.on_state(now, self.state.packet)
        if self.session is not None:
            try:
                self.check_live(now)
                if (not self.stop_requested and self.changed_at is not None
                        and now - self.changed_at > .5
                        and not np.allclose(self.state.packet['requested_command'], self.command, atol=1e-5)):
                    raise RuntimeError('RL 요청 속도 불일치: 다른 명령 송신기 또는 설정을 확인하세요.')
            except RuntimeError as exc:
                self.fault = self.fault or str(exc)
            if self.fault:
                self.request_stop(now)
            if now >= self.next_send:
                self._send(now)
        if self.fault:
            raise RuntimeError(self.fault)

    def live_ready(self, now):
        p = self.state.packet
        return (self.state.fresh(now) and p['execute'] and p['ready']
                and p['phase'] == 'REAL POLICY READY')

    def check_live(self, now):
        if not self.live_ready(now):
            raise RuntimeError('RL live 상태 없음/지연/준비 해제: 주행 중단')
        p = self.state.packet
        if self.session is not None and (p['session'] != self.session or p['checkpoint_sha256'] != self.checkpoint):
            raise RuntimeError('RL 세션/정책 변경: 자동 재주행하지 않습니다.')

    def activate(self, velocity, now):
        if self.session is not None or self.fault or self.stop_requested:
            raise RuntimeError('자동 실험 클라이언트를 다시 활성화할 수 없습니다.')
        self.check_live(now)
        p = self.state.packet
        limits = np.minimum(p['velocity_limits'], [self.config[k] for k in
                            ('max_forward_speed', 'max_lateral_speed', 'max_yaw_rate')])
        values = np.asarray(velocity, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(np.abs(values) > limits):
            raise RuntimeError('요청 속도가 RL/설정 상한을 초과합니다: ' + str(limits.tolist()))
        if max(np.abs(p['command'])) > .01 or max(np.abs(p['requested_command'])) > .01:
            raise RuntimeError('RL이 이미 이동 중입니다. 정지 후 수집기를 시작하세요.')
        self.session, self.checkpoint = p['session'], p['checkpoint_sha256']
        self.limits = limits
        self.changed_at = now
        self._send(now)

    def set_velocity(self, velocity, now):
        if self.session is None or self.fault or self.stop_requested:
            raise RuntimeError(self.fault or '정지 요청 후 재주행 금지')
        self.check_live(now)
        values = np.asarray(velocity, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(np.abs(values) > self.limits):
            raise RuntimeError('유효하지 않은 속도 명령/상한 초과')
        if not np.array_equal(values, self.command):
            self.command = values.copy()
            self.intent = uuid.uuid4().hex
            self.changed_at = now
            self._send(now)

    def _send(self, now):
        if self.session is None:
            return
        self.sender.sendto(velocity_packet(Velocity(*self.command), self.sequence, now,
                                          self.session, self.intent),
                           ('127.0.0.1', self.config['command_port']))
        if self.on_send:
            self.on_send(now, self.sequence, self.command.copy(), self.intent)
        self.sequence += 1
        self.next_send = now + self.period

    def request_stop(self, now=None):
        now = time.monotonic() if now is None else now
        if not self.stop_requested:
            self.command = np.zeros(3)
            self.intent = uuid.uuid4().hex
            self.stop_requested = True
            self.changed_at = now
        self._send(now)

    def stopped(self, now):
        p = self.state.packet
        return (self.session is not None and self.live_ready(now)
                and p['session'] == self.session and p['monotonic'] >= self.changed_at
                and max(np.abs(p['requested_command'])) < 1e-5
                and max(np.abs(p['command'])) < 1e-3)

    def close(self):
        self.receiver.close()
        self.sender.close()


class TrialTimeline:
    """Pure automatic trial state machine. post begins after zero-command ACK."""
    def __init__(self, pre, duration, post, stop_timeout):
        self.pre, self.duration, self.post = pre, duration, post
        self.stop_timeout = stop_timeout
        self.started = None
        self.phase = 'waiting'
        self.stop_at = self.post_at = None

    def start(self, now):
        self.started = now
        self.phase = 'pre'

    def step(self, now, stopped=False):
        if self.phase == 'pre' and now - self.started >= self.pre:
            self.phase = 'move'
            self.move_at = now
        elif self.phase == 'move' and now - self.move_at >= self.duration:
            self.phase = 'settle'
            self.stop_at = now
        elif self.phase == 'settle':
            if stopped:
                self.phase = 'post'
                self.post_at = now
            elif now - self.stop_at > self.stop_timeout:
                raise RuntimeError('RL 적용 속도 0 확인 시간 초과 (물리적 정지 확인은 별도 필요)')
        elif self.phase == 'post' and now - self.post_at >= self.post:
            self.phase = 'done'
        return self.phase
