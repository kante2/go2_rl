#!/usr/bin/env python3
"""실로봇 + 마이크 + 시뮬을 Enter 한 번으로 함께 녹화하는 상주 세션.

실로봇 제어(R2) 실행 중에 ROS 2 노드를 새로 띄우면, DDS 연결 과정에서 R2의 모터 루프가
10 ms 이상 멈추고 안전 정지된다. 이 프로그램은 R2보다 먼저 한 번만 실행해 DDS 연결을 유지하고,
녹화는 같은 프로세스 안에서 반복한다 (녹화를 시작·종료해도 새 DDS 참가자가 생기지 않는다).

  Enter  : 녹화 시작 (--duration 초 후 자동 저장; 녹화 중 Enter = 바로 저장)
  q      : 종료 (녹화 중이면 저장 후 종료)

저장: <dir>/<base>_v<N>_<YYYYmmdd_HHMMSS>.npz/.wav  (실로봇, sync_record_rl.py와 같은 형식)
      <dir>/<base>_v<N>_<YYYYmmdd_HHMMSS>_sim.npz   (시뮬, record_sim.py와 같은 형식)
N은 <dir>의 같은 <base> 파일을 보고 자동으로 매긴다. 시각은 모두 PC time.monotonic 수신 시각.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import select
import shutil
import signal
import socket
import sys
import tempfile
import termios
import threading
import time
import tty

from sync_record_rl import FS, SCHEMAS, DiskStream, RemoteAudio, save_recording
from record_sim import PROJECT_ROOT, SimRecorder


def next_version(directory, base):
    pattern = re.compile(re.escape(base) + r'_v(\d+)_')
    found = [int(m.group(1)) for p in directory.iterdir()
             if not p.name.startswith('.') and (m := pattern.match(p.name))]
    return max(found, default=0) + 1


class Take:
    """녹화 한 번. 콜백 스레드(ROS, 시뮬 UDP)는 add()로만 쓴다."""

    def __init__(self, directory, base, args, device):
        self.name = '%s_v%d' % (base, next_version(directory, base))
        self.prefix = directory / (self.name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        for suffix in ('.npz', '.wav', '_sim.npz'):
            if Path(str(self.prefix) + suffix).exists():
                raise RuntimeError('기존 파일이 있습니다: %s%s' % (self.prefix, suffix))
        self.tmp = Path(tempfile.mkdtemp(prefix='.' + self.prefix.name + '_', dir=directory))
        self.streams = {k: DiskStream(self.tmp, k, d) for k, d in SCHEMAS.items() if k != 'audio_clock'}
        self.audio = RemoteAudio(args.host, device, self.tmp)
        self.sim = SimRecorder(Path(str(self.prefix) + '_sim.npz')) if args.sim_port else None
        self.sim_gaps, self.sim_last = 0, None
        self.args, self.device = args, device
        self.lock = threading.Lock()
        self.closed = False
        self.started = time.monotonic()
        self.audio_ok = None
        self.events = [(self.started, 'move_start')]
        self.audio.start()

    def add(self, stream, row):
        with self.lock:
            if not self.closed:
                self.streams[stream].append(row)

    def add_sim(self, received, packet):
        with self.lock:
            if self.closed:
                return
            self.sim.add(received, packet)
            key = (packet['session'], packet['tick'])
            if self.sim_last and key[0] == self.sim_last[0] and key[1] != self.sim_last[1] + 1:
                self.sim_gaps += 1
            self.sim_last = key

    def status(self, now):
        return '%.0f초 | lowstate=%d sport=%d lowcmd=%d | audio=%.1f초%s' % (
            now - self.started, self.streams['ls'].count, self.streams['sp'].count,
            self.streams['lc'].count, self.audio.samples / FS,
            ' | sim=%d 끊김 %d' % (len(self.sim.rows), self.sim_gaps) if self.sim else '')

    def finish(self, reason):
        with self.lock:
            self.closed = True
        finished = time.monotonic()
        self.events += [(finished, 'move_stop'), (finished, 'end')]
        print('종료 사유: %s. 저장 중...' % reason, flush=True)
        try:
            self.audio.stop()
        finally:
            for stream in self.streams.values():
                stream.close()
        a = self.args
        meta = dict(controller='rl', passive=True, vx=float('nan'), vy=float('nan'), vyaw=float('nan'),
                    duration=finished - self.started, pre=0., jump_wait=0., settle=0., post=0.,
                    start_jump=False, end_jump=False, jump_id=-1, move=False,
                    event_semantics='move_start/stop delimit recording, not robot commands',
                    command_velocity_recorded=False, stop_reason=reason,
                    host=a.host or 'local', device=self.device,
                    topics=dict(low=a.low, sport=a.sport, lowcmd=a.lowcmd),
                    counts={k: s.count for k, s in self.streams.items()},
                    audio_error=self.audio.error, timestamp_clock='PC time.monotonic receive time',
                    recorder='record_session', sim_file=str(self.sim.path) if self.sim else None)
        try:
            save_recording(self.prefix, self.streams, self.audio, self.events, meta)
        except Exception:
            print('!! 실로봇 저장 실패. 복구용 원시 기록 보존: %s\n'
                  '   복구: python3 src/db_collector/sync_record_rl.py --recover %s' % (self.tmp, self.tmp),
                  file=sys.stderr, flush=True)
        else:
            shutil.rmtree(self.tmp)
        if self.sim:
            n = self.sim.save()
            print('저장: %s (물리 스텝 %d, 끊김 %d)' % (self.sim.path, n, self.sim_gaps), flush=True)
            if not n:
                print('!! 시뮬 수신 없음: terminal2_simulation.sh를 --state-port %d로 실행했는지 확인'
                      % a.sim_port, flush=True)
        if not self.audio.samples or not self.streams['ls'].count or reason.startswith('error'):
            print('!! 불완전한 기록입니다 (오디오/lowstate 확인).', file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, help='파일 이름 앞부분, 예: NORMAL_0.5_0.5 (뒤에 _v<N>_<시각>)')
    parser.add_argument('--dir', type=Path, default=Path('src/sim_to_real_data/real_sim'),
                        help='저장 폴더, 프로젝트 루트 기준')
    parser.add_argument('--duration', type=float, help='녹화 시간 [초], 오디오 수신 시작부터 (생략하면 Enter까지)')
    parser.add_argument('--sim-port', type=int, default=9881, help='시뮬 --state-port (0이면 시뮬 녹화 안 함)')
    parser.add_argument('--host', help='원격 마이크 SSH 주소 (예: unitree@192.168.123.18). 생략하면 이 PC 마이크')
    parser.add_argument('--card', help='ALSA 카드 이름 (원격 기본 Microphone)')
    parser.add_argument('--device', help='ALSA 장치 전체 이름')
    parser.add_argument('--low', default='/lowstate')
    parser.add_argument('--sport', default='/sportmodestate')
    parser.add_argument('--lowcmd', default='/lowcmd')
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error('Enter/q 입력을 받을 터미널에서 실행하세요.')
    base = re.sub(r'[^\w.-]+', '_', args.base).strip('._') or 'record'
    directory = args.dir if args.dir.is_absolute() else PROJECT_ROOT / args.dir
    directory.mkdir(parents=True, exist_ok=True)
    device = args.device or ('plughw:CARD=%s,DEV=0' % (args.card or 'Microphone')
                             if args.card or args.host else 'default')
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from rclpy.signals import SignalHandlerOptions
        from unitree_go.msg import LowState, SportModeState, LowCmd
    except ImportError as exc:
        parser.error('ROS 2 / unitree_ros2 환경을 source하세요: ' + str(exc))

    state = dict(take=None)
    counts = dict(ls=0, sp=0, lc=0, sim=0)
    quit_event = threading.Event()

    class Listener(Node):
        def __init__(self):
            super().__init__('record_session')
            qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(LowState, args.low, self.low, qos)
            self.create_subscription(SportModeState, args.sport, self.sport, qos)
            self.create_subscription(LowCmd, args.lowcmd, self.command, qos)

        def low(self, msg):
            counts['ls'] += 1
            take = state['take']
            if take is None:
                return
            motors, imu = msg.motor_state[:12], msg.imu_state
            take.add('ls', (time.monotonic(), msg.tick, msg.foot_force,
                            imu.accelerometer, imu.gyroscope, imu.rpy, imu.quaternion,
                            [m.q for m in motors], [m.dq for m in motors], [m.tau_est for m in motors],
                            [m.ddq for m in motors]))

        def sport(self, msg):
            counts['sp'] += 1
            take = state['take']
            if take is not None:
                take.add('sp', (time.monotonic(), msg.mode, msg.gait_type, msg.foot_raise_height,
                                msg.body_height, msg.velocity, msg.yaw_speed, msg.position, msg.foot_force,
                                msg.foot_position_body, msg.foot_speed_body))

        def command(self, msg):
            counts['lc'] += 1
            take = state['take']
            if take is not None:
                motors = msg.motor_cmd[:12]
                take.add('lc', (time.monotonic(), *[[getattr(m, k) for m in motors]
                                                    for k in ('kp', 'kd', 'q', 'dq', 'tau')]))

    def sim_loop(receiver):
        # 대기 중에도 계속 비워서, 녹화 시작 시 쌓인 옛 패킷이 들어가지 않게 한다.
        while not quit_event.is_set():
            try:
                payload, _ = receiver.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            received = time.monotonic()
            counts['sim'] += 1
            take = state['take']
            if take is None or take.sim is None:
                continue
            try:
                take.add_sim(received, json.loads(payload))
            except (ValueError, KeyError, TypeError) as error:
                print('!! 잘못된 시뮬 패킷: %s' % error, flush=True)

    def spin_loop(executor):
        while not quit_event.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    def on_signal(number, frame):
        quit_event.set()

    receiver = None
    if args.sim_port:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver.bind(('127.0.0.1', args.sim_port))
        except OSError:
            parser.error('UDP %d 사용 중: 기존 sim_record.sh(S3)를 끄세요. 이 세션이 시뮬도 녹화합니다.'
                         % args.sim_port)
        receiver.settimeout(.1)

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Listener()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threads = [threading.Thread(target=spin_loop, args=(executor,), daemon=True)]
    if receiver:
        threads.append(threading.Thread(target=sim_loop, args=(receiver,), daemon=True))
    for thread in threads:
        thread.start()
    handlers = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    terminal = termios.tcgetattr(sys.stdin.fileno())

    def prompt():
        print('\n대기 중 | 다음: %s_v%d | Enter: 녹화 시작%s | q: 종료' % (
            base, next_version(directory, base),
            ' (%g초)' % args.duration if args.duration else ''), flush=True)

    print('녹화 세션: %s → %s | 마이크 %s %s | 시뮬 UDP %s' % (
        base, directory, args.host or 'local', device, args.sim_port or '없음'), flush=True)
    print('※ 실로봇 제어(R2)보다 먼저 켜 두고, 끝날 때까지 이 세션을 유지하세요.', flush=True)
    try:
        tty.setcbreak(sys.stdin.fileno())
        before, mark = dict(counts), time.monotonic()
        time.sleep(2)
        rates = {k: (counts[k] - before[k]) / (time.monotonic() - mark) for k in counts}
        print('수신 확인: lowstate %.0fHz | sport %.0fHz | lowcmd %.0fHz | sim %.0fHz' % (
            rates['ls'], rates['sp'], rates['lc'], rates['sim']), flush=True)
        if not rates['ls']:
            print('!! lowstate 수신 없음 (로봇 연결 / ROS_DOMAIN_ID 확인)', flush=True)
        prompt()
        next_status = 0.
        while not quit_event.is_set():
            key = b''
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.buffer.raw.read(1)
            take = state['take']
            now = time.monotonic()
            if take is None:
                if key in (b'q', b'Q'):
                    break
                if key in (b'\n', b'\r'):
                    try:
                        state['take'] = take = Take(directory, base, args, device)
                    except Exception as error:
                        print('!! 녹화 시작 실패: %s' % error, flush=True)
                        prompt()
                        continue
                    print('● 녹화 시작: %s%s | Enter: 지금 저장 | q: 저장 후 종료' % (
                        take.prefix.name, ' (%g초 후 자동 저장)' % args.duration if args.duration else ''),
                        flush=True)
                    next_status = now + 1
                continue
            reason = None
            if key in (b'q', b'Q'):
                reason = 'q'
                quit_event.set()
            elif key in (b'\n', b'\r'):
                reason = 'enter'
            if take.audio_ok is None and take.audio.first.is_set():
                take.audio_ok = now
                take.events.append((now, 'audio_ok'))
            if reason is None:
                if take.audio.error or not take.audio.thread.is_alive() or take.audio.proc.poll() is not None:
                    reason = 'error: SSH/오디오 스트림 종료: ' + take.audio.error
                elif take.audio_ok is None and now - take.started > 10:
                    reason = 'error: 10초 동안 오디오 수신 없음'
                elif args.duration and take.audio_ok is not None and now - take.audio_ok >= args.duration:
                    reason = 'duration %gs' % args.duration
            if reason is None:
                if now >= next_status:
                    print(take.status(now), flush=True)
                    next_status = now + 5
                continue
            state['take'] = None
            print(take.status(now), flush=True)
            take.finish(reason)
            if not quit_event.is_set():
                prompt()
        if state['take'] is not None:  # Ctrl+C / SIGTERM 중 녹화
            take, state['take'] = state['take'], None
            take.finish('signal')
    finally:
        quit_event.set()
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for thread in threads:
            thread.join(timeout=2)
        if receiver:
            receiver.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        print('세션 종료', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
