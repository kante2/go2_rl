#!/usr/bin/env python3
"""RL 자동 주행 중 ROS 2 로봇 상태와 모터 명령 수집.

RL 제어기는 별도 실행하고 사람이 ARM한다. 키보드는 종료한다.
  bash src/db_collector/data_collector_RL.sh --vx 0.2 --pre 5 --duration 20 --post 5 --out trial01
기본은 자동 실험이며 --passive는 주행 명령 없이 기록만 한다.
자동 실험: live 준비/센서 확인 -> pre -> duration 주행 -> 속도 0 확인 -> post -> 저장.
q/Q, Ctrl+C 또는 오류 시 속도 0을 요청하고 데이터를 저장한다. RL 제어기는 유지한다.
수신 중단/프로세스 정지 시 RL 제어기의 기존 command timeout도 적용된다.
속도 0 확인은 정책 입력 명령 기준이며 물리적 정지 보장은 아니다.

결과: db_collector/data/<이름>_<시각>.npz (로봇 상태·명령·이벤트).
상대 --out은 data 폴더 기준. 같은 이름의 NPZ는 저장 완료 후 덮어쓴다.
모든 시각은 PC monotonic 수신 기반이며 하드웨어 동기화 보장은 없다.
  python3 src/db_collector/scripts/RL_data_collector.py --recover <임시폴더> --out recovered
"""

import argparse
import json
import os
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
from datetime import datetime

import numpy as np

from rl_command_client import RLCommandClient, TrialTimeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def positive_seconds(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("유한한 양수 시간을 지정하세요.")
    return value


def configure_dds(interface, domain):
    """ROS 초기화 전에 실기 DDS를 선택한다. 기존 Wi-Fi/시뮬 설정과 분리."""
    if not 0 <= domain <= 232:
        raise ValueError("domain은 0~232여야 합니다.")
    if interface:
        if interface not in {name for _, name in socket.if_nameindex()}:
            raise ValueError("네트워크 인터페이스를 찾을 수 없습니다: " + interface)
        # XML attribute에 들어가는 이름을 제한한다.
        if not re.fullmatch(r"[\w.:-]+", interface):
            raise ValueError("지원하지 않는 인터페이스 이름: " + interface)
        os.environ["CYCLONEDDS_URI"] = (
            '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="'
            + interface + '"/></Interfaces></General></Domain></CycloneDDS>')
    os.environ.update(ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY="0",
                      RMW_IMPLEMENTATION="rmw_cyclonedds_cpp")


def stream_statistics(stream):
    records = stream.array()
    if len(records) < 2:
        return dict(count=len(records), hz=None, max_gap_ms=None)
    dt = np.diff(records["t"])
    return dict(count=len(records), hz=float(1 / dt.mean()) if dt.mean() > 0 else None,
                max_gap_ms=float(dt.max() * 1000))

# 기존 NPZ와 필드 이름, 축 길이, dtype을 일치시킨다.
SCHEMAS = {
    "ls": np.dtype([
        ("t", "<f8"), ("tick", "<i8"), ("ff", "<i2", (4,)),
        ("acc", "<f4", (3,)), ("gyro", "<f4", (3,)),
        ("rpy", "<f4", (3,)), ("quaternion", "<f4", (4,)),
        ("q", "<f8", (12,)), ("dq", "<f8", (12,)), ("tau", "<f8", (12,)),
    ]),
    "sp": np.dtype([
        ("t", "<f8"), ("mode", "<i8"), ("gait", "<i8"),
        ("raise_h", "<f8"), ("body_h", "<f8"), ("vel", "<f4", (3,)),
        ("yaw", "<f8"), ("pos", "<f4", (3,)), ("ff", "<i2", (4,)),
        ("foot_pos", "<f4", (12,)), ("foot_vel", "<f4", (12,)),
    ]),
    "lc": np.dtype([( "t", "<f8")] + [
        (key, "<f8", (12,)) for key in ("kp", "kd", "q", "dq", "tau")
    ]),
    "rl": np.dtype([
        ("t", "<f8"), ("source_t", "<f8"), ("sequence", "<i8"),
        ("command", "<f8", (3,)), ("requested", "<f8", (3,)),
        ("ready", "?"), ("execute", "?"), ("phase", "U64"),
        ("session", "U64"), ("checkpoint", "U64"),
    ]),
    "tx": np.dtype([("t", "<f8"), ("sequence", "<i8"),
                    ("command", "<f8", (3,)), ("intent", "U32")]),
}


class DiskStream:
    """단일 writer가 고정 길이 레코드를 디스크에 기록한다."""

    def __init__(self, directory, name, dtype):
        self.path = directory / (name + ".raw")
        self.dtype = np.dtype(dtype)
        self.file = self.path.open("wb")
        self.count = 0

    def append(self, values):
        self.file.write(np.array([values], dtype=self.dtype).tobytes())
        self.count += 1

    def close(self):
        self.file.close()

    def array(self):
        if not self.count:
            return np.empty(0, dtype=self.dtype)
        return np.memmap(self.path, dtype=self.dtype, mode="r", shape=(self.count,))


def save_recording(prefix, streams, events, meta):
    arrays = dict(ev_t=np.array([e[0] for e in events], dtype=np.float64),
                  ev_api=np.full(len(events), -1, dtype=np.int64),
                  ev_label=np.array([e[1] for e in events]))
    for name, stream in streams.items():
        records = stream.array()
        arrays.update({name + "_" + key: records[key] for key in records.dtype.names})
    meta["format_version"] = 2
    meta["stream_statistics"] = {name: stream_statistics(stream) for name, stream in streams.items()}
    arrays["meta"] = json.dumps(meta)
    # 기존 결과는 압축이 완료될 때까지 유지한다. 남은 임시 파일과도 충돌하지 않는다.
    with tempfile.NamedTemporaryFile(mode="wb", prefix="." + prefix.name + "_",
                                     suffix=".npz.tmp", dir=prefix.parent, delete=False) as handle:
        npz_tmp = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    npz_tmp.replace(str(prefix) + ".npz")
    print("저장: %s.npz" % prefix)
    for name, stats in meta["stream_statistics"].items():
        print("%s: %d개 | Hz=%s | 최대 수신 간격(ms)=%s" % (
            name, stats["count"], stats["hz"], stats["max_gap_ms"]))


class RecoveredStream:
    """임시 폴더의 .raw를 읽기 전용으로 연다. 마지막 미완성 레코드는 버린다."""

    def __init__(self, path, dtype):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.count = path.stat().st_size // self.dtype.itemsize if path.exists() else 0

    def array(self):
        if not self.count:
            return np.empty(0, dtype=self.dtype)
        return np.memmap(self.path, dtype=self.dtype, mode="r", shape=(self.count,))


def recover(directory, out=None):
    directory = directory.expanduser().resolve()
    if not (directory / "ls.raw").exists():
        raise SystemExit("임시 기록 폴더가 아닙니다 (ls.raw 없음): %s" % directory)
    if out is None:
        # mkdtemp가 붙인 "." 접두사와 "_xxxxxxxx" 접미사를 제거한다.
        out = directory.parent / directory.name.lstrip(".").rsplit("_", 1)[0]
    out = out.expanduser()
    prefix = (out if out.is_absolute() else DATA_DIR / out).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    streams = {name: RecoveredStream(directory / (name + ".raw"), dtype)
               for name, dtype in SCHEMAS.items()}
    times = [float(stream.array()["t"][[0, -1]].min()) for stream in streams.values() if stream.count]
    ends = [float(stream.array()["t"][-1]) for stream in streams.values() if stream.count]
    started, finished = (min(times), max(ends)) if times else (0., 0.)
    events = [(started, "record_start"), (finished, "record_stop"), (finished, "end")]
    meta = dict(controller="rl", passive=None, vx=float("nan"), vy=float("nan"), vyaw=float("nan"),
                duration=finished - started, pre=0., jump_wait=0., settle=0., post=0.,
                start_jump=False, end_jump=False, jump_id=-1, move=False,
                event_semantics="reconstructed recording boundaries; original trial events unknown",
                command_velocity_recorded=bool(streams['tx'].count),
                stop_reason="recovered", recovered_from=str(directory),
                counts={name: stream.count for name, stream in streams.items()},
                timestamp_clock="PC time.monotonic receive time")
    print("복구: %s" % directory, flush=True)
    print("counts: %s" % meta["counts"], flush=True)
    save_recording(prefix, streams, events, meta)
    print("원본 임시 폴더는 남겨 두었습니다. 결과 확인 후 직접 삭제하세요: %s" % directory)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recover", type=Path, metavar="DIR",
                        help="남은 임시 로봇 상태 기록 폴더를 .npz로 복구하고 종료 (--out으로 저장 prefix 지정 가능)")
    parser.add_argument("-n", "--name", help="저장 이름. 생략하면 실행 시 입력받는다.")
    parser.add_argument("-o", "--out", type=Path,
                        help="출력 prefix (.npz 제외). 상대 경로는 db_collector/data 기준, --name보다 우선")
    parser.add_argument("--low", default="/lowstate")
    parser.add_argument("--sport", default="/sportmodestate")
    parser.add_argument("--lowcmd", default="/lowcmd", help="RL이 발행하는 Unitree LowCmd 토픽")
    parser.add_argument("--interface", default=os.environ.get("GO2_NETWORK_INTERFACE"),
                        help="Go2 유선 인터페이스. 생략하면 기존 CycloneDDS 설정 사용")
    parser.add_argument("--domain", type=int, default=0, help="실기 DDS domain (기본 0)")
    parser.add_argument("-d", "--duration", type=positive_seconds,
                        help="자동 주행 시간 [s], 기본 20. passive에서는 준비 후 기록 시간(생략 시 q까지)")
    parser.add_argument("--passive", action="store_true", help="자동 명령 없이 기록만 수행")
    parser.add_argument("--vx", type=float, default=0., help="자동 주행 전후 속도 [m/s]")
    parser.add_argument("--vy", type=float, default=0., help="자동 주행 좌우 속도 [m/s]")
    parser.add_argument("--vyaw", type=float, default=0., help="자동 주행 회전 속도 [rad/s]")
    parser.add_argument("--pre", type=float, default=5., help="주행 전 정지 기록 [s]")
    parser.add_argument("--post", type=float, default=5., help="속도 0 확인 후 기록 [s]")
    parser.add_argument("--stop-timeout", type=positive_seconds, default=8., help="감속 정지 응답 제한 [s]")
    parser.add_argument("--model", choices=("baseline", "quietpaw", "quietpaw_long"), default="baseline",
                        help="RL 제어기와 같은 모델 설정 선택")
    parser.add_argument("--control-config", type=Path, help="RL 제어기와 동일한 모델 JSON (UDP 포트/상한)")
    parser.add_argument("--startup-timeout", type=positive_seconds, default=10.,
                        help="RL·필수 토픽 준비 제한 시간(초)")
    parser.add_argument("--state-timeout", type=positive_seconds, default=2.,
                        help="필수 토픽 수신이 끊기면 기록 종료·저장하는 시간(초)")
    parser.add_argument("--allow-missing-lowcmd", action="store_true",
                        help="dry 실행처럼 LowCmd가 없는 경우 허용")
    parser.add_argument("--policy-metadata", type=Path,
                        help="현재 RL 실행기가 사용하는 policy.json을 실험 메타데이터에 복사")
    args = parser.parse_args()
    if args.recover is not None:
        return recover(args.recover, args.out)
    velocity = np.array([args.vx, args.vy, args.vyaw])
    if not np.isfinite(velocity).all() or not all(np.isfinite(v) and v >= 0 for v in (args.pre, args.post)):
        parser.error("속도는 유한값, pre/post는 유한한 0 이상 값이어야 합니다.")
    if args.passive and np.any(velocity):
        parser.error("--passive와 주행 속도를 함께 지정할 수 없습니다.")
    if not args.passive and args.allow_missing_lowcmd:
        parser.error("자동 실험은 live LowCmd가 필요합니다. dry 기록은 --passive를 사용하세요.")
    if not args.passive and args.duration is None:
        args.duration = 20.
    control_config = None
    try:
        if not args.passive:
            config_path = args.control_config or (Path(__file__).resolve().parents[2]
                          / "go2_autodrive/config" / (args.model + ".json"))
            control_config = json.loads(config_path.read_text())
            for key in ("command_port", "keyboard_status_port"):
                if type(control_config[key]) is not int or not 1 <= control_config[key] <= 65535:
                    raise ValueError("유효하지 않은 UDP 포트: " + key)
            for key in ("command_timeout", "sensor_timeout", "max_forward_speed", "max_lateral_speed", "max_yaw_rate"):
                if not np.isfinite(control_config[key]) or control_config[key] <= 0:
                    raise ValueError("유효하지 않은 제어기 설정: " + key)
        configure_dds(args.interface, args.domain)
        policy_metadata = None
        if args.policy_metadata:
            policy_metadata = json.loads(args.policy_metadata.read_text())
            if not isinstance(policy_metadata, dict):
                raise ValueError("policy.json은 JSON object여야 합니다.")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if not sys.stdin.isatty():
        parser.error("q 입력을 받을 터미널에서 실행하세요.")
    if args.out is None:
        name = args.name
        while not name:
            try:
                name = input("저장 이름: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 1
        # 경로 구분자/공백 등은 _로 바꿔 data 폴더 밖에 저장되지 않게 한다.
        name = re.sub(r"[^\w.-]+", "_", name).strip("._") or "record"
        args.out = DATA_DIR / (name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = args.out.expanduser()
    prefix = (output if output.is_absolute() else DATA_DIR / output).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from rclpy.signals import SignalHandlerOptions
        from unitree_go.msg import LowState, SportModeState, LowCmd
    except ImportError as exc:
        parser.error("ROS 2 / unitree_ros2 환경을 source하세요: " + str(exc))

    directory = Path(tempfile.mkdtemp(prefix="." + prefix.name + "_", dir=prefix.parent))
    streams = {name: DiskStream(directory, name, dtype)
               for name, dtype in SCHEMAS.items()}
    stop = threading.Event()
    reason = "q"
    events = []
    last_received = {}
    required = ("ls",) if args.allow_missing_lowcmd else ("ls", "lc")
    ready_at = None
    client = None
    stop_confirmed = False
    trial = None if args.passive else TrialTimeline(args.pre, args.duration, args.post, args.stop_timeout)

    def log_state(now, packet):
        streams["rl"].append((now, packet['monotonic'], packet['sequence'], packet['command'],
            packet['requested_command'], packet['ready'], packet['execute'], packet['phase'],
            packet['session'], packet['checkpoint_sha256']))

    def log_send(now, sequence, command, intent):
        streams["tx"].append((now, sequence, command, intent))

    class Recorder(Node):
        def __init__(self):
            super().__init__("sync_recorder_rl")
            qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(LowState, args.low, self.low, qos)
            self.create_subscription(SportModeState, args.sport, self.sport, qos)
            self.create_subscription(LowCmd, args.lowcmd, self.command, qos)

        def low(self, msg):
            last_received["ls"] = stamp = time.monotonic()
            motors = msg.motor_state[:12]
            imu = msg.imu_state
            streams["ls"].append((stamp, msg.tick, msg.foot_force,
                imu.accelerometer, imu.gyroscope, imu.rpy, imu.quaternion,
                [m.q for m in motors], [m.dq for m in motors], [m.tau_est for m in motors]))

        def sport(self, msg):
            streams["sp"].append((time.monotonic(), msg.mode, msg.gait_type,
                msg.foot_raise_height, msg.body_height, msg.velocity, msg.yaw_speed,
                msg.position, msg.foot_force, msg.foot_position_body, msg.foot_speed_body))

        def command(self, msg):
            last_received["lc"] = stamp = time.monotonic()
            motors = msg.motor_cmd[:12]
            streams["lc"].append((stamp, *[
                [getattr(m, key) for m in motors] for key in ("kp", "kd", "q", "dq", "tau")]))

    def on_signal(number, frame):
        nonlocal reason
        reason = signal.Signals(number).name
        stop.set()

    node = None
    initialized = False
    terminal = termios.tcgetattr(sys.stdin.fileno())
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    try:
        # rclpy가 먼저 shutdown하면 마지막 상태 저장 중 예외가 생기므로 직접 처리한다.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        initialized = True
        node = Recorder()
        for sig in handlers:
            signal.signal(sig, on_signal)
        tty.setcbreak(sys.stdin.fileno())
        if not args.passive:
            client = RLCommandClient(control_config,
                expected_hash=policy_metadata.get('checkpoint_sha256') if policy_metadata else None,
                on_state=log_state, on_send=log_send)
        events.append((started, "record_start"))
        print("RL은 별도 ARM. 자동 실험 중 키보드 사용 금지. q → 정지 요청·저장" if client else "passive 기록: q → 저장 (로봇 제어 없음)", flush=True)
        print("저장 위치: %s.npz" % prefix, flush=True)
        print("임시 기록: %s" % directory, flush=True)
        next_status = started + 1
        while not stop.is_set():
            if select.select([sys.stdin], [], [], 0)[0]:
                if os.read(sys.stdin.fileno(), 1) in (b"q", b"Q"):
                    break
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if client:
                client.poll(now)
            missing = [name for name in required
                       if name not in last_received or now - last_received[name] > args.state_timeout]
            if ready_at is None:
                if not missing and (client is None or client.live_ready(now)):
                    if client:
                        client.activate(velocity, now)
                        trial.start(now)
                    ready_at = now
                    events.append((now, "ready"))
                    if args.passive:
                        events.append((now, "move_start"))
                    print("RECORD READY: " + ("자동 실험 pre 구간 시작" if client else "기록 전용"), flush=True)
                elif now - started > args.startup_timeout:
                    raise RuntimeError("기록 준비 시간 초과: missing=%s, RL_live=%s" % (missing, client.live_ready(now) if client else "passive"))
            elif missing:
                raise RuntimeError("필수 토픽 수신 중단: " + ", ".join(missing))
            if trial is not None and ready_at is not None:
                old_phase = trial.phase
                phase = trial.step(now, client.stopped(now) if client.stop_requested else False)
                if phase != old_phase:
                    events.append((now, "phase_" + phase))
                    print("TRIAL: " + phase, flush=True)
                    if phase == 'move':
                        client.set_velocity(velocity, now)
                        events.append((now, "move_start"))
                    elif phase == 'settle':
                        client.request_stop(now)
                        events.append((now, "move_stop"))
                    elif phase == 'post':
                        stop_confirmed = True
                        events.append((now, "zero_command_confirmed"))
                    elif phase == 'done':
                        reason = "completed"
                        break
            elif args.passive and args.duration is not None and ready_at is not None and now - ready_at >= args.duration:
                reason = "duration"
                break
            if not rclpy.ok():
                raise RuntimeError("ROS 컨텍스트 종료")
            if now >= next_status:
                print("%.1f초 | lowstate=%d sport=%d lowcmd=%d rl=%d tx=%d" % (
                    now - started, streams["ls"].count, streams["sp"].count,
                    streams["lc"].count, streams["rl"].count, streams["tx"].count), flush=True)
                if next_status == started + 1:
                    for name, topic in (("ls", args.low), ("sp", args.sport), ("lc", args.lowcmd)):
                        if not streams[name].count:
                            print("!! 수신 없음: %s (ROS_DOMAIN_ID / 토픽 / DDS 연결 확인)" % topic)
                next_status = now + 5
    except Exception as exc:
        reason = "error: " + str(exc)
        print("!! %s; 수신된 데이터까지 저장합니다." % reason, file=sys.stderr)
    finally:
        # Zero request before compression, including q/signals/errors.
        if client is not None:
            try:
                if client.session is not None:
                    client.request_stop()
                    if not any(label == 'move_stop' for _, label in events):
                        events.append((time.monotonic(), "move_stop"))
                    events.append((time.monotonic(), "stop_requested_on_exit"))
                    deadline = time.monotonic() + args.stop_timeout
                    while time.monotonic() < deadline:
                        try:
                            client.poll()
                        except RuntimeError:
                            pass  # fault is latched; poll only transmits zero now
                        if node is not None and rclpy.ok():
                            rclpy.spin_once(node, timeout_sec=.01)
                        if client.stopped(time.monotonic()):
                            stop_confirmed = True
                            break
                        time.sleep(.01)
                    if not stop_confirmed:
                        reason = "error: " + reason + "; RL 속도 0 응답 미확인"
                        print("!! 정지 응답 미확인. RL timeout/리모컨으로 로봇 상태를 확인하세요.", file=sys.stderr)
            except Exception as exc:
                reason = "error: 정지 전송 실패: " + str(exc)
            finally:
                client.close()
        finished = time.monotonic()
        if ready_at is not None and args.passive:
            events.append((finished, "move_stop"))
        events.extend([(finished, "record_stop"), (finished, "end")])
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal)
        # 저장 중 Ctrl+C/SIGTERM도 저장을 끊지 않도록 기존 handler는 저장 후 복원한다.
        for stream in streams.values():
            stream.close()
        if node is not None:
            node.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()

    meta = dict(controller="rl", passive=args.passive,
                vx=args.vx if client else float("nan"), vy=args.vy if client else float("nan"),
                vyaw=args.vyaw if client else float("nan"),
                duration=args.duration if client else finished - started,
                recording_duration=finished - started, pre=args.pre if client else 0.,
                jump_wait=0., settle=0., post=args.post if client else 0.,
                start_jump=False, end_jump=False, jump_id=-1, move=bool(client and np.any(velocity)),
                event_semantics="velocity requests" if client else "recording boundaries",
                command_velocity_recorded=client is not None, stop_reason=reason,
                stop_confirmed=stop_confirmed, stop_semantics="RL command zero, not physical velocity",
                control_config=control_config, session=client.session if client else None,
                checkpoint_sha256=client.checkpoint if client else None,
                ready=ready_at is not None, ready_t=ready_at,
                requested_duration=args.duration,
                ready_duration=finished - ready_at if ready_at is not None else 0.,
                interface=args.interface, ros_domain_id=args.domain,
                joint_order="SDK: FR, FL, RR, RL; hip, thigh, calf per leg",
                torque_semantics=dict(ls_tau="estimated motor torque", lc_tau="feedforward command torque"),
                sport_state_validated=False, policy_metadata=policy_metadata,
                topics=dict(low=args.low, sport=args.sport, lowcmd=args.lowcmd),
                counts={name: stream.count for name, stream in streams.items()},
                timestamp_clock="PC time.monotonic receive time")
    try:
        print("종료 사유: %s. 압축·저장 중..." % reason, flush=True)
        save_recording(prefix, streams, events, meta)
    except Exception:
        print("저장 실패. 복구용 원시 기록 보존: %s" % directory, file=sys.stderr)
        raise
    else:
        shutil.rmtree(directory)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if ready_at is None or not streams["ls"].count or reason.startswith("error:"):
        print("!! 불완전한 기록입니다. 저장된 meta와 토픽 수신 상태를 확인하세요.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
