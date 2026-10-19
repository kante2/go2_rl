#!/usr/bin/env python3
"""RL 보행 중 ROS 2 상태 + 마이크를 기록하고 q로 종료한다.

PC에서 ROS 2 / unitree_ros2 환경을 source하고 RL 정책은 별도로 실행한다.
실행하면 저장 이름을 묻고 data/<이름>_<YYYYmmdd_HHMMSS>.npz/.wav로 저장한다.
  python3 src/db_collector/sync_record_rl.py
  python3 src/db_collector/sync_record_rl.py --name walk_01 --lowcmd /policy/lowcmd
  python3 src/db_collector/sync_record_rl.py --host unitree@192.168.123.18  # 로봇 마이크(SSH)

기본은 이 PC에 연결된 마이크(ALSA default)로 녹음한다.

저장 실패/강제 종료로 남은 임시 폴더(data/.<이름>_<시각>_xxxxxxxx)는 복구한다.
  python3 src/db_collector/sync_record_rl.py --recover src/db_collector/data/.walk_01_20261019_153050_abcd1234

q/Q는 Enter 없이 종료·저장한다. Ctrl+C도 저장한다. 로봇 제어 명령은
전혀 발행하지 않으므로 기록을 종료해도 RL 정책과 로봇은 계속 동작한다.
RL 실행기가 DDS를 직접 사용하면 ROS 2에서 토픽이 보이도록 별도 연결이
필요하다. 토픽 이름과 메시지 타입은 기존 Unitree Go2 기록기와 같다.

data/1007과 같은 <prefix>.npz + 48 kHz / mono / PCM16 <prefix>.wav를
생성한다. CSV는 기존 preprocessing 단계에서 생성한다. 시간은 모두 PC의
monotonic 수신 시각이며 SSH 전달 지연을 포함한다. 물리적 동기화 보장은 없다.
move_start/move_stop은 후처리 호환용 '기록 구간' 표식이며 동작 명령이 아니다.
속도 명령은 관측하지 않으므로 meta의 vx/vy/vyaw는 NaN으로 저장한다.

시간 제한 없이 디스크에 임시 기록한다. 정상 저장 후 임시 폴더를 지우며,
저장 실패 시 복구용 임시 폴더를 남긴다. WAV(RIFF) 한계인 약 4 GiB보다
긴 단일 녹음은 파일을 나눠 수집해야 한다.
"""

import argparse
import json
import os
from pathlib import Path
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import wave
from datetime import datetime

import numpy as np

FS = 48000
DATA_DIR = Path(__file__).resolve().parent / "data"

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
    "audio_clock": np.dtype([("t", "<f8"), ("n_end", "<i8")]),
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


class RemoteAudio:
    """host가 None이면 이 PC의 마이크, 아니면 SSH로 원격 마이크를 녹음한다."""

    def __init__(self, host, device, directory):
        self.host = host
        self.path = directory / "audio.raw"
        self.clock = DiskStream(directory, "audio_clock", SCHEMAS["audio_clock"])
        self.file = self.path.open("wb")
        self.samples = 0
        self.first = threading.Event()
        self.error = ""
        self.stderr = tempfile.TemporaryFile()
        # stdin EOF가 오면 이 세션의 arecord PID만 종료한다. 다른 녹음은 건드리지 않는다.
        self.arecord = [
            "arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(FS),
            "-c", "1", "-t", "raw", "--period-time=10000", "--buffer-time=100000",
        ]
        command = shlex.join(self.arecord)
        self.remote_command = (
            'exec 3<&0; ' + command + ' </dev/null 3<&- & recorder=$!; '
            'trap \'kill "$recorder" 2>/dev/null || :\' HUP INT TERM EXIT; '
            '(cat >/dev/null; kill "$recorder" 2>/dev/null || :) <&3 3<&- >&2 & watcher=$!; '
            'exec 3<&-; '
            'wait "$recorder"; status=$?; '
            'kill "$watcher" 2>/dev/null || :; wait "$watcher" 2>/dev/null; '
            'trap - HUP INT TERM EXIT; exit "$status"'
        )
        self.proc = None
        self.thread = None

    def start(self):
        if self.host is None:
            self.proc = subprocess.Popen(self.arecord, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0)
        else:
            self.proc = subprocess.Popen([
                "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                "-o", "Compression=no", "-o", "ServerAliveInterval=2",
                "-o", "ServerAliveCountMax=3", self.host, self.remote_command,
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        pending = b""
        try:
            while True:
                data = os.read(self.proc.stdout.fileno(), 65536)
                if not data:
                    break
                arrived = time.monotonic()
                pending += data
                size = len(pending) // 2 * 2
                if size:
                    self.file.write(pending[:size])
                    self.samples += size // 2
                    self.clock.append((arrived, self.samples))
                    pending = pending[size:]
                    self.first.set()
        except Exception as exc:
            self.error = str(exc)

    def stop(self):
        if self.proc is not None:
            if self.proc.stdin is not None:
                self.proc.stdin.close()  # remote watcher가 이 녹음만 종료
            else:
                self.proc.terminate()  # 로컬 arecord 종료
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError("오디오 수신 스레드가 종료되지 않았습니다.")
            self.proc.stdout.close()
        self.file.close()
        self.clock.close()
        self.stderr.seek(0)
        self.error = "\n".join(filter(None, [self.error, self.stderr.read().decode(errors="replace").strip()]))
        self.stderr.close()

    def arrays(self):
        clock = self.clock.array()
        times, ends = clock["t"], clock["n_end"]
        if not self.samples:
            return np.empty(0, dtype="<i2"), 1 / FS, 0., times, ends
        audio = np.memmap(self.path, dtype="<i2", mode="r", shape=(self.samples,))
        # 짧은 기록에서도 polyfit 실패 없이 저장한다. 긴 기록은 기존 하한선 fit 유지.
        slope = 1 / FS
        if len(times) >= 2:
            centered = times - times[0]
            estimate = np.polyfit(ends, centered, 1)[0]
            if np.isfinite(estimate) and estimate > 0:
                slope = float(estimate)
        t0 = float(np.min(times - slope * ends))
        return audio, slope, t0, times, ends


def save_recording(prefix, streams, audio, events, meta):
    samples, slope, t0, times, ends = audio.arrays()
    arrays = dict(audio=samples, fs=FS, audio_slope=slope, audio_t0=t0,
                  audio_cb_t=times, audio_n_end=ends, audio_status=np.array([]),
                  ev_t=np.array([e[0] for e in events], dtype=np.float64),
                  ev_api=np.full(len(events), -1, dtype=np.int64),
                  ev_label=np.array([e[1] for e in events]), meta=json.dumps(meta))
    for name, stream in streams.items():
        records = stream.array()
        arrays.update({name + "_" + key: records[key] for key in records.dtype.names})
    # 큰 npz 압축 도중 중단되더라도 완료된 파일로 보이지 않도록 임시 이름 사용.
    npz_tmp = Path(str(prefix) + ".npz.tmp")
    wav_tmp = Path(str(prefix) + ".wav.tmp")
    with npz_tmp.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    if samples.nbytes > 0xFFFFFFFF - 36:
        raise ValueError("WAV의 4 GiB 한계를 넘었습니다. 임시 audio.raw를 분할해 복구하세요.")
    with wave.open(str(wav_tmp), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(FS)
        for start in range(0, len(samples), FS * 60):
            handle.writeframes(np.ascontiguousarray(samples[start:start + FS * 60]).tobytes())
    npz_tmp.rename(str(prefix) + ".npz")
    wav_tmp.rename(str(prefix) + ".wav")
    print("저장: %s.npz / %s.wav (오디오 %.2f초)" % (prefix, prefix, len(samples) / FS))
    if len(times) < 2:
        print("!! 오디오가 너무 짧아 nominal 48000 Hz로 시간축을 저장했습니다.")
    elif abs(1 / slope - FS) > 10:
        print("!! 추정 오디오 샘플레이트 %.2f Hz: 누락/전달 지연을 확인하세요." % (1 / slope))


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
    if not (directory / "audio.raw").exists():
        raise SystemExit("임시 기록 폴더가 아닙니다 (audio.raw 없음): %s" % directory)
    if out is None:
        # mkdtemp가 붙인 "." 접두사와 "_xxxxxxxx" 접미사를 제거한다.
        out = directory.parent / directory.name.lstrip(".").rsplit("_", 1)[0]
    prefix = out.expanduser().resolve()
    for suffix in (".npz", ".wav", ".npz.tmp", ".wav.tmp"):
        if Path(str(prefix) + suffix).exists():
            raise SystemExit("기존 파일이 있습니다. --out으로 다른 이름을 지정하세요: %s%s" % (prefix, suffix))
    streams = {name: RecoveredStream(directory / (name + ".raw"), dtype)
               for name, dtype in SCHEMAS.items() if name != "audio_clock"}
    audio = RemoteAudio.__new__(RemoteAudio)
    audio.path = directory / "audio.raw"
    audio.clock = RecoveredStream(directory / "audio_clock.raw", SCHEMAS["audio_clock"])
    audio.samples = audio.path.stat().st_size // 2
    # clock에 기록된 마지막 샘플까지만 신뢰한다.
    clock = audio.clock.array()
    if len(clock):
        audio.samples = min(audio.samples, int(clock["n_end"][-1]))
    times = [float(stream.array()["t"][[0, -1]].min()) for stream in streams.values() if stream.count]
    ends = [float(stream.array()["t"][-1]) for stream in streams.values() if stream.count]
    if len(clock):
        times.append(float(clock["t"][0]))
        ends.append(float(clock["t"][-1]))
    started, finished = (min(times), max(ends)) if times else (0., 0.)
    events = [(started, "move_start"), (finished, "move_stop"), (finished, "end")]
    meta = dict(controller="rl", passive=True, vx=float("nan"), vy=float("nan"), vyaw=float("nan"),
                duration=finished - started, pre=0., jump_wait=0., settle=0., post=0.,
                start_jump=False, end_jump=False, jump_id=-1, move=False,
                event_semantics="move_start/stop delimit recording, not robot commands",
                command_velocity_recorded=False, stop_reason="recovered", recovered_from=str(directory),
                counts={name: stream.count for name, stream in streams.items()},
                timestamp_clock="PC time.monotonic receive time")
    print("복구: %s" % directory, flush=True)
    print("counts: %s | audio=%.1f초" % (meta["counts"], audio.samples / FS), flush=True)
    save_recording(prefix, streams, audio, events, meta)
    print("원본 임시 폴더는 남겨 두었습니다. 결과 확인 후 직접 삭제하세요: rm -r %s" % shlex.quote(str(directory)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recover", type=Path, metavar="DIR",
                        help="남은 임시 기록 폴더를 .npz/.wav로 복구하고 종료 (--out으로 저장 prefix 지정 가능)")
    parser.add_argument("-n", "--name", help="저장 이름. 생략하면 실행 시 입력받는다.")
    parser.add_argument("-o", "--out", type=Path,
                        help="출력 prefix 직접 지정 (.npz/.wav 제외, --name보다 우선)")
    parser.add_argument("--host", help="원격 마이크 SSH 주소 (예: unitree@192.168.123.18). 생략하면 이 PC 마이크")
    parser.add_argument("--card", help="ALSA 카드 이름 (원격 기본 Microphone)")
    parser.add_argument("--device", help="ALSA 장치 전체 이름 (지정하면 --card보다 우선, 로컬 기본 default)")
    parser.add_argument("--low", default="/lowstate")
    parser.add_argument("--sport", default="/sportmodestate")
    parser.add_argument("--lowcmd", default="/lowcmd", help="RL이 발행하는 Unitree LowCmd 토픽")
    args = parser.parse_args()
    if args.recover is not None:
        return recover(args.recover, args.out)
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
    if args.device:
        device = args.device
    elif args.card or args.host:
        device = "plughw:CARD=%s,DEV=0" % (args.card or "Microphone")
    else:
        device = "default"
    prefix = args.out.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".wav", ".npz.tmp", ".wav.tmp"):
        if Path(str(prefix) + suffix).exists():
            parser.error("기존 파일이 있습니다. 다른 --out을 사용하세요: " + str(prefix) + suffix)
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
               for name, dtype in SCHEMAS.items() if name != "audio_clock"}
    audio = RemoteAudio(args.host, device, directory)
    stop = threading.Event()
    reason = "q"
    events = []

    class Recorder(Node):
        def __init__(self):
            super().__init__("sync_recorder_rl")
            qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(LowState, args.low, self.low, qos)
            self.create_subscription(SportModeState, args.sport, self.sport, qos)
            self.create_subscription(LowCmd, args.lowcmd, self.command, qos)

        def low(self, msg):
            motors = msg.motor_state[:12]
            imu = msg.imu_state
            streams["ls"].append((time.monotonic(), msg.tick, msg.foot_force,
                imu.accelerometer, imu.gyroscope, imu.rpy, imu.quaternion,
                [m.q for m in motors], [m.dq for m in motors], [m.tau_est for m in motors]))

        def sport(self, msg):
            streams["sp"].append((time.monotonic(), msg.mode, msg.gait_type,
                msg.foot_raise_height, msg.body_height, msg.velocity, msg.yaw_speed,
                msg.position, msg.foot_force, msg.foot_position_body, msg.foot_speed_body))

        def command(self, msg):
            motors = msg.motor_cmd[:12]
            streams["lc"].append((time.monotonic(), *[
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
        audio.start()
        events.append((started, "move_start"))
        print("RL 정책은 별도 실행하세요. 기록 시작: q → 종료·저장 (로봇 보행은 계속됨)", flush=True)
        print("저장 위치: %s.npz/.wav | 마이크: %s %s" % (
            prefix, args.host or "local", device), flush=True)
        print("임시 기록: %s" % directory, flush=True)
        next_status = started + 1
        audio_marked = False
        while not stop.is_set():
            if select.select([sys.stdin], [], [], 0)[0]:
                if os.read(sys.stdin.fileno(), 1) in (b"q", b"Q"):
                    break
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if audio.first.is_set() and not audio_marked:
                events.append((now, "audio_ok"))
                audio_marked = True
            if audio.error or not audio.thread.is_alive() or audio.proc.poll() is not None:
                raise RuntimeError("SSH/오디오 스트림 종료: " + audio.error)
            if not audio.first.is_set() and now - started > 10:
                raise RuntimeError("10초 동안 오디오 수신 없음: 마이크 연결을 확인하세요.")
            if not rclpy.ok():
                raise RuntimeError("ROS 컨텍스트 종료")
            if now >= next_status:
                print("%.1f초 | lowstate=%d sport=%d lowcmd=%d | audio=%.1f초" % (
                    now - started, streams["ls"].count, streams["sp"].count,
                    streams["lc"].count, audio.samples / FS), flush=True)
                if next_status == started + 1:
                    for name, topic in (("ls", args.low), ("sp", args.sport), ("lc", args.lowcmd)):
                        if not streams[name].count:
                            print("!! 수신 없음: %s (ROS_DOMAIN_ID / 토픽 / DDS 연결 확인)" % topic)
                next_status = now + 5
    except Exception as exc:
        reason = "error: " + str(exc)
        print("!! %s; 수신된 데이터까지 저장합니다." % reason, file=sys.stderr)
    finally:
        finished = time.monotonic()
        events.extend([(finished, "move_stop"), (finished, "end")])
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal)
        # 저장 중 Ctrl+C/SIGTERM도 저장을 끊지 않도록 기존 handler는 저장 후 복원한다.
        try:
            audio.stop()
        finally:
            for stream in streams.values():
                stream.close()
            if node is not None:
                node.destroy_node()
            if initialized and rclpy.ok():
                rclpy.shutdown()

    meta = dict(controller="rl", passive=True, vx=float("nan"), vy=float("nan"), vyaw=float("nan"),
                duration=finished - started, pre=0., jump_wait=0., settle=0., post=0.,
                start_jump=False, end_jump=False, jump_id=-1, move=False,
                event_semantics="move_start/stop delimit recording, not robot commands",
                command_velocity_recorded=False, stop_reason=reason,
                host=args.host or "local", device=device,
                topics=dict(low=args.low, sport=args.sport, lowcmd=args.lowcmd),
                counts={name: stream.count for name, stream in streams.items()},
                audio_error=audio.error, timestamp_clock="PC time.monotonic receive time")
    try:
        print("종료 사유: %s. 압축·저장 중..." % reason, flush=True)
        save_recording(prefix, streams, audio, events, meta)
    except Exception:
        print("저장 실패. 복구용 원시 기록 보존: %s" % directory, file=sys.stderr)
        raise
    else:
        shutil.rmtree(directory)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if not audio.samples or not streams["ls"].count or reason.startswith("error:"):
        print("!! 불완전한 기록입니다. 저장된 meta와 토픽 수신 상태를 확인하세요.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
