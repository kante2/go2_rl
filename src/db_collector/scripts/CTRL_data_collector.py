#!/usr/bin/env python3
"""
sync_record_remote.py - PC(원격)에서 전부 기록 + 주행
  * 오디오          : SSH 로 Jetson 의 arecord 원시 스트림을 받아옴
  * /lowstate       : PC 의 ROS2(Humble) 로 직접 구독
  * /sportmodestate : 〃
  * /lowcmd         : 〃 (kp, kd, 목표 q/dq/tau) - MotorState 에는 kp/kd 가 없어서 명령 쪽에서 받음
  -> 전부 PC 의 time.monotonic() 하나로 시각을 찍음

실행 순서 (타임라인)
  기록 시작 -> 오디오 수신 확인
  -> [pre]       정지 상태 대기 (기준 소음)
  -> [주행]      입력 속도로 duration 초
  -> StopMove -> [settle] 정지 안정화
  -> [post]      꼬리 구간 -> 기록 종료

사용법
  python3 sync_record_remote.py -d 30 -o data/0.1_1 --vx 0.1
  python3 sync_record_remote.py -d 30 -o data/0.0_1 --move          # 0 m/s 명령 (제자리걸음 확인용)
  python3 sync_record_remote.py -d 20 -o tap1                        # 정지 기록 (사람 탭 등)
"""
import argparse
import json
import os
import subprocess
import threading
import time

import numpy as np
from scipy.io import wavfile

FS = 48000
CHUNK = 480               # 10 ms (샘플)
CHUNK_BYTES = CHUNK * 2   # S16_LE 모노

SPORT_API_ID_STOPMOVE = 1003
SPORT_API_ID_MOVE = 1008
MOVE_PERIOD = 0.1               # 속도 명령 재전송 주기 (s)


class Aborted(Exception):
    pass


# ------------------------------------------------------------ 오디오 (SSH)
class RemoteAudio:
    def __init__(self, host, device, duration):
        self.host = host
        remote = ("exec arecord -q -D plughw:CARD=%s,DEV=0 -f S16_LE -r %d -c 1 "
                  "-t raw --period-time=10000 --buffer-time=100000 -d %d"
                  % (device, FS, int(np.ceil(duration)) + 5))
        self.cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "Compression=no",
                    host, remote]
        self.chunks, self.t = [], []
        self.stop_flag = threading.Event()
        self.err = ""

    def _reader(self):
        fd = self.proc.stdout.fileno()
        buf = b""
        while not self.stop_flag.is_set():
            data = os.read(fd, 65536)
            if not data:
                break
            t = time.monotonic()
            buf += data
            n = len(buf) // CHUNK_BYTES * CHUNK_BYTES
            if n:
                self.chunks.append(np.frombuffer(buf[:n], dtype="<i2").copy())
                self.t.append(t)
                buf = buf[n:]

    def start(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self.th = threading.Thread(target=self._reader, daemon=True)
        self.th.start()

    def wait_first(self, timeout):
        """첫 오디오 청크가 도착할 때까지 대기. 실패 시 False."""
        t_end = time.monotonic() + timeout
        while not self.chunks:
            if self.proc.poll() is not None or time.monotonic() > t_end:
                return False
            time.sleep(0.01)
        return True

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        self.stop_flag.set()
        self.proc.terminate()
        try:
            _, err = self.proc.communicate(timeout=3)
            self.err = err.decode(errors="ignore").strip()
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.th.join(timeout=2)
        # ssh -T 는 원격에 종료 신호가 안 갈 수 있음 -> 원격 arecord 직접 정리
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                        self.host, "pkill -x arecord"],
                       capture_output=True, timeout=10)

    def result(self):
        if not self.chunks:
            return None
        audio = np.concatenate(self.chunks)
        n_end = np.cumsum([len(c) for c in self.chunks])
        t = np.array(self.t)
        a, b = np.polyfit(n_end, t, 1)
        b_low = b + np.min(t - (a * n_end + b))   # 지연은 늦게만 생김 -> 하한선
        resid = t - (a * n_end + b_low)
        return audio, a, b_low, t, n_end, resid


# ------------------------------------------------------------ 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--duration", type=float, default=20.0, help="주행 시간 (s)")
    ap.add_argument("-o", "--out", default="sync_test")
    ap.add_argument("--host", default="unitree@192.168.123.18")
    ap.add_argument("--card", default="UMIK1", help="arecord -l 의 카드 이름")
    ap.add_argument("--low", default="/lowstate")
    ap.add_argument("--sport", default="/sportmodestate")
    ap.add_argument("--lowcmd", default="/lowcmd", help="kp/kd 를 받을 토픽 (구독만 함)")
    ap.add_argument("--vx", type=float, default=0.0, help="전신 선속도 x (m/s)")
    ap.add_argument("--vy", type=float, default=0.0, help="전신 선속도 y (m/s)")
    ap.add_argument("--vyaw", type=float, default=0.0, help="전신 각속도 yaw (rad/s)")
    ap.add_argument("--move", action="store_true", help="속도가 0 이어도 Move 전송")
    # 타임라인
    ap.add_argument("--pre", type=float, default=2.0, help="주행 전 정지 대기 (s)")
    ap.add_argument("--settle", type=float, default=2.0, help="StopMove 후 안정화 대기 (s)")
    ap.add_argument("--post", type=float, default=2.0, help="마지막 꼬리 구간 (s)")
    args = ap.parse_args()

    move_requested = args.move or bool(args.vx or args.vy or args.vyaw)
    total = args.pre + args.duration + args.settle + args.post

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ---- SSH/마이크 사전 점검 (남아 있는 arecord 정리 포함)
    chk = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", args.host,
                          "pkill -x arecord; sleep 0.3; "
                          "arecord -l | grep -q 'card.*%s' && echo MIC_OK" % args.card],
                         capture_output=True, text=True)
    if "MIC_OK" not in chk.stdout:
        msg = chk.stderr.strip() or "마이크(%s)를 찾지 못함" % args.card
        raise SystemExit("사전 점검 실패: %s\n"
                         "-> 비밀번호 없이 접속되는지 확인: ssh -o BatchMode=yes %s echo ok"
                         % (msg, args.host))
    print("SSH/마이크 점검 OK")

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    try:
        from unitree_go.msg import LowState, SportModeState, LowCmd
        from unitree_api.msg import Request as ApiRequest
    except ImportError:
        raise SystemExit("unitree_go/unitree_api 메시지 없음 -> unitree_ros2 setup.sh 를 source 하세요.")

    qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)

    class Rec(Node):
        def __init__(self):
            super().__init__("sync_recorder_remote")
            self.lock = threading.Lock()
            self.L = {k: [] for k in ["t", "tick", "ff", "acc", "gyro", "q", "dq", "tau", "quaternion", "rpy"]}
            self.S = {k: [] for k in ["t", "mode", "gait", "raise_h", "body_h",
                                      "vel", "yaw", "pos", "ff", "foot_pos", "foot_vel"]}
            self.C = {k: [] for k in ["t", "kp", "kd", "q", "dq", "tau"]}
            self.events = []            # (t, api_id, label)
            self.create_subscription(LowState, args.low, self.cb_low, qos)
            self.create_subscription(SportModeState, args.sport, self.cb_sport, qos)
            # 구독만 함 (lowcmd 에 publish 하면 sport mode 와 충돌하므로 절대 publish 금지)
            self.create_subscription(LowCmd, args.lowcmd, self.cb_cmd, qos)
            self.cmd_pub = self.create_publisher(ApiRequest, "/api/sport/request", 10)
            self._req_id = 0
            self._move_th = None
            self._move_stop = threading.Event()

        def mark(self, label, api_id=-1):
            with self.lock:
                self.events.append((time.monotonic(), api_id, label))

        def send_api(self, api_id, param=None, label=None):
            req = ApiRequest()
            with self.lock:
                req.header.identity.id = self._req_id
                self._req_id += 1
            req.header.identity.api_id = api_id
            if param is not None:
                req.parameter = json.dumps(param)
            if label:
                self.mark(label, api_id)
            self.cmd_pub.publish(req)

        @property
        def moving(self):
            return self._move_th is not None

        def start_move(self, vx, vy, vyaw):
            p = {"x": vx, "y": vy, "z": vyaw}
            self.send_api(SPORT_API_ID_MOVE, p, label="move_start")
            self._move_stop.clear()

            def loop():
                while not self._move_stop.wait(MOVE_PERIOD):
                    self.send_api(SPORT_API_ID_MOVE, p)
            self._move_th = threading.Thread(target=loop, daemon=True)
            self._move_th.start()

        def stop_move(self):
            if self._move_th is not None:
                self._move_stop.set()
                self._move_th.join(timeout=1)
                self._move_th = None
            self.send_api(SPORT_API_ID_STOPMOVE, label="move_stop")

        def cb_low(self, m):
            t = time.monotonic()
            ms = m.motor_state[:12]
            with self.lock:
                L = self.L
                L["t"].append(t); L["tick"].append(m.tick)
                L["ff"].append(list(m.foot_force))
                L["acc"].append(list(m.imu_state.accelerometer))
                L["gyro"].append(list(m.imu_state.gyroscope))
                L["quaternion"].append(list(m.imu_state.quaternion))
                L["rpy"].append(list(m.imu_state.rpy))
                L["q"].append([s.q for s in ms])
                L["dq"].append([s.dq for s in ms])
                L["tau"].append([s.tau_est for s in ms])

        def cb_cmd(self, m):
            t = time.monotonic()
            mc = m.motor_cmd[:12]
            with self.lock:
                C = self.C
                C["t"].append(t)
                C["kp"].append([c.kp for c in mc])
                C["kd"].append([c.kd for c in mc])
                C["q"].append([c.q for c in mc])
                C["dq"].append([c.dq for c in mc])
                C["tau"].append([c.tau for c in mc])

        def cb_sport(self, m):
            t = time.monotonic()
            with self.lock:
                S = self.S
                S["t"].append(t); S["mode"].append(m.mode)
                S["gait"].append(m.gait_type); S["raise_h"].append(m.foot_raise_height)
                S["body_h"].append(m.body_height)
                S["vel"].append(list(m.velocity)); S["yaw"].append(m.yaw_speed)
                S["pos"].append(list(m.position))
                S["ff"].append(list(m.foot_force))
                S["foot_pos"].append(list(m.foot_position_body))
                S["foot_vel"].append(list(m.foot_speed_body))

    rclpy.init()
    node = Rec()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_th = threading.Thread(target=executor.spin, daemon=True)
    spin_th.start()

    def ros_close():
        executor.shutdown()          # spin 스레드를 먼저 멈춰야 종료 시 segfault 안 남
        spin_th.join(timeout=2)
        node.destroy_node()
        rclpy.shutdown()

    time.sleep(1.0)
    with node.lock:
        nl, ns, nc = len(node.L["t"]), len(node.S["t"]), len(node.C["t"])
    print("1초 점검: lowstate %d 개, sportmodestate %d 개, lowcmd %d 개" % (nl, ns, nc))
    if nc == 0:
        print("!! lowcmd 수신 없음 -> kp/kd 는 저장되지 않음 (ros2 topic hz %s 로 확인)" % args.lowcmd)
    if nl == 0:
        print("!! lowstate 수신 없음 (ros2 topic hz %s 로 확인)" % args.low)

    # ---- 오디오 시작, 실제 수신 확인 후에만 로봇 동작
    audio = RemoteAudio(args.host, args.card, total)
    audio.start()
    if not audio.wait_first(3.0):
        audio.stop()
        ros_close()
        raise SystemExit("오디오 시작 실패: %s" % (audio.err or "청크 수신 없음"))
    node.mark("audio_ok")
    print("오디오 수신 확인, 전체 기록 약 %.1f 초" % total)

    def wait(sec):
        t_end = time.monotonic() + sec
        while time.monotonic() < t_end:
            if not audio.alive():
                raise Aborted("원격 arecord 가 종료됨")
            time.sleep(0.02)

    try:
        print("[1] 정지 대기 %.1f s" % args.pre)
        wait(args.pre)

        if move_requested:
            print("[2] 주행: vx=%.2f vy=%.2f vyaw=%.2f, %.1f s"
                  % (args.vx, args.vy, args.vyaw, args.duration))
            node.start_move(args.vx, args.vy, args.vyaw)
            wait(args.duration)
            node.stop_move()
            print("[3] StopMove, %.1f s 안정화" % args.settle)
        else:
            print("[2] 속도 명령 없음, %.1f s 정지 기록" % args.duration)
            wait(args.duration)
        wait(args.settle)

        wait(args.post)
        node.mark("end")
    except KeyboardInterrupt:
        print("\n중단, 저장합니다.")
        node.mark("interrupted")
    except Aborted as e:
        print("!! %s" % e)
        node.mark("aborted")
    finally:
        if node.moving:
            node.stop_move()

    audio.stop()
    with node.lock:
        L = {k: np.array(v) for k, v in node.L.items()}
        S = {k: np.array(v) for k, v in node.S.items()}
        C = {k: np.array(v) for k, v in node.C.items()}
        ev = list(node.events)
    ros_close()

    res = audio.result()
    if res is None:
        raise SystemExit("오디오 수신 실패. ssh 에러: %s" % audio.err)
    a_data, slope, t0, cb_t, n_end, resid = res

    meta = dict(vx=args.vx, vy=args.vy, vyaw=args.vyaw, duration=args.duration,
                pre=args.pre, settle=args.settle, post=args.post,
                move=move_requested)

    np.savez_compressed(
        args.out + ".npz",
        audio=a_data, fs=FS, audio_slope=slope, audio_t0=t0,
        audio_cb_t=cb_t, audio_n_end=n_end, audio_status=np.array([]),
        ev_t=np.array([e[0] for e in ev]),
        ev_api=np.array([e[1] for e in ev]),
        ev_label=np.array([e[2] for e in ev]),
        meta=json.dumps(meta),
        **{"ls_" + k: v for k, v in L.items()},
        **{"sp_" + k: v for k, v in S.items()},
        **{"lc_" + k: v for k, v in C.items()})
    wavfile.write(args.out + ".wav", FS, a_data)

    print("=" * 56)
    fs_est = 1 / slope
    print("오디오   : %.2f s, 추정 fs %.2f Hz, 도착 지터 중앙 %.1f / 최대 %.1f ms"
          % (len(a_data) / FS, fs_est, np.median(resid) * 1e3, resid.max() * 1e3))
    if abs(fs_est - FS) > 10:
        print("!! 추정 fs 가 %d 에서 크게 벗어남 -> 샘플 누락(xrun) 의심, 이 trial 확인 필요" % FS)
    for nm, D in (("lowstate", L), ("sport   ", S), ("lowcmd  ", C)):
        if len(D["t"]) > 1:
            dt = np.diff(D["t"])
            print("%s: %d 개, %.1f Hz, dt 중앙 %.2f / 최대 %.2f ms"
                  % (nm, len(D["t"]), 1 / dt.mean(), np.median(dt) * 1e3, dt.max() * 1e3))
        else:
            print("%s: 수신 없음" % nm)
    if len(C["t"]):
        print("kp 범위: %.1f ~ %.1f, kd 범위: %.2f ~ %.2f"
              % (C["kp"].min(), C["kp"].max(), C["kd"].min(), C["kd"].max()))
    if ev:
        t_ref = ev[0][0]
        print("이벤트   : " + ", ".join("%s@%.2fs" % (e[2], e[0] - t_ref) for e in ev))
    if audio.err:
        print("ssh stderr: %s" % audio.err[:200])
    print("저장     : %s.npz, %s.wav" % (args.out, args.out))
    print("다음     : python3 sync_check.py %s.npz" % args.out)


if __name__ == "__main__":
    main()