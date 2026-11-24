#!/usr/bin/env python3
"""Record MuJoCo simulation states with the real recorder's NPZ keys; q or Ctrl+C saves.

The simulation (simulation.py --state-port 9881) streams one packet per physics step (200 Hz)
over local UDP. This program only records: no noise model and no dashboard. The output uses the
keys, shapes and leg order (FR, FL, RR, RL) of sync_record_rl.py, so real and simulated
recordings are read by the same code:
  <dir>/<name>_<YYYYmmdd_HHMMSS>_sim.npz

ls_* / lc_* / sp_* mirror the real LowState / LowCmd / SportModeState fields; sim_* are
simulation-only extras. *_t is time.monotonic() at receipt, the same host clock as the real ls_t.
Data is kept in memory and written once on exit (q, Ctrl+C or SIGTERM).
  python src/db_collector/record_sim.py --name NORMAL_01 --dir src/sim_to_real_data/real_sim
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import sys
import termios
import time
import tty

import numpy as np

from sync_record_rl import clock_fit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / 'sim_data'
RAW_FIELDS = ('acc', 'tau', 'ddq', 'q_target', 'base_pos_w', 'base_lin_vel_w', 'yaw_rate_w',
              'foot_pos_b', 'foot_vel_b', 'foot_force_vec', 'command', 'action')
SIZES = (35, 3, 12, 12, 12, 3, 3, 1, 12, 12, 12, 3, 12)  # "low", then RAW_FIELDS


def quat_to_rpy(quat):
    """(w, x, y, z) -> [roll, pitch, yaw], intrinsic Z-Y-X (no scipy, so the host recorder can use it)."""
    w, x, y, z = np.asarray(quat, dtype='<f8').reshape(-1, 4).T
    return np.stack([np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
                     np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))], axis=1)


class SimRecorder:
    def __init__(self, path):
        self.path = path
        self.rows, self.kp, self.kd = [], None, None
        self.started = time.monotonic()
        self.events = None  # record_session: the take's shared (time, label) events

    def add(self, received, packet):
        raw = packet['raw']
        self.kp, self.kd = raw['kp'], raw['kd']
        self.rows.append((received, packet['tick'], packet['sim_time'], packet['low'],
                          *[raw[name] for name in RAW_FIELDS]))

    def save(self):
        n = len(self.rows)
        cols = list(zip(*self.rows)) if n else [[]] * (3 + len(SIZES))
        t, tick, sim_time = (np.asarray(cols[i], dtype=dtype) for i, dtype in ((0, '<f8'), (1, '<i8'), (2, '<f8')))
        # The simulation computes one policy step (4 physics steps) in a ~4 ms burst, then sleeps,
        # so receive times t are bunched (-12..+7 ms off sim_time). The first step of each burst
        # arrives right after the policy read its command; map sim_time onto those arrivals.
        burst_start = np.r_[True, np.diff(t) > 0.5 * np.median(np.diff(sim_time))] if n > 1 else np.ones(n, bool)
        t_sync, residual = clock_fit(sim_time, t, burst_start) if n else (t, float('nan'))
        events = self.events or [(t[0] if n else self.started, 'move_start'), (t[-1] if n else self.started, 'end')]
        low, acc, tau, ddq, q_target, pos, vel, yaw_rate, foot_pos, foot_vel, force_vec, command, action = (
            np.asarray(cols[i], dtype='<f8').reshape(n, size) for i, size in zip(range(3, 3 + len(SIZES)), SIZES))
        quat = low[:, 4:8]
        rpy = quat_to_rpy(quat)
        f32 = lambda a: np.ascontiguousarray(a, dtype='<f4')  # noqa: E731
        zeros12 = np.zeros((n, 12))
        arrays = dict(
            # LowState (real: /lowstate)
            ls_t=t, ls_t_sync=t_sync, ls_tick=tick, ls_ff=f32(low[:, 0:4]), ls_acc=f32(acc), ls_gyro=f32(low[:, 8:11]),
            ls_rpy=f32(rpy), ls_quaternion=f32(quat), ls_q=low[:, 11:23], ls_dq=low[:, 23:35], ls_tau=tau,
            ls_ddq=ddq,
            # SportModeState (empty on the real robot under RL control; simulation ground truth here)
            sp_t=t, sp_mode=np.full(n, -1, '<i8'), sp_gait=np.full(n, -1, '<i8'),
            sp_raise_h=np.full(n, np.nan), sp_body_h=pos[:, 2].copy(), sp_vel=f32(vel), sp_yaw=yaw_rate[:, 0].copy(),
            sp_pos=f32(pos), sp_ff=f32(low[:, 0:4]), sp_foot_pos=f32(foot_pos), sp_foot_vel=f32(foot_vel),
            # LowCmd (real: /lowcmd): PD position targets with constant gains, no feed-forward.
            lc_t=t, lc_kp=np.tile(np.asarray(self.kp or [np.nan] * 12, '<f8'), (n, 1)),
            lc_kd=np.tile(np.asarray(self.kd or [np.nan] * 12, '<f8'), (n, 1)),
            lc_q=q_target, lc_dq=zeros12, lc_tau=zeros12.copy(),
            # Simulation-only extras
            sim_time=sim_time, sim_command=f32(command), sim_action=f32(action),
            sim_foot_force_vec=f32(force_vec.reshape(n, 4, 3)),
            # Recording events, as in the real file
            ev_t=np.array([e[0] for e in events], dtype='<f8'),
            ev_api=np.full(len(events), -1, '<i8'), ev_label=np.array([e[1] for e in events]),
            meta=json.dumps(dict(
                source='mujoco', counts=dict(ls=n, sp=n, lc=n), rate_hz='physics step (200)',
                timestamp_clock='PC time.monotonic receive time (same as real ls_t)',
                ls_t_sync='sim_time fitted to burst-start receive times (PC monotonic); use for alignment',
                ls_t_sync_max_residual_ms=residual,
                notes=dict(ls_ff='|contact force| [N] + real airborne reading (8,10,12,9); float, real is int16',
                           ls_acc='IMU accelerometer incl. gravity', ls_gyro='base angular velocity, body frame',
                           ls_ddq='joint acceleration [rad/s^2]', ls_tau='actuator force = joint torque [N m]',
                           sp_vel='base linear velocity, world', sp_yaw='yaw rate, world', sp_pos='base position, world',
                           sp_foot_pos='foot - base, body frame', sp_foot_vel='foot - base velocity, body frame',
                           sp_mode='-1: not applicable', lc_q='policy joint position target',
                           sim_command='policy velocity command [vx, vy, wz]', sim_action='raw policy action'))))
        tmp = Path(str(self.path) + '.tmp')
        with tmp.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        tmp.rename(self.path)
        return n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', '--name', help='저장 이름. 생략하면 실행 시 입력받는다.')
    parser.add_argument('--dir', type=Path, default=DATA_DIR,
                        help='저장 폴더, 프로젝트 루트 기준 (기본 src/db_collector/sim_data)')
    parser.add_argument('--port', type=int, default=9881, help='simulation.py --state-port')
    parser.add_argument('--duration', type=float, help='녹화 시간 [초]. 지나면 자동 종료·저장 (생략하면 q까지 무제한)')
    args = parser.parse_args()
    interactive = sys.stdin.isatty()
    name = args.name
    while not name:
        if not interactive:
            parser.error('--name is required without a terminal')
        try:
            name = input('저장 이름: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
    name = re.sub(r'[^\w.-]+', '_', name).strip('._') or 'sim'
    directory = args.dir if args.dir.is_absolute() else PROJECT_ROOT / args.dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_sim.npz"
    if path.exists():
        parser.error(f'기존 파일이 있습니다: {path}')

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', args.port))
    receiver.settimeout(.05)
    record = SimRecorder(path)
    def interrupted(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)  # docker stop still saves
    terminal = termios.tcgetattr(sys.stdin.fileno()) if interactive else None
    print(f'시뮬 기록 시작: UDP {args.port} → {path}\nq 또는 Ctrl+C → 종료·저장'
          + (f' | {args.duration:g}초 후 자동 종료' if args.duration else ''), flush=True)
    session, last_tick, gaps, next_status = None, None, 0, time.monotonic() + 1
    try:
        if interactive:
            tty.setcbreak(sys.stdin.fileno())
        while True:
            if interactive and select.select([sys.stdin], [], [], 0)[0]:
                if os.read(sys.stdin.fileno(), 1) in (b'q', b'Q'):
                    break
            try:
                payload, _ = receiver.recvfrom(65536)
            except socket.timeout:
                payload = None
            now = time.monotonic()
            if args.duration and now - record.started >= args.duration:
                print(f'{args.duration:g}초 경과 → 자동 종료', flush=True)
                break
            if payload:
                try:
                    packet = json.loads(payload)
                    record.add(now, packet)
                    if packet['session'] == session and packet['tick'] != last_tick + 1:
                        gaps += 1  # dropped UDP packet(s)
                    session, last_tick = packet['session'], packet['tick']
                except (ValueError, KeyError, TypeError) as error:
                    print(f'!! 잘못된 패킷: {error}', flush=True)
            if now >= next_status:
                print(f'{now - record.started:.0f}초 | 물리 스텝 {len(record.rows)} | 끊김 {gaps}'
                      + ('' if record.rows else ' | !! 수신 없음: 시뮬을 --state-port로 실행했는지 확인'), flush=True)
                next_status = now + 5
    except KeyboardInterrupt:
        pass
    finally:
        if terminal is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal)
        receiver.close()
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # do not interrupt the save
        print(f'저장 중... 물리 스텝 {len(record.rows)}', flush=True)
        record.save()
        print(f'저장: {path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
