#!/usr/bin/env python3
"""DiffImpact 학습용 착지(touchdown) 기준 데이터 재추출.

입력 : src/sim_to_real_data/{real_sim, real_sim_straight}/*.npz  (실로봇 .npz + 시뮬 _sim.npz)
출력 : src/diffimpact/data/touchdown_w100_200hz/  (기본: 창 -20~+80 ms, 실로봇 200 Hz)
  touchdowns_real.csv / touchdowns_sim.csv   다리별 착지 1행
  impacts_real.npz / impacts_sim.npz         착지 묶음(대각 쌍)별 창: 소리(실로봇) + 발 힘 + 발 속도
  summary.json                               파일별 검출 수·기준값·소리 지연 추정 등
  figures/<녹화>.png                         검출 확인 그림 (--plots)

착지 검출 (발 센서 ls_ff, 다리별)
  air  = 공중 기준값. 실로봇: 파일·다리별 5 백분위수, 시뮬: 공중 고정값(8,10,12,9)
  on   = air + ON_DELTA  (이 값을 넘으면 접지 후보)
  off  = air + OFF_DELTA (이 값 아래로 내려가면 공중)
  조건 : 착지 전 MIN_SWING 이상 off 아래(유각기) + 착지 후 MIN_STANCE 이상 on 위 유지
  시각 : on 교차 지점에서 거슬러 올라가 힘이 air + RISE_DELTA 이하인 마지막 샘플의 다음 샘플
         (= 힘이 오르기 시작한 첫 샘플). 시각 기준은 PC monotonic (실/시뮬 공통).

착지 묶음 (impact group)
  트롯이라 대각 두 다리가 0~60 ms 차이로 착지(횡방향일수록 큼) → GROUP_WINDOW(80 ms) 안의 착지를 묶고,
  묶음의 첫 착지 시각을 창의 기준(0)으로 둔다. 창 = [-PRE, +POST] (기본 -50 ms ~ +250 ms).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC / 'db_prediction' / 'preprocessing'))
from go2_kinematics import Go2Kinematics  # noqa: E402

LEGS = ('FR', 'FL', 'RR', 'RL')            # Unitree 순서 (ls_ff, ls_q 열 순서와 동일)
SIM_AIR = np.array([8., 10., 12., 9.])     # record_sim.py 의 공중값
DATASETS = {'real_sim': 'lateral', 'real_sim_straight': 'straight'}
FS_GRID = 1000.                            # 힘·속도 창의 공통 격자 [Hz]

ON_DELTA = 10.
OFF_DELTA = 4.
RISE_DELTA = 2.
MIN_SWING = 0.06
MIN_STANCE = 0.04
GROUP_WINDOW = 0.08
PEAK_WINDOW = 0.03
VEL_WINDOW = 0.01
GRF_WINDOW = 0.02          # 접촉력 특징: 착지 후 20 ms
GRF_RATE_LAG = 0.01        # 상승률: 10 ms 차분


# ---------------------------------------------------------------- 입력
def trial_key(path):
    """'NORMAL_0.7_0.5_v2_20261120_173316_sim.npz' -> 'NORMAL_0.7_0.5_v2'"""
    return re.sub(r'_\d{8}_\d{6}(_sim)?$', '', path.stem)


def find_pairs(data_root):
    pairs = []
    for folder, walk in DATASETS.items():
        files = sorted((data_root / folder).glob('*.npz'))
        real = {trial_key(f): f for f in files if not f.stem.endswith('_sim')}
        sim = {trial_key(f): f for f in files if f.stem.endswith('_sim')}
        for key in sorted(set(real) | set(sim)):
            pairs.append(dict(dataset=folder, walk=walk, trial=key, real=real.get(key), sim=sim.get(key)))
    return pairs


def downsample_hold(d, rate):
    """실로봇 lowstate 를 시뮬처럼 rate [Hz] 틱마다 그 시점의 최신 샘플로 줄임 (sample-and-hold).
    MuJoCo 기록(물리 스텝 5 ms = 200 Hz)과 같은 시각 해상도·검출 지연으로 학습하기 위함. 소리는 그대로."""
    t = d['t']
    grid = np.arange(t[0], t[-1], 1. / rate)
    idx = np.searchsorted(t, grid, side='right') - 1
    o = dict(d)
    for k, v in d.items():
        if k.startswith('ls_') and isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(t):
            o[k] = v[idx]
    o['t'] = grid
    o['ls_t'] = grid
    return o


def load(path, source):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if len(d['ls_t']) == 0:
        return None
    if source == 'sim':
        # sim ls_t 는 4 스텝씩 몰려 도착 → 정확한 sim_time 격자를 PC 시계로 옮겨 쓴다
        offset = np.median(d['ls_t'] - d['sim_time'])
        d['t'] = d['sim_time'] + offset
    else:
        d['t'] = d['ls_t'].astype(float)
    return d


def command_of(sim):
    if sim is None or 'sim_command' not in sim:
        return [np.nan] * 3
    c = sim['sim_command']
    moving = np.abs(c).sum(1) > 1e-6
    return [float(x) for x in (np.median(c[moving], 0) if moving.any() else np.zeros(3))]


# ---------------------------------------------------------------- 기구학
def foot_kinematics(d, model):
    """몸체 대비 발 위치(몸체 좌표)·속도(중력 정렬 좌표)와 관절 토크 기반 접촉력(중력 정렬 좌표).

    접촉력 = 지면이 발에 주는 힘 F = -J(q)^-T τ  (정적 평형, 다리 관성·중력·마찰 무시).
    실로봇 ls_tau(모터 tau_est)와 시뮬 ls_tau(액추에이터 힘)에 같은 식을 써서 단위[N]가 같다.
    시뮬 정답(sim_foot_force_vec, 부호 반대)과 입각기 수직력 차이 ~2 N, 시계열 상관 0.86~0.92.
    실로봇은 몸체 선속도가 없어 발 속도는 양쪽 다 몸체 대비 상대값으로 통일.
    """
    rot = Rotation.from_quat(d['ls_quaternion'][:, [1, 2, 3, 0]].astype(float))
    gyro = d['ls_gyro'].astype(float)
    pos, vel, grf = [], [], []
    for i in range(4):
        q = d['ls_q'][:, 3 * i:3 * i + 3].astype(float)
        dq = d['ls_dq'][:, 3 * i:3 * i + 3].astype(float)
        tau = d['ls_tau'][:, 3 * i:3 * i + 3].astype(float)
        p, jac = model.position_jacobian(LEGS[i].lower(), q)
        v = np.einsum('nij,nj->ni', jac, dq) + np.cross(gyro, p)
        f = -np.linalg.solve(np.transpose(jac, (0, 2, 1)), tau[..., None])[..., 0]
        pos.append(p)
        vel.append(rot.apply(v))
        grf.append(rot.apply(f))
    return np.stack(pos, 1), np.stack(vel, 1), np.stack(grf, 1)   # (N,4,3) 각각


def force_features(t, fz, t0, pre_mask, prefix):
    """착지 후 20 ms 접촉력 특징. 실로봇 500 Hz / 시뮬 200 Hz 를 1 ms 격자로 보간해 같은 방식으로 계산."""
    g = np.arange(0, GRF_WINDOW + 1e-9, 1e-3)
    f = np.interp(t0 + g, t, fz)
    base = fz[pre_mask].mean()
    lag = int(round(GRF_RATE_LAG / 1e-3))
    return {f'{prefix}_pre': base, f'{prefix}_at_td': f[0], f'{prefix}_peak20': f.max(),
            f'{prefix}_rise20': f.max() - base,
            f'{prefix}_impulse20': float(np.sum(f - base) * 1e-3),
            f'{prefix}_rate10': float(np.max((f[lag:] - f[:-lag]) / GRF_RATE_LAG))}


# ---------------------------------------------------------------- 착지 검출
def detect_leg(t, f, air):
    on, off = air + ON_DELTA, air + OFF_DELTA
    above = f > on
    events = []
    crossings = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    last_end = -np.inf
    for c in crossings:
        if t[c] <= last_end:
            continue
        # 유각기: 착지 직전 MIN_SWING 동안 off 아래였던 구간이 있어야 함
        k = c - 1
        while k >= 0 and f[k] <= on:
            k -= 1
        swing_idx = np.arange(k + 1, c)
        low = swing_idx[f[swing_idx] < off] if len(swing_idx) else swing_idx
        if len(low) == 0 or t[low[-1]] - t[low[0]] < MIN_SWING:
            continue
        # 시작 시각: 힘이 오르기 시작한 첫 샘플
        j = c
        while j > low[-1] and f[j - 1] > air + RISE_DELTA:
            j -= 1
        # 입각기: MIN_STANCE 동안 off 위 유지
        e = c
        while e < len(f) and f[e] >= off:
            e += 1
        stance = t[min(e, len(t) - 1)] - t[j]
        if stance < MIN_STANCE:
            continue
        swing_start = t[low[0]]
        events.append(dict(idx=j, cross_idx=c, end_idx=min(e, len(t) - 1),
                           swing_s=t[j] - swing_start, stance_s=stance,
                           censored_swing=bool(k < 0), censored_stance=bool(e >= len(f))))
        last_end = t[min(e, len(t) - 1)]
    return events


def detect(d, source, model):
    t = d['t']
    ff = d['ls_ff'].astype(float)
    air = SIM_AIR.copy() if source == 'sim' else np.percentile(ff, 5, axis=0)
    pos, vel, grf = foot_kinematics(d, model)
    rows = []
    for i, leg in enumerate(LEGS):
        f = ff[:, i]
        for ev in detect_leg(t, f, air[i]):
            j = ev['idx']
            t0 = t[j]
            peak = (t >= t0) & (t <= t0 + PEAK_WINDOW)
            pre = (t >= t0 - VEL_WINDOW) & (t < t0)
            if not pre.any():
                pre = np.zeros_like(pre); pre[max(j - 1, 0)] = True
            fp = f[peak]
            dt = np.diff(t[peak])
            rate = np.max(np.diff(fp) / dt) if len(fp) > 1 else np.nan
            row = dict(leg=leg, t_abs=t0, t_cross_abs=t[ev['cross_idx']],
                       force_air=air[i], force_at_td=f[j], force_peak30=fp.max(),
                       force_peak30_minus_air=fp.max() - air[i],
                       time_to_peak_ms=(t[peak][np.argmax(fp)] - t0) * 1e3,
                       force_rate_max=rate,
                       foot_vx_pre=vel[pre, i, 0].mean(), foot_vy_pre=vel[pre, i, 1].mean(),
                       foot_vz_pre=vel[pre, i, 2].mean(),
                       foot_x=pos[j, i, 0], foot_y=pos[j, i, 1], foot_z=pos[j, i, 2],
                       swing_s=ev['swing_s'], stance_s=ev['stance_s'],
                       censored=ev['censored_swing'] or ev['censored_stance'])
            row.update(force_features(t, grf[:, i, 2], t0, pre, 'grf_fz'))
            if source == 'sim':
                row.update(force_features(t, -d['sim_foot_force_vec'][:, i, 2], t0, pre, 'sim_gt_fz'))
                fv = d['sim_foot_force_vec'][:, i, :]
                k = np.flatnonzero(peak)[np.argmax(np.linalg.norm(fv[peak], axis=1))]
                row.update(sim_fx_peak=fv[k, 0], sim_fy_peak=fv[k, 1], sim_fz_peak=fv[k, 2],
                           sim_foot_vz_gt_pre=d['sp_foot_vel'][pre, 3 * i + 2].mean(),
                           sim_body_vz_pre=d['sp_vel'][pre, 2].mean())
            rows.append(row)
    if not rows:
        return None, air, pos, vel, grf
    table = pd.DataFrame(rows).sort_values('t_abs').reset_index(drop=True)
    return table, air, pos, vel, grf


def group(table):
    gid, start, out = -1, -np.inf, []
    for t in table.t_abs:
        if t - start > GROUP_WINDOW:
            gid += 1; start = t
        out.append(gid)
    table['group'] = out
    g = table.groupby('group')
    table['group_t0'] = g.t_abs.transform('min')
    table['offset_in_group_ms'] = (table.t_abs - table.group_t0) * 1e3
    table['group_size'] = g.leg.transform('size')
    table['group_legs'] = g.leg.transform(lambda s: '+'.join(s))
    starts = g.t_abs.min()
    prev_gap = starts.diff(); next_gap = -starts.diff(-1)
    table['prev_group_gap_ms'] = table.group.map(prev_gap) * 1e3
    table['next_group_gap_ms'] = table.group.map(next_gap) * 1e3
    return table


# ---------------------------------------------------------------- 소리
def audio_axis(d):
    return float(d['audio_t0']), float(d['audio_slope'])


def clip_envelope_onset(clips, fs, pre, rise_db=6.0):
    """착지 기준 소리 클립들의 중앙값 1 ms RMS 포락선과, 착지 전 기준보다 rise_db 오른 첫 시각 [s].

    진단용 (클립에는 --audio-lag 로만 적용). 음수 = 힘 센서 착지보다 소리가 먼저 시작.
    """
    a = np.asarray(clips, float); a -= a.mean(1, keepdims=True)
    w = max(int(0.001 * fs), 1)
    k = np.ones(w) / w
    env = np.median(np.sqrt(np.array([np.convolve(x * x, k, 'same') for x in a])), 0)
    t = np.arange(a.shape[1]) / fs - pre
    db = 20 * np.log10(env / np.median(env[t < -0.02]) + 1e-12)
    cand = np.flatnonzero((t > -0.02) & (db > rise_db))
    return (float(t[cand[0]]) if len(cand) else np.nan), t, db


def audio_clip(d, t_start, n_samples):
    t0, slope = audio_axis(d)
    s = int(round((t_start - t0) / slope))
    if s < 0 or s + n_samples > len(d['audio']):
        return None
    return d['audio'][s:s + n_samples]


def dbfs(x):
    x = x.astype(float); x = x - x.mean()
    return 20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-12) / 32768.)


# ---------------------------------------------------------------- 창 만들기
def resample(t, y, grid):
    return np.stack([np.interp(grid, t, y[:, k]) for k in range(y.shape[1])], 1)


def build_windows(d, table, source, pos, vel, grf, pre, post, audio_lag):
    t = d['t']
    grid_rel = np.arange(-pre, post, 1 / FS_GRID)
    ff = d['ls_ff'].astype(float)
    groups, kept = [], []
    for gid, g in table.groupby('group'):
        g0 = g.t_abs.min()
        grid = g0 + grid_rel
        if grid[0] < t[0] or grid[-1] > t[-1]:
            continue
        item = dict(group=gid, t0_abs=g0,
                    force=resample(t, ff, grid).astype(np.float32),
                    foot_vel=resample(t, vel.reshape(len(t), 12), grid).reshape(-1, 4, 3).astype(np.float32),
                    foot_pos=resample(t, pos.reshape(len(t), 12), grid).reshape(-1, 4, 3).astype(np.float32),
                    grf=resample(t, grf.reshape(len(t), 12), grid).reshape(-1, 4, 3).astype(np.float32),
                    td_offset_ms=np.full(4, np.nan, np.float32), td_mask=np.zeros(4, bool))
        for r in g.itertuples():
            i = LEGS.index(r.leg)
            item['td_offset_ms'][i] = r.offset_in_group_ms
            item['td_mask'][i] = True
        if source == 'sim':
            fv = d['sim_foot_force_vec'].reshape(len(t), 12)
            item['force_vec'] = resample(t, fv, grid).reshape(-1, 4, 3).astype(np.float32)
        else:
            fs = int(d['fs'])
            n = int(round((pre + post) * fs))
            clip = audio_clip(d, g0 - pre + audio_lag, n)
            if clip is None:
                continue
            item['audio'] = clip.astype(np.int16)
            pre_clip = audio_clip(d, g0 - pre + audio_lag, int(round((pre - 0.01) * fs)))
            post_clip = audio_clip(d, g0 + audio_lag, int(round(0.04 * fs)))
            item['audio_pre_dbfs'] = dbfs(pre_clip)
            item['audio_post40_dbfs'] = dbfs(post_clip)
        groups.append(item); kept.append(gid)
    return groups, kept


def stack_groups(all_groups):
    keys = [k for k in all_groups[0] if isinstance(all_groups[0][k], np.ndarray)]
    out = {k: np.stack([g[k] for g in all_groups]) for k in keys}
    for k in all_groups[0]:
        if k not in keys and k not in ('meta',):
            out[k] = np.array([g[k] for g in all_groups])
    return out


# ---------------------------------------------------------------- 그림
def plot_trial(path, name, real, rtab, sim, stab, lag, lags, xc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(LEGS, ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')))
    rows = 3 if real is not None else 1
    fig, axes = plt.subplots(rows + 1, 1, figsize=(15, 3 * (rows + 1)))
    ax = list(np.atleast_1d(axes))
    ref = real if real is not None else sim
    tab = rtab if real is not None else stab
    mid = tab.t_abs.median() if len(tab) else ref['t'][len(ref['t']) // 2]
    lo, hi = mid - 1.0, mid + 1.0
    k = 0
    for d, tb, label in ((real, rtab, 'real'), (sim, stab, 'sim')):
        if d is None or k >= rows:
            continue
        a = ax[k]; k += 1
        for i, leg in enumerate(LEGS):
            a.plot(d['t'] - lo, d['ls_ff'][:, i], color=colors[leg], lw=.8, label=leg)
        for r in tb.itertuples():
            if lo <= r.t_abs <= hi:
                a.axvline(r.t_abs - lo, color=colors[r.leg], ls='--', lw=.8)
        a.set_xlim(0, hi - lo); a.set_ylabel(f'{label} ls_ff'); a.legend(ncol=4, fontsize=8, loc='upper right')
        if real is not None and label == 'real':
            b = ax[k]; k += 1
            t0, slope = audio_axis(real)
            ta = t0 + slope * np.arange(len(real['audio']))
            m = (ta >= lo) & (ta <= hi)
            b.plot(ta[m] - lo, real['audio'][m], lw=.3, color='k')
            for r in tb.itertuples():
                if lo <= r.t_abs <= hi:
                    b.axvline(r.t_abs - lo, color=colors[r.leg], ls='--', lw=.8)
            b.set_xlim(0, hi - lo); b.set_ylabel('audio (raw)')
    a = ax[-1]
    if xc is not None:
        a.plot(lags * 1e3, xc); a.axvline(lag * 1e3, color='r'); a.axvline(0, color='k', ls=':')
        a.set_xlabel('ms from first touchdown of group'); a.set_ylabel('median 1ms RMS [dB re pre]')
        a.set_title(f'touchdown-aligned median audio envelope; onset = {lag * 1e3:.1f} ms (negative: sound before force onset)')
    else:
        a.axis('off')
    for x in ax[:-1]:
        x.set_xlabel(f'time [s] from {lo:.2f}'); x.grid(alpha=.3)
    fig.suptitle(name); fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=SRC / 'sim_to_real_data')
    p.add_argument('--out', type=Path, default=HERE / 'data' / 'touchdown_w100_200hz')
    p.add_argument('--pre', type=float, default=0.02, help='착지 전 창 길이 [s]')
    p.add_argument('--post', type=float, default=0.08, help='착지 후 창 길이 [s]')
    p.add_argument('--audio-lag', type=float, default=0.0,
                   help='소리 클립을 이만큼 늦게 자름 [s] (기본 0 = 기록된 시계 그대로)')
    p.add_argument('--plots', action='store_true')
    p.add_argument('--real-rate', type=float, default=200,
                   help='실로봇 lowstate 를 이 주기[Hz]로 줄여 사용 (예: 200 = MuJoCo 물리 스텝). 0 = 원본 약 500 Hz')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plots:
        (a.out / 'figures').mkdir(exist_ok=True)
    model = Go2Kinematics()

    tables = {'real': [], 'sim': []}
    windows = {'real': [], 'sim': []}
    summary = dict(settings=dict(on_delta=ON_DELTA, off_delta=OFF_DELTA, rise_delta=RISE_DELTA,
                                 min_swing_s=MIN_SWING, min_stance_s=MIN_STANCE,
                                 group_window_s=GROUP_WINDOW, peak_window_s=PEAK_WINDOW,
                                 vel_window_s=VEL_WINDOW, pre_s=a.pre, post_s=a.post,
                                 audio_lag_applied_s=a.audio_lag, grid_hz=FS_GRID, legs=LEGS,
                                 real_rate_hz=a.real_rate or 'native (~500)',
                                 sim_air=SIM_AIR.tolist()),
                   trials=[])
    for pair in find_pairs(a.data_root):
        real = load(pair['real'], 'real') if pair['real'] else None
        if real is not None and a.real_rate:
            real = downsample_hold(real, a.real_rate)
        sim = load(pair['sim'], 'sim') if pair['sim'] else None
        cmd = command_of(sim)
        info = dict(dataset=pair['dataset'], walk=pair['walk'], trial=pair['trial'],
                    real_file=pair['real'].name if pair['real'] else None,
                    sim_file=pair['sim'].name if pair['sim'] else None,
                    cmd_vx=cmd[0], cmd_vy=cmd[1], cmd_wz=cmd[2])
        res = {}
        lag, lags, xc = np.nan, None, None
        for source, d in (('real', real), ('sim', sim)):
            if d is None:
                info[f'{source}_status'] = 'missing_or_empty'
                continue
            tab, air, pos, vel, grf = detect(d, source, model)
            if tab is None:
                info[f'{source}_status'] = 'no_touchdowns'
                info[f'{source}_air'] = [float(x) for x in air]
                continue
            tab = group(tab)
            if sim is not None:
                # 착지 순간 정책에 들어간 속도 명령 (sim_command, PC 시계로 보간; 실로봇도 같은 명령)
                for k, name in enumerate(('vx', 'vy', 'wz')):
                    tab[f'cmd_{name}_at_td'] = np.interp(tab.t_abs, sim['ls_t'], sim['sim_command'][:, k],
                                                         left=np.nan, right=np.nan)
            groups, kept = build_windows(d, tab, source, pos, vel, grf, a.pre, a.post, a.audio_lag)
            if source == 'real' and groups:
                lag, lags, xc = clip_envelope_onset([g['audio'] for g in groups], int(d['fs']), a.pre)
            tab['in_impacts_npz'] = tab.group.isin(kept)
            tab.insert(0, 'trial', pair['trial'])
            tab.insert(0, 'walk', pair['walk'])
            tab.insert(0, 'dataset', pair['dataset'])
            tab['cmd_vx'], tab['cmd_vy'], tab['cmd_wz'] = cmd
            tab['t_rel'] = tab.t_abs - d['t'][0]
            cmd_cols = ['cmd_vx_at_td', 'cmd_vy_at_td', 'cmd_wz_at_td']
            for g in groups:
                at_td = (tab.loc[tab.group == g['group'], cmd_cols].mean().to_numpy(np.float32)
                         if cmd_cols[0] in tab else np.full(3, np.nan, np.float32))
                g.update(trial=pair['trial'], dataset=pair['dataset'], walk=pair['walk'],
                         cmd=np.array(cmd, np.float32), cmd_at_td=at_td, t0_rel=g['t0_abs'] - d['t'][0])
            tables[source].append(tab); windows[source].extend(groups); res[source] = (tab, air)
            dur = d['t'][-1] - d['t'][0]
            info.update({f'{source}_status': 'ok', f'{source}_duration_s': float(dur),
                         f'{source}_air': [float(x) for x in air],
                         f'{source}_touchdowns': {leg: int((tab.leg == leg).sum()) for leg in LEGS},
                         f'{source}_groups': int(tab.group.nunique()) if len(tab) else 0,
                         f'{source}_groups_in_npz': len(groups),
                         f'{source}_pair_fraction': float((tab.groupby('group').leg.size() == 2).mean()) if len(tab) else np.nan})
        info['audio_onset_vs_touchdown_s'] = lag
        summary['trials'].append(info)
        print(f"[{pair['dataset']}] {pair['trial']:22s} cmd={np.round(cmd, 2)} "
              f"real={info.get('real_touchdowns')} sim={info.get('sim_touchdowns')} "
              f"groups real/sim={info.get('real_groups_in_npz')}/{info.get('sim_groups_in_npz')} "
              f"audio_onset={lag * 1e3 if np.isfinite(lag) else float('nan'):.1f}ms", flush=True)
        if a.plots and res:
            plot_trial(a.out / 'figures' / f"{pair['dataset']}__{pair['trial']}.png", pair['trial'],
                       real if 'real' in res else None, res.get('real', (pd.DataFrame(columns=['t_abs', 'leg']),))[0],
                       sim if 'sim' in res else None, res.get('sim', (pd.DataFrame(columns=['t_abs', 'leg']),))[0], lag, lags, xc)

    for source in ('real', 'sim'):
        tab = pd.concat(tables[source], ignore_index=True)
        tab.to_csv(a.out / f'touchdowns_{source}.csv', index=False, float_format='%.10g')
        arrays = stack_groups(windows[source])
        arrays['t_rel_ms'] = (np.arange(-a.pre, a.post, 1 / FS_GRID) * 1e3).astype(np.float32)
        arrays['legs'] = np.array(LEGS)
        if source == 'real':
            arrays['fs'] = np.int64(int(real['fs']) if real is not None else 48000)
        arrays['meta'] = np.array(json.dumps(summary['settings']))
        np.savez_compressed(a.out / f'impacts_{source}.npz', **arrays)
        print(f'{source}: {len(tab)} touchdowns, {len(windows[source])} impact windows')
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
