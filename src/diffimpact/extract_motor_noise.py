#!/usr/bin/env python3
"""착지가 없는 기록(기본: real_sim/NORMAL_07)에서 모터·배경 소음을 상태별로 잘라 내고,
보행 중 착지 사이 소음과 스펙트럼을 비교한다. DiffImpact 배경 소음 항을 맞추는 데 쓴다.

상태 분류 (0.1 s 블록, 실로봇 lowstate/lowcmd 기준)
  stand    : 모터 제어 중(lowcmd 수신) + 평균 |tau| >= 1 N·m + 평균 |dq| < 0.08 rad/s  → 자세 유지 모터 소음
  motor_off: lowcmd 끊김 또는 평균 |tau| < 0.3 + 평균 |dq| < 0.08                    → 모터 거의 꺼짐(주변 소음 기준선)
  moving   : 그 밖 (자세 변화). 소음 데이터에서 제외하고 구간만 기록
  같은 상태가 MIN_RUN 이상 이어진 구간만 쓰고, 양 끝 TRIM 만큼 잘라 전이 구간을 뺀다.
  구간을 CHUNK(0.5 s) 조각으로 나누고, 조각 RMS 가 그 상태 중앙값 + 6 dB 를 넘으면 outlier 표시
  (사람 소리·물건 소리 같은 돌발음).

보행 중 비교 소음 (data/touchdown_w300_500hz/pairs_real.npz — 착지 후 +130~250 ms 가 필요해 300 ms 창 사용)
  walk_pre   : 세트 첫 착지 전 -50 ~ -10 ms (유각기, 직전 세트 충격음 끝난 뒤)
  walk_inter : 세트 첫 착지 후 +130 ~ +250 ms (두 번째 발 착지와 충격음 꼬리 이후, 다음 세트 전)

출력 : data/motor_noise/
  noise_chunks.npz      상태별 0.5 s 소리 조각 (int16) + 시각·상태·outlier
  segments.csv          상태 구간 표
  psd.csv               같은 조건(40 ms Hann, Welch)으로 구한 상태별 평균 PSD [dB re full scale]
  band_levels.csv       대역별 레벨 비교
  noise_spectra.png     PSD 비교 그림
  summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
BLOCK = 0.1
MIN_RUN = 1.0
TRIM = 0.3
CHUNK = 0.5
OUTLIER_DB = 6.0
SEG = 0.04                          # PSD 조각 길이 = 착지 전 창 길이(40 ms)와 같게
BANDS = [(20, 200), (200, 1000), (1000, 4000), (4000, 12000), (12000, 24000)]


def classify(z):
    t0 = z['ls_t'][0]
    t = z['ls_t'] - t0
    lc_t = z['lc_t'] - t0 if len(z['lc_t']) else np.array([])
    edges = np.arange(0, t[-1] + BLOCK, BLOCK)
    states = []
    for s in edges[:-1]:
        m = (t >= s) & (t < s + BLOCK)
        if not m.any():
            states.append('none'); continue
        tau = np.abs(z['ls_tau'][m]).mean()
        dq = np.abs(z['ls_dq'][m]).mean()
        cmd_on = len(lc_t) and np.any((lc_t >= s) & (lc_t < s + BLOCK))
        if dq < 0.08 and (not cmd_on or tau < 0.3):
            states.append('motor_off')
        elif dq < 0.08 and cmd_on and tau >= 1.0:
            states.append('stand')
        else:
            states.append('moving')
    runs, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            runs.append(dict(state=states[start], t_start=edges[start], t_end=edges[i]))
            start = i
    seg = pd.DataFrame(runs)
    seg['duration_s'] = seg.t_end - seg.t_start
    for k in ('tau', 'dq'):
        seg[f'mean_abs_{k}'] = [np.abs(z[f'ls_{k}'][(t >= r.t_start) & (t < r.t_end)]).mean() for r in seg.itertuples()]
    return seg, t0


def dbfs(x):
    x = np.asarray(x, float); x = x - x.mean()
    return 20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-12) / 32768.)


def psd_db(clips, fs):
    """조각마다 40 ms Hann Welch PSD → 평균 (선형) → dB. 입력 길이가 달라도 같은 조건."""
    n = int(SEG * fs)
    acc, count = None, 0
    for c in clips:
        c = np.asarray(c, float) / 32768.
        if len(c) < n:
            continue
        f, p = welch(c - c.mean(), fs=fs, window='hann', nperseg=n, noverlap=n // 2, detrend=False)
        acc = p * 1 if acc is None else acc + p
        count += 1
    return f, 10 * np.log10(acc / count + 1e-30), count


def band_level(f, p_db, lo, hi):
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(np.sum(10 ** (p_db[m] / 10)) * (f[1] - f[0]) + 1e-30)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--file', type=Path, default=SRC / 'sim_to_real_data/real_sim/NORMAL_07_20261118_174116.npz')
    p.add_argument('--pairs', type=Path, default=HERE / 'data/touchdown_w300_500hz/pairs_real.npz')
    p.add_argument('--out', type=Path, default=HERE / 'data/motor_noise')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    with np.load(a.file, allow_pickle=False) as z:
        z = {k: z[k] for k in z.files}
    fs = int(z['fs'])
    seg, t0 = classify(z)
    audio = z['audio']
    ta0 = float(z['audio_t0']) - t0
    slope = float(z['audio_slope'])
    a_end = ta0 + slope * len(audio)

    seg['used'] = False
    seg['use_start'] = seg.t_start + TRIM
    seg['use_end'] = np.minimum(seg.t_end - TRIM, a_end)
    seg['use_start'] = np.maximum(seg.use_start, ta0)
    chunks, meta = [], []
    n_chunk = int(CHUNK * fs)
    for r in seg.itertuples():
        if r.state not in ('stand', 'motor_off') or r.duration_s < MIN_RUN or r.use_end - r.use_start < CHUNK:
            continue
        seg.loc[r.Index, 'used'] = True
        s = r.use_start
        while s + CHUNK <= r.use_end + 1e-9:
            i = int(round((s - ta0) / slope))
            clip = audio[i:i + n_chunk]
            if len(clip) == n_chunk:
                chunks.append(clip)
                meta.append(dict(state=r.state, segment=int(r.Index), t_start=s, dbfs=dbfs(clip)))
            s += CHUNK
    meta = pd.DataFrame(meta)
    meta['outlier'] = meta.dbfs > meta.groupby('state').dbfs.transform('median') + OUTLIER_DB
    seg.to_csv(a.out / 'segments.csv', index=False, float_format='%.4g')
    np.savez_compressed(a.out / 'noise_chunks.npz', audio=np.stack(chunks).astype(np.int16),
                        state=meta.state.to_numpy().astype(str), t_start=meta.t_start.to_numpy(),
                        dbfs=meta.dbfs.to_numpy(), outlier=meta.outlier.to_numpy(), fs=np.int64(fs),
                        source=np.array(a.file.name),
                        meta=np.array(json.dumps(dict(chunk_s=CHUNK, trim_s=TRIM, min_run_s=MIN_RUN,
                                                      outlier_db=OUTLIER_DB, t_start='s from ls_t[0]'))))

    # ---- 보행 중 소음 (대각 쌍 세트 창에서)
    with np.load(a.pairs, allow_pickle=False) as pz:
        pa = pz['audio']; walk = pz['walk'].astype(str); tr = pz['t_rel_ms']
        assert int(pz['fs']) == fs
    pre_ms = float(-tr[0])
    def cut(lo_ms, hi_ms):
        i0 = int(round((lo_ms + pre_ms) / 1e3 * fs)); i1 = int(round((hi_ms + pre_ms) / 1e3 * fs))
        return pa[:, i0:i1]
    walk_pre, walk_inter = cut(-50, -10), cut(130, 250)

    clean = ~meta.outlier.to_numpy()
    sets = {
        'stand (NORMAL_07)': [c for c, s, ok in zip(chunks, meta.state, clean) if s == 'stand' and ok],
        'motor_off (NORMAL_07)': [c for c, s, ok in zip(chunks, meta.state, clean) if s == 'motor_off' and ok],
    }
    for w in ('straight', 'lateral'):
        sets[f'walk_pre {w}'] = list(walk_pre[walk == w])
        sets[f'walk_inter {w}'] = list(walk_inter[walk == w])
    spectra, counts, levels = {}, {}, []
    for name, clips in sets.items():
        f, pdb, n = psd_db(clips, fs)
        spectra[name] = pdb; counts[name] = n
        row = dict(set=name, clips=n, rms_dbfs_median=float(np.median([dbfs(c) for c in clips])))
        for lo, hi in BANDS:
            row[f'{lo}-{hi}Hz_db'] = band_level(f, pdb, lo, hi)
        levels.append(row)
    levels = pd.DataFrame(levels)
    pd.DataFrame(dict(freq_hz=f, **spectra)).to_csv(a.out / 'psd.csv', index=False, float_format='%.5g')
    levels.to_csv(a.out / 'band_levels.csv', index=False, float_format='%.2f')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    styles = {'stand': ('k', '-'), 'motor_off': ('gray', '--'), 'walk_pre straight': ('tab:blue', '-'),
              'walk_inter straight': ('tab:blue', ':'), 'walk_pre lateral': ('tab:red', '-'),
              'walk_inter lateral': ('tab:red', ':')}
    for name, pdb in spectra.items():
        c, ls = styles[name.replace(' (NORMAL_07)', '')]
        ax[0].semilogx(f[1:], pdb[1:], color=c, ls=ls, lw=1.2, label=f'{name} (n={counts[name]})')
        ax[1].semilogx(f[1:], pdb[1:] - spectra['motor_off (NORMAL_07)'][1:], color=c, ls=ls, lw=1.2, label=name)
    ax[0].set_ylabel('PSD [dB re FS²/Hz]'); ax[0].set_title('noise PSD (40 ms Hann, Welch average)')
    ax[1].set_ylabel('dB relative to motor_off'); ax[1].set_title('relative to motor-off baseline')
    ax[1].axhline(0, color='gray', lw=.8)
    for x in ax:
        x.set_xlim(20, fs / 2); x.grid(alpha=.3, which='both'); x.set_xlabel('Hz'); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out / 'noise_spectra.png', dpi=100); plt.close(fig)

    summary = dict(source=a.file.name, fs=fs,
                   segments=seg.round(3).to_dict('records'),
                   chunks={s: dict(total=int((meta.state == s).sum()),
                                   outliers=int((meta.outlier & (meta.state == s)).sum()),
                                   outlier_t_start=meta.t_start[meta.outlier & (meta.state == s)].round(2).tolist(),
                                   median_dbfs=float(meta.dbfs[meta.state == s].median()))
                           for s in ('stand', 'motor_off')},
                   band_levels=levels.round(2).to_dict('records'))
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float) + '\n')
    print(seg[['state', 't_start', 't_end', 'duration_s', 'mean_abs_tau', 'mean_abs_dq', 'used']].round(2).to_string())
    print(json.dumps(summary['chunks'], indent=1))
    print(levels.round(1).to_string())


if __name__ == '__main__':
    main()
