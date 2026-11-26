#!/usr/bin/env python3
"""연속 음향 예측: 녹화 구간 전체의 착지 소리를 DiffImpact 로 합성해 실측과 비교 (MuJoCo 적용 모습).

세트(대각 쌍)마다 모델 창(-20 ~ +80 ms)의 결정적 충격음 + 확률적 충격음 실현값을 만들어
세트 첫 착지 시각(t0_abs, PC 시계)에 겹쳐 더하고, 정상 배경소음(측정값)을 구간 전체에 깐다.
  - 실로봇 상태 기반 : 실로봇 착지 시각·상태(발 운동, 토크 기반 접촉력, 명령)
  - 시뮬 상태 기반   : 같은 시간대 MuJoCo 착지 (보행 위상은 실로봇과 맞지 않음)
창 밖(착지 후 80 ms 이후)은 배경소음만 있으므로, 입각기 중 소리는 예측하지 않는다.

출력: runs/<run>/figures/stream_prediction.png, runs/<run>/wav/stream_<녹화>_{real,pred_real_state,pred_sim_state}.wav
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_diffimpact as T          # noqa: E402
from diffimpact_model import DiffImpact  # noqa: E402
from plot_timing_analysis import BLUE, ORANGE, AQUA, SURFACE, INK, INK2, GRID, gait_of, style  # noqa: E402

RECORDINGS = [('real_sim_straight', 'NORMAL_0.7_0.5_v6'), ('real_sim', 'NORMAL_0.5_0.5_v4')]   # 검증 녹화
SPAN = 3.0
LEVEL_MS = 50.


def raw_audio(data_root, dataset, trial):
    f = sorted(p for p in (data_root / dataset).glob(f'{trial}_*.npz') if not p.stem.endswith('_sim'))[0]
    with np.load(f) as z:
        return z['audio'].astype(np.float64) / 32768., float(z['audio_t0']), float(z['audio_slope']), int(z['fs'])


@torch.no_grad()
def synth_stream(model, d, t_start, n, fs, gen):
    """구간 [t_start, t_start + n/fs) 에 세트 창들을 겹쳐 더한 충격음 (배경소음 제외)."""
    out = model(torch.as_tensor(d['td']), torch.as_tensor(d['legs']), None, torch.as_tensor(d['feats']), jitter=False)
    stoch = model._shaped_noise(len(d['td']), model.stoch_log_psd, gen) * model.stoch_weight(out, model.t_clip).sqrt() \
        if model.stochastic else 0.
    clips = (out['y'] + stoch).numpy()
    y = np.zeros(n)
    for c, t0 in zip(clips, d['t0_abs']):
        s = int(round((t0 - model.pre - t_start) * fs))
        a, b = max(s, 0), min(s + len(c), n)
        if b > a:
            y[a:b] += c[a - s:b - s]
    return y


@torch.no_grad()
def background(model, n, gen):
    w = torch.randn(n, generator=gen, dtype=torch.float64)
    psd = torch.exp(model._interp(model.noise0.double(), n // 2 + 1))
    return torch.fft.irfft(torch.fft.rfft(w) * psd.sqrt(), n=n).numpy()


def level_db(x, fs, ms=LEVEL_MS):
    w = int(ms * 1e-3 * fs); hop = w // 5
    idx = np.arange(0, len(x) - w, hop)
    return (idx + w / 2) / fs, np.array([10 * np.log10(np.mean((x[i:i + w] - x[i:i + w].mean()) ** 2) + 1e-20) for i in idx])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, default=HERE / 'runs/w100_200hz_kin_grf_s0')
    p.add_argument('--data', type=Path, default=HERE / 'data/touchdown_w100_200hz')
    p.add_argument('--raw', type=Path, default=HERE.parent / 'sim_to_real_data')
    a = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ('NotoSansCJK-Regular.ttc', 'NotoSansCJK-Bold.ttc'):
        path = Path('/usr/share/fonts/opentype/noto') / f
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = next((n for n in ('Noto Sans CJK KR', 'Noto Sans CJK JP') if n in names), 'DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False

    ck = torch.load(a.run / 'model_feat.pt', weights_only=False)
    T.FEATURES = list(ck['features'])
    model = DiffImpact(**ck['config']); T.load_state(model, ck['state']); model.eval()
    sets = {}
    for src in ('real', 'sim'):
        d = T.load_pairs(a.data / f'pairs_{src}.npz')
        d['feats'] = ((d['feats'] - ck['feat_mu']) / ck['feat_sd']).astype(np.float32)
        with np.load(a.data / f'pairs_{src}.npz') as z:
            d['t0_abs'] = z['t0_abs']
        sets[src] = d
    (a.run / 'wav').mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 16), facecolor=SURFACE)
    gs = fig.add_gridspec(6, len(RECORDINGS), height_ratios=[1, 1, 1, 1.2, 1.1, 1.1], hspace=.55, wspace=.12,
                          left=.05, right=.99, top=.875, bottom=.04)
    summary = []
    for c, (ds, tr) in enumerate(RECORDINGS):
        audio, a_t0, slope, fs = raw_audio(a.raw, ds, tr)
        sel = {s: (sets[s]['dataset'] == ds) & (sets[s]['trial'] == tr) for s in sets}
        # 실로봇·시뮬 세트가 모두 있는 구간에서 3초: 실로봇 세트 시각 중앙에서 시작
        tr_t, si_t = sets['real']['t0_abs'][sel['real']], sets['sim']['t0_abs'][sel['sim']]
        lo = max(tr_t.min(), si_t.min()); hi = min(tr_t.max(), si_t.max())
        t_start = max(lo, (lo + hi) / 2 - SPAN / 2) - .1
        n = int(SPAN * fs)
        i0 = int(round((t_start - a_t0) / slope))
        real = audio[i0:i0 + n] - audio[i0:i0 + n].mean()
        gen = torch.Generator().manual_seed(7)
        bg = background(model, n, gen)
        pr_real = synth_stream(model, T.subset(sets['real'], sel['real']), t_start, n, fs, gen) + bg
        pr_sim = synth_stream(model, T.subset(sets['sim'], sel['sim']), t_start, n, fs, gen) + background(model, n, gen)
        t = np.arange(n) / fs
        td_real = tr_t[(tr_t >= t_start) & (tr_t < t_start + SPAN)] - t_start
        td_sim = si_t[(si_t >= t_start) & (si_t < t_start + SPAN)] - t_start
        amp = np.abs(np.r_[real, pr_real, pr_sim]).max() * 1.05

        name = tr.replace('NORMAL_', '')
        rows = [(real, '실측', BLUE, td_real), (pr_real, '예측 · 실로봇 상태 기반', ORANGE, td_real),
                (pr_sim, '예측 · 시뮬(MuJoCo) 상태 기반', AQUA, td_sim)]
        for r, (x, label, col, tds) in enumerate(rows):
            ax = fig.add_subplot(gs[r, c]); style(ax)
            ax.plot(t, x, color=col, lw=.35)
            for tt in tds:
                ax.axvline(tt, color=INK2, lw=.6, ls=':', zorder=1)
            ax.set_xlim(0, SPAN); ax.set_ylim(-amp, amp)
            ax.set_title((f'{gait_of(ds, tr)} · {name}\n' if r == 0 else '') + f'{label} (점선 = 대각 쌍 첫 착지)',
                         loc='left', fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel('진폭 [FS]')
        ax = fig.add_subplot(gs[3, c]); style(ax)
        lv = {}
        for x, label, col, _ in rows:
            tl, l = level_db(x, fs); lv[label] = l
            ax.plot(tl, l, color=col, lw=1.4 if label == '실측' else 1.2, label=label)
        ax.set_xlim(0, SPAN); ax.set_ylabel(f'{LEVEL_MS:.0f} ms RMS [dBFS]')
        top = max(v.max() for v in lv.values()) + 1
        ax.set_ylim(-70, top)                                     # 아래쪽 여백에 범례
        ax.legend(frameon=False, fontsize=8, loc='lower center', ncol=3)
        r_rs = np.corrcoef(lv['실측'], lv['예측 · 실로봇 상태 기반'])[0, 1]
        mean_db = {k: 10 * np.log10(np.mean(v ** 2)) for k, v in
                   (('실측', real), ('실로봇 상태', pr_real), ('시뮬 상태', pr_sim))}
        ax.set_title(f'레벨 시계열 · 실측 vs 실로봇 상태 예측 상관 r = {r_rs:.2f} · 구간 평균 '
                     f'실측 {mean_db["실측"]:.1f} / 실로봇 상태 {mean_db["실로봇 상태"]:.1f} / 시뮬 상태 {mean_db["시뮬 상태"]:.1f} dBFS',
                     loc='left', fontsize=9.5, color=INK)
        for r, (x, label) in enumerate(((real, '실측'), (pr_real, '예측 · 실로봇 상태 기반'))):
            ax = fig.add_subplot(gs[4 + r, c]); style(ax); ax.grid(False)
            ax.specgram(x, NFFT=1024, Fs=fs, noverlap=768, cmap='magma', vmin=-150, vmax=-70)
            ax.set_ylim(0, 8000); ax.set_xlim(0, SPAN)
            ax.set_title(f'스펙트로그램 · {label}', loc='left', fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel('Hz')
        ax.set_xlabel('시간 [s]')
        for x, tag in ((real, 'real'), (pr_real, 'pred_real_state'), (pr_sim, 'pred_sim_state')):
            wavfile.write(a.run / f'wav/stream_{name}_{tag}.wav', fs, np.clip(x * 32768, -32768, 32767).astype(np.int16))
        summary.append((name, r_rs, mean_db))
        print(name, f'level r={r_rs:.2f}', {k: round(v, 1) for k, v in mean_db.items()})

    fig.suptitle(f'연속 음향 예측 — 실측 vs DiffImpact ({a.run.name}), 검증 녹화 3초 구간', x=.05, ha='left',
                 fontsize=15, color=INK, fontweight='bold')
    fig.text(.05, .93, '세트마다 모델 창(-20 ~ +80 ms)을 착지 시각에 겹쳐 더하고 측정 배경소음을 깖 · 창 밖 입각기 소리는 예측하지 않음 · '
             '시뮬은 같은 시간대 MuJoCo 착지(보행 위상이 실로봇과 다름)', fontsize=9.5, color=INK2)
    out = a.run / 'figures' / 'stream_prediction.png'
    fig.savefig(out, dpi=100, facecolor=SURFACE); plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
