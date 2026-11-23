#!/usr/bin/env python3
"""검증 세트에서 실측 착지 소리와 DiffImpact 예측을 파형·포락선으로 비교.

모델의 확률적 충격음과 배경소음은 위상이 매번 다른 랜덤 실현값이라, 파형을 샘플 단위로 겹쳐
맞추는 비교는 의미가 없다. 그래서 세 가지를 같이 그린다.
  1행  실측 파형
  2행  모델 샘플 파형 (결정적 + 확률적 + 배경소음, 실현값 1개)과 결정적 충격음만 (진한 선)
  3행  2 ms RMS 포락선 [dBFS]: 실측 vs 모델 기대 포락선 (+ 샘플 20회의 10~90% 범위)
마지막 행: 검증 세트 전체의 중앙값 포락선 (보행 종류별). 실측과 같은 통계가 되도록 모델도 실현값 샘플의
         2 ms RMS dB 중앙값을 쓴다 (기대 파워의 dB 와 비교하면 들쭉날쭉한 실측의 dB 중앙값이 낮게 나와 모델이 커 보임).

출력: runs/<run>/figures/waveform_comparison.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_diffimpact as T          # noqa: E402
from diffimpact_model import DiffImpact  # noqa: E402
from plot_timing_analysis import BLUE, ORANGE, AQUA, SURFACE, INK, INK2, GRID, gait_of, style  # noqa: E402

ENV_MS = 2.0
N_DRAWS = 20


def rms_env_db(x, fs, ms=ENV_MS):
    """이동 평균 RMS 포락선 [dBFS] (B,N)."""
    w = max(int(ms * 1e-3 * fs), 1)
    k = np.ones(w) / w
    p = np.array([np.convolve(v * v, k, 'same') for v in np.atleast_2d(x)])
    return 10 * np.log10(p + 1e-20)


@torch.no_grad()
def expected_env_db(model, out, fs, ms=ENV_MS):
    """모델 기대 포락선: 결정적 파워(이동 평균) + 확률적 충격음 분산 + 배경소음 분산."""
    w = max(int(ms * 1e-3 * fs), 1)
    k = np.ones(w) / w
    det = np.array([np.convolve(v * v, k, 'same') for v in out['y'].cpu().numpy()])
    noise = torch.exp(model.noise0).mean().item()
    stoch = 0.
    if model.stochastic:
        stoch = (model.stoch_weight(out, model.t_clip) * torch.exp(model.stoch_log_psd).mean()).cpu().numpy()
    return 10 * np.log10(det + stoch + noise + 1e-20)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, default=HERE / 'runs/w100_200hz_kin_grf_s0')
    p.add_argument('--data', type=Path, default=HERE / 'data/touchdown_w100_200hz')
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
    real = T.load_pairs(a.data / 'pairs_real.npz')
    is_val = np.array([(d, t) in T.VAL_TRIALS for d, t in zip(real['dataset'], real['trial'])])
    val = T.subset(real, is_val)
    model = DiffImpact(**ck['config']); T.load_state(model, ck['state']); model.eval()
    val['feats'] = ((val['feats'] - ck['feat_mu']) / ck['feat_sd']).astype(np.float32)
    fs, pre = model.fs, model.pre
    t_ms = (np.arange(model.N) / fs - pre) * 1e3
    gait = np.array([gait_of(d, t) for d, t in zip(val['dataset'], val['trial'])])
    level = T.dbfs_window(val['audio'], argparse.Namespace(pre=pre, fs=fs))

    with torch.no_grad():
        out = model(torch.as_tensor(val['td']), torch.as_tensor(val['legs']), None, torch.as_tensor(val['feats']), jitter=False)
        gens = [torch.Generator().manual_seed(100 + k) for k in range(N_DRAWS)]
        samples = np.stack([model.sample(out, g).numpy() for g in gens])          # (draws,G,N)
    det = out['y'].numpy()
    env_real = rms_env_db(val['audio'], fs)
    env_exp = expected_env_db(model, out, fs)
    env_samp = np.stack([rms_env_db(s, fs) for s in samples])
    pred_level = np.mean([T.dbfs_window(s, argparse.Namespace(pre=pre, fs=fs)) for s in samples], 0)

    # 녹화마다 실측 레벨이 중앙값에 가장 가까운 세트 1개 (대표 예시)
    recs = [f'{d}/{t}' for d, t in T.VAL_TRIALS]
    rec_of = np.char.add(np.char.add(val['dataset'], '/'), val['trial'])
    picks = []
    for r in recs:
        idx = np.flatnonzero(rec_of == r)
        picks.append(idx[np.argmin(np.abs(level[idx] - np.median(level[idx])))])

    fig = plt.figure(figsize=(18, 15), facecolor=SURFACE)
    gs = fig.add_gridspec(4, len(picks), height_ratios=[1, 1, 1.15, 1.25], hspace=.5, wspace=.18,
                          left=.05, right=.99, top=.875, bottom=.05)
    amp = max(np.abs(val['audio'][picks]).max(), np.abs(samples[0][picks]).max()) * 1.05
    leg_names = ('FR', 'FL', 'RR', 'RL')
    for c, j in enumerate(picks):
        axs = [fig.add_subplot(gs[r, c]) for r in range(3)]
        for x in axs:
            style(x); x.set_xlim(t_ms[0], t_ms[-1])
        tds = [(val['td'][j, s] * 1e3, leg_names[val['legs'][j, s]]) for s in range(2)]

        def mark(ax, top=True):
            close = abs(tds[0][0] - tds[1][0]) < 12
            for k, (tt, name) in enumerate(tds):
                ax.axvline(tt, color=INK2, lw=.8, ls=':', zorder=1)
                if top:
                    ax.text(tt + 2, .97 - (.09 * k if close else 0), name, transform=ax.get_xaxis_transform(),
                            fontsize=8, color=INK2, va='top')

        axs[0].plot(t_ms, val['audio'][j], color=BLUE, lw=.5)
        axs[0].set_ylim(-amp, amp); mark(axs[0])
        trial = val['trial'][j].replace('NORMAL_', '')
        axs[0].set_title(f'{gait[j]} · {trial} · {val["diagonal"][j]}\n실측 파형', loc='left', fontsize=10, color=INK)

        axs[1].plot(t_ms, samples[0][j], color=ORANGE, lw=.5, label='모델 샘플 (전체)')
        axs[1].plot(t_ms, det[j], color=INK, lw=.8, label='결정적 충격음만')
        axs[1].set_ylim(-amp, amp); mark(axs[1], top=False)
        axs[1].set_title('모델 파형 (확률적 성분은 실현값 1개)', loc='left', fontsize=10, color=INK)
        if c == 0:
            axs[1].legend(frameon=False, fontsize=8, loc='lower right')
            axs[0].set_ylabel('진폭 [FS]'); axs[1].set_ylabel('진폭 [FS]')

        lo, hi = np.percentile(env_samp[:, j], [10, 90], axis=0)
        axs[2].fill_between(t_ms, lo, hi, color=ORANGE, alpha=.18, lw=0, label='모델 샘플 10–90%')
        axs[2].plot(t_ms, env_exp[j], color=ORANGE, lw=1.6, label='모델 기대 포락선')
        axs[2].plot(t_ms, env_real[j], color=BLUE, lw=1.0, label='실측')
        axs[2].set_ylim(-80, -25); mark(axs[2], top=False)
        axs[2].set_xlabel('세트 첫 착지 기준 시간 [ms]')
        axs[2].set_title(f'2 ms RMS 포락선 · 0–40 ms 레벨 실측 {level[j]:.1f} / 예측 {pred_level[j]:.1f} dBFS',
                         loc='left', fontsize=10, color=INK)
        if c == 0:
            axs[2].set_ylabel('dBFS'); axs[2].legend(frameon=False, fontsize=8, loc='upper right')

    # 마지막 행: 검증 세트 전체 중앙값 포락선 (보행 종류별)
    gaits = [g for g in ('직진', '횡방향', '제자리 회전') if (gait == g).any()]
    sub = gs[3, :].subgridspec(1, len(gaits) + 1, wspace=.22, width_ratios=[1] * len(gaits) + [1.1])
    for k, g in enumerate(gaits):
        ax = fig.add_subplot(sub[0, k]); style(ax)
        m = gait == g
        ax.fill_between(t_ms, *np.percentile(env_real[m], [25, 75], axis=0), color=BLUE, alpha=.15, lw=0, label='실측 25–75%')
        ax.plot(t_ms, np.median(env_real[m], 0), color=BLUE, lw=1.4, label='실측 중앙값')
        # 실측과 같은 통계: 실현값 샘플의 2 ms RMS dB → (세트·샘플) 중앙값
        ax.plot(t_ms, np.median(env_samp[:, m].reshape(-1, len(t_ms)), 0), color=ORANGE, lw=1.6, label='모델 샘플 중앙값')
        ax.set_xlim(t_ms[0], t_ms[-1]); ax.set_ylim(-72, -38)
        ax.axvline(0, color=INK2, lw=.8, ls=':')
        ax.set_title(f'검증 세트 전체 · {g} (n={m.sum()})', loc='left', fontsize=10, color=INK, fontweight='bold')
        ax.set_xlabel('세트 첫 착지 기준 시간 [ms]')
        if k == 0:
            ax.set_ylabel('2 ms RMS [dBFS]'); ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax = fig.add_subplot(sub[0, -1]); style(ax)
    for g, c in zip(('직진', '횡방향', '제자리 회전'), (BLUE, ORANGE, AQUA)):
        m = gait == g
        if m.any():
            ax.scatter(level[m], pred_level[m], s=26, color=c, edgecolor=SURFACE, lw=1, label=g, zorder=3)
    lim = (min(level.min(), pred_level.min()) - 1, max(level.max(), pred_level.max()) + 1)
    ax.plot(lim, lim, color=INK2, lw=.8, ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('실측 0–40 ms 레벨 [dBFS]'); ax.set_ylabel('예측 [dBFS]')
    r = np.corrcoef(level, pred_level)[0, 1]
    ax.text(.03, .97, f'세트별 r = {r:.2f}\nMAE = {np.abs(level - pred_level).mean():.2f} dB',
            transform=ax.transAxes, va='top', fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title('검증 세트 전체 · 착지 직후 레벨', loc='left', fontsize=10, color=INK, fontweight='bold')

    fig.suptitle(f'실측 vs DiffImpact 예측 착지 소리 — 검증 녹화 4개 ({a.run.name}, 크기는 착지 상태로 예측)',
                 x=.05, ha='left', fontsize=15, color=INK, fontweight='bold')
    fig.text(.05, .93, '위 3행: 녹화마다 실측 레벨이 중앙값에 가장 가까운 세트 · 점선 = 각 발 착지 시각 (힘 센서 기준) · '
             '모델 확률적 성분은 위상이 랜덤이라 파형 일치가 아니라 포락선으로 비교', fontsize=9.5, color=INK2)
    path = a.run / 'figures' / 'waveform_comparison.png'
    fig.savefig(path, dpi=105, facecolor=SURFACE); plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
