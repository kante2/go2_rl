#!/usr/bin/env python3
"""대각 쌍 착지 타이밍·발 속도와 착지 소리의 관계 그림 (실로봇 vs 시뮬, DiffImpact v5).

출력 : runs/<run>/figures/timing_analysis.png
  A. 앞발이 먼저 닿는 비율 (보행 종류 × 실로봇/시뮬)
  B. 두 발 착지 간격 분포 (누적 분포)
  C. 실로봇: 간격·순서에 따른 착지 소리 레벨 (녹화 평균 제거, 같은 녹화 안 비교)
  D. 모델: 시뮬 착지 타이밍만 바꿨을 때 예측 레벨 변화 vs 실측 방향
  E. 녹화별 착지 직전 발 하강 속도: 실로봇 vs 시뮬
  F. 녹화별 실로봇 앞발 하강 속도 vs 착지 소리 레벨
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

# 색: 참고 팔레트 1~3번 (all-pairs 검증 통과). 실로봇/시뮬 = 파랑/주황, 셋째 범주 = 청록
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
GAITS = ('직진', '횡방향', '제자리 회전')
GAIT_COLORS = dict(zip(GAITS, (BLUE, ORANGE, AQUA)))


def gait_of(dataset, trial):
    """폴더 이름이 아니라 실제 명령으로 분류 (real_sim 의 NORMAL_01/04 는 직진)."""
    if dataset == 'real_sim_straight' or trial in ('NORMAL_01', 'NORMAL_04'):
        return '직진'
    return '제자리 회전' if trial.startswith('NORMAL_0.5_0.5_0.5') else '횡방향'


def style(ax):
    ax.set_facecolor(SURFACE)
    for k in ('top', 'right'):
        ax.spines[k].set_visible(False)
    for k in ('left', 'bottom'):
        ax.spines[k].set_color(INK2); ax.spines[k].set_linewidth(.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)


def title(ax, letter, text):
    ax.set_title(f'{letter}. {text}', loc='left', fontsize=11, color=INK, fontweight='bold', pad=8)


def mean_ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.


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
    sim = T.load_pairs(a.data / 'pairs_sim.npz')
    ns = argparse.Namespace(pre=real['pre'], fs=real['fs'])
    level = T.dbfs_window(real['audio'], ns)                      # 착지 후 0~40 ms [dBFS]
    for d in (real, sim):
        d['gait'] = np.array([gait_of(ds, tr) for ds, tr in zip(d['dataset'], d['trial'])])
        d['rec'] = np.char.add(np.char.add(d['dataset'], '/'), d['trial'])
        d['front_first'] = d['td'][:, 0] <= d['td'][:, 1]        # slot 0 = 앞발
        d['spread'] = np.abs(d['td'][:, 0] - d['td'][:, 1]) * 1e3
        d['vz_front'] = d['feats'][:, 0, T.FEATURES.index('foot_vz_pre')].copy()               # 아직 정규화 전 값
        d['vz_rear'] = d['feats'][:, 1, T.FEATURES.index('foot_vz_pre')].copy()
    resid = level - np.array([level[real['rec'] == r].mean() for r in real['rec']])

    # 모델: 시뮬 착지에 타이밍만 바꿔 예측
    model = DiffImpact(**ck['config']); T.load_state(model, ck['state']); model.eval()
    sim_n = dict(sim); sim_n['feats'] = ((sim['feats'] - ck['feat_mu']) / ck['feat_sd']).astype(np.float32)

    def predict(td):
        d = dict(sim_n); d['td'] = td.astype(np.float32)
        f = d['feats'].copy()
        if 'is_second' in T.FEATURES:
            f[:, :, T.FEATURES.index('is_second')] = (td > 0).astype(np.float32)
        d['feats'] = f
        return T.predicted_post40(model, T.evaluate(model, d, 'cpu', draws=8)['samples'])
    base = predict(sim['td'])
    front_first_td = np.stack([np.zeros(len(sim['td'])), sim['spread'] / 1e3], 1)
    manip = [('순서 뒤집기', predict(sim['td'][:, ::-1].copy()) - base),
             ('동시 착지로', predict(np.zeros_like(sim['td'])) - base),
             ('앞발 먼저로\n(간격 유지)', predict(front_first_td) - base)]

    fig = plt.figure(figsize=(16, 15), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, hspace=.42, wspace=.22, left=.06, right=.98, top=.91, bottom=.05)
    ax = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for x in ax:
        style(x)

    # A. 앞발 먼저 비율
    x = np.arange(len(GAITS)); w = .36
    for k, (d, name, c) in enumerate(((real, '실로봇', BLUE), (sim, '시뮬', ORANGE))):
        vals = [100 * d['front_first'][d['gait'] == g].mean() for g in GAITS]
        ns_ = [(d['gait'] == g).sum() for g in GAITS]
        bars = ax[0].bar(x + (k - .5) * w, vals, w - .03, color=c, label=name, zorder=3)
        for b, v, n in zip(bars, vals, ns_):
            ax[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}%', ha='center', fontsize=9, color=INK)
    ax[0].axhline(50, color=INK2, lw=.8, ls='--', zorder=2)
    ax[0].text(2.62, 51.5, '반반', fontsize=8, color=INK2, ha='right')
    ax[0].set_xticks(x, GAITS); ax[0].set_ylim(0, 105); ax[0].set_ylabel('앞발이 먼저 닿는 세트 비율 [%]')
    ax[0].legend(frameon=False, fontsize=9, loc='upper right')
    title(ax[0], 'A', '착지 순서: 실로봇은 앞발 먼저, 시뮬은 뒷발 먼저 (직진)')

    # B. 간격 누적 분포
    for d, name, c in ((real, '실로봇', BLUE), (sim, '시뮬', ORANGE)):
        v = np.sort(d['spread'])
        ax[1].step(v, np.arange(1, len(v) + 1) / len(v) * 100, where='post', color=c, lw=2, label=f'{name} (중앙값 {np.median(v):.0f} ms)')
    ax[1].set_xlim(0, 70); ax[1].set_ylim(0, 102)
    ax[1].set_xlabel('대각 두 발 착지 간격 [ms]'); ax[1].set_ylabel('누적 비율 [%]')
    ax[1].legend(frameon=False, fontsize=9, loc='lower right')
    title(ax[1], 'B', '착지 간격 크기는 실로봇·시뮬이 비슷함')

    # C. 실로봇 간격/순서 효과
    bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 80)]
    centers, means, cis, counts = [], [], [], []
    for lo, hi in bins:
        m = (real['spread'] >= lo) & (real['spread'] < hi)
        mu, ci = mean_ci(resid[m]); centers.append(f'{lo}–{hi}' if hi < 80 else f'{lo}+'); means.append(mu); cis.append(ci); counts.append(m.sum())
    xc = np.arange(len(bins))
    ax[2].errorbar(xc, means, yerr=cis, color=BLUE, lw=2, marker='o', ms=8, mec=SURFACE, mew=2, capsize=4, zorder=3)
    for xi, mu, ci, n in zip(xc, means, cis, counts):
        ax[2].text(xi + .12, mu, f'{mu:+.1f} dB\n(n={n})', fontsize=8, color=INK, va='center')
    ax[2].axhline(0, color=INK2, lw=.8)
    ax[2].set_xticks(xc, centers); ax[2].set_xlabel('대각 두 발 착지 간격 [ms]')
    ax[2].set_ylabel('착지 후 40 ms 레벨 − 녹화 평균 [dB]')
    r = np.corrcoef(real['spread'], resid)[0, 1]
    ff, rf = mean_ci(resid[real['front_first']]), mean_ci(resid[~real['front_first']])
    ax[2].text(.98, .97, f'간격과 상관 r = {r:+.2f}\n순서 차이: 앞발 먼저 {ff[0]:+.2f} / 뒷발 먼저 {rf[0]:+.2f} dB',
               transform=ax[2].transAxes, ha='right', va='top', fontsize=9, color=INK)
    title(ax[2], 'C', '실로봇: 동시에 닿을수록 큼, 순서는 영향 거의 없음')

    # D. 모델 타이밍 반응
    labels = [m[0] for m in manip]
    vals = [m[1].mean() for m in manip]
    cis_d = [mean_ci(m[1])[1] for m in manip]
    yd = np.arange(len(manip))
    ax[3].barh(yd, vals, .5, xerr=cis_d, color=BLUE, capsize=4, zorder=3, label=f'모델({a.run.name}) 예측 변화')
    for yi, v in zip(yd, vals):
        ax[3].text(v - .05 if v < 0 else v + .05, yi + .32, f'{v:+.2f} dB', fontsize=9, color=INK, ha='right' if v < 0 else 'left')
    real_effect = means[0] - np.average(means, weights=counts)       # 실측: 간격 <5 ms 세트가 평균보다 큰 정도
    ax[3].scatter([real_effect], [1], marker='D', s=70, color=ORANGE, edgecolor=SURFACE, lw=1.5, zorder=4,
                  label='실측 참고: 간격 <5 ms 세트 − 전체 평균')
    ax[3].annotate(f'실측: 동시 착지 세트가 {real_effect:+.1f} dB\n→ 모델은 반대 방향', (real_effect, 1), (real_effect - .05, 1.55),
                   fontsize=8.5, color=INK, ha='right', arrowprops=dict(arrowstyle='-', color=INK2, lw=.8))
    ax[3].set_xlim(-1.6, 1.3)
    ax[3].axvline(0, color=INK2, lw=.8)
    ax[3].set_yticks(yd, labels); ax[3].invert_yaxis()
    ax[3].set_xlabel('시뮬 착지에서 타이밍만 바꿨을 때 예측 레벨 변화 [dB]')
    ax[3].legend(frameon=False, fontsize=8.5, loc='lower right')
    title(ax[3], 'D', '모델: 타이밍 반응이 실측과 어긋남 (수정 필요)')

    # E. 녹화별 발 하강 속도
    recs = sorted(set(real['rec']) & set(sim['rec']))
    rows = []
    for rec in recs:
        mr, ms = real['rec'] == rec, sim['rec'] == rec
        rows.append((rec, real['gait'][mr][0], np.median(real['vz_front'][mr]), np.median(sim['vz_front'][ms]),
                     np.median(real['vz_rear'][mr]), np.median(sim['vz_rear'][ms]), np.median(level[mr])))
    lim = (-1.12, -.42)
    ax[4].plot(lim, lim, color=INK2, lw=.8, ls='--', zorder=2)
    ax[4].text(-.47, -.45, '실로봇 = 시뮬', fontsize=8, color=INK2, ha='right', rotation=45)
    for g in GAITS:
        sel = [r for r in rows if r[1] == g]
        if not sel:
            continue
        ax[4].scatter([r[2] for r in sel], [r[3] for r in sel], s=64, marker='o', color=GAIT_COLORS[g],
                      edgecolor=SURFACE, lw=1.5, zorder=3, label=f'{g} · 앞발')
        ax[4].scatter([r[4] for r in sel], [r[5] for r in sel], s=64, marker='^', color=GAIT_COLORS[g],
                      edgecolor=SURFACE, lw=1.5, zorder=3, label=f'{g} · 뒷발')
    fast = [r for r in rows if r[5] < -.95]
    if fast:
        ax[4].annotate(f'직진 0.7 v1–v3 뒷발\n(실로봇 {np.mean([r[4] for r in fast]):.2f}, 시뮬 {np.mean([r[5] for r in fast]):.2f})',
                       (fast[0][4], fast[0][5]), (fast[0][4] - .3, fast[0][5] + .07), fontsize=8, color=INK,
                       arrowprops=dict(arrowstyle='-', color=INK2, lw=.8))
    ax[4].set_xlim(lim); ax[4].set_ylim(lim)
    ax[4].set_xlabel('실로봇: 착지 직전 발 수직속도 중앙값 [m/s]'); ax[4].set_ylabel('시뮬: 같은 녹화의 값 [m/s]')
    ax[4].legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
    cf = np.corrcoef([r[2] for r in rows], [r[3] for r in rows])[0, 1]
    cr = np.corrcoef([r[4] for r in rows], [r[5] for r in rows])[0, 1]
    ax[4].text(.98, .2, f'녹화 간 상관: 앞발 {cf:.2f} / 뒷발 {cr:.2f}\n점선 아래 = 시뮬이 더 빨리 내려옴', transform=ax[4].transAxes,
               ha='right', va='bottom', fontsize=9, color=INK)
    title(ax[4], 'E', '시뮬은 일부 조건에서 발이 더 빨리 내려옴')

    # F. 실로봇 앞발 속도 vs 레벨
    for g in GAITS:
        sel = [r for r in rows if r[1] == g]
        ax[5].scatter([r[2] for r in sel], [r[6] for r in sel], s=64, color=GAIT_COLORS[g], edgecolor=SURFACE, lw=1.5, zorder=3, label=g)
    xv = np.array([r[2] for r in rows]); yv = np.array([r[6] for r in rows])
    k, b0 = np.polyfit(xv, yv, 1); xs = np.linspace(xv.min(), xv.max(), 10)
    ax[5].plot(xs, k * xs + b0, color=INK2, lw=1.2, zorder=2)
    ax[5].text(.98, .97, f'r = {np.corrcoef(xv, yv)[0, 1]:+.2f}\n{abs(k) / 10:.1f} dB / 0.1 m/s', transform=ax[5].transAxes,
               ha='right', va='top', fontsize=9, color=INK)
    ax[5].set_xlabel('실로봇: 착지 직전 앞발 수직속도 중앙값 [m/s]  (← 빨리 내려옴)')
    ax[5].set_ylabel('실로봇: 착지 후 40 ms 레벨 중앙값 [dBFS]')
    ax[5].legend(frameon=False, fontsize=9, loc='lower left')
    title(ax[5], 'F', '실로봇: 발이 빨리 내려올수록 착지 소리가 큼 (녹화별)')

    fig.suptitle(f'대각 쌍 착지 타이밍·발 속도와 착지 소리 — 실로봇 vs 시뮬, DiffImpact {a.run.name}', x=.06, ha='left',
                 fontsize=15, color=INK, fontweight='bold')
    fig.text(.06, .935, f'실로봇 세트 {len(real["td"])}개, 시뮬 세트 {len(sim["td"])}개, 녹화 {len(recs)}개 · '
             '레벨 = 세트 첫 착지 후 0~40 ms RMS · 오차 막대 = 95% 신뢰구간 · 시뮬 예측은 소음 샘플 8회 평균',
             fontsize=9.5, color=INK2)
    out = a.run / 'figures' / 'timing_analysis.png'
    fig.savefig(out, dpi=110, facecolor=SURFACE); plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
