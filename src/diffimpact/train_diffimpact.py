#!/usr/bin/env python3
"""DiffImpact 학습: 트롯 대각 쌍 세트(pairs_real.npz) → 착지 소리 합성.

단계 1 (free, 식별)  : 학습 녹화로 모달·공간·소음·펄스 + 세트별 충격 크기/시각 미세조정을 맞춘다.
단계 2 (feat, 예측)  : 단계 1의 공통 파라미터에서 시작해, 충격 크기를 착지 상태 특징으로 예측하게 학습.
                       특징은 시뮬에도 있는 값만 사용 (발 센서 크기는 실/시뮬 단위가 달라 제외).
평가                 : 녹화 단위로 분리한 검증 녹화에서
                       - 모델(feat), 소음만, 학습 평균 템플릿의 로그 스펙트럼 거리·spectral convergence
                       - 착지 후 40 ms 소리 레벨 예측 (MAE, 상관)
                       그리고 시뮬 세트(pairs_sim.npz)에 적용해 녹화별 예측 레벨을 실측과 비교.

출력 : runs/<name>/  model_free.pt, model_feat.pt (레벨 보정본), model_feat_raw.pt, metrics.json, history.csv,
       figures/*.png, wav/*.wav
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

from diffimpact_model import DiffImpact, FFT_SIZES, power_spec, spectral_terms, template_terms

HERE = Path(__file__).resolve().parent
LEGS = ('FR', 'FL', 'RR', 'RL')
VAL_TRIALS = [('real_sim_straight', 'NORMAL_0.7_0.5_v6'), ('real_sim_straight', 'NORMAL_0.5_0.5_v3'),
              ('real_sim', 'NORMAL_0.5_0.5_v4'), ('real_sim', 'NORMAL_0.5_0.5_0.5_v5')]
KIN = ['foot_vx_pre', 'foot_vy_pre', 'foot_vz_pre', 'foot_speed_pre', 'foot_x', 'foot_y', 'foot_z',
       'cmd_vx', 'cmd_vy', 'cmd_wz']
GRF = ['log_grf_rise20', 'log_grf_impulse20']        # 토크 기반 접촉력: 착지 후 20 ms 상승폭 [N], 충격량 [N·s] (로그)
FEATURE_SETS = {
    'kin': KIN + ['is_second'],                      # v5 와 같음
    'kin_grf': KIN + GRF + ['is_second'],
    'kin_grf_spread': KIN + GRF + ['pair_spread_ms'],  # 두 번째 발 여부 대신 두 발 간격 (동시 착지 효과용)
}
FEATURES = FEATURE_SETS['kin']
UNSCALED = ('is_second',)                            # 정규화하지 않는 특징


def run_data_dir(run):
    """실행의 데이터 폴더 (metrics.json 의 data, 상대 경로는 이 폴더 기준). 기록이 없는 예전 실행은 300 ms 창."""
    d = Path(json.loads((run / 'metrics.json').read_text()).get('data', 'data/touchdown_w300_500hz'))
    return d if d.is_absolute() else HERE / d


def load_state(model, state):
    """체크포인트 로드. 고정값인 STFT 창 버퍼(win*)가 없는 예전 체크포인트도 허용."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith('win')] + list(unexpected)
    if bad:
        raise RuntimeError(f'state_dict mismatch: {bad}')
    return model


# ---------------------------------------------------------------- 데이터
def grf_window_features(grf_z, idx, n_pre=10, n_win=20):
    """1 ms 격자 접촉력에서 착지 후 20 ms 상승폭·충격량 (extract_touchdowns.force_features 와 같은 정의)."""
    G, T = grf_z.shape
    rows = np.arange(G)
    base = np.stack([grf_z[rows, np.clip(idx - k, 0, T - 1)] for k in range(1, n_pre + 1)], 1).mean(1)
    win = np.stack([grf_z[rows, np.clip(idx + k, 0, T - 1)] for k in range(0, n_win + 1)], 1)
    rise = win.max(1) - base
    impulse = (win - base[:, None]).sum(1) * 1e-3
    return rise, impulse


def load_pairs(path, features=None):
    features = features or FEATURES
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    t_ms = d['t_rel_ms']
    G = len(d['td_offset_ms'])
    cmd = np.where(np.isfinite(d['cmd_at_td']), d['cmd_at_td'], d['cmd'])
    spread = np.abs(d['td_offset_ms'][:, 0] - d['td_offset_ms'][:, 1])
    feats = np.zeros((G, 2, len(features)), np.float32)
    for s in range(2):
        idx = np.clip(np.round(d['td_offset_ms'][:, s] - t_ms[0]).astype(int), 10, len(t_ms) - 1)
        rows = np.arange(G)
        pre = np.stack([d['foot_vel'][rows, idx - k, s] for k in range(1, 11)], 1).mean(1)   # (G,3)
        pos = d['foot_pos'][rows, idx, s]
        cand = dict(foot_vx_pre=pre[:, 0], foot_vy_pre=pre[:, 1], foot_vz_pre=pre[:, 2],
                    foot_speed_pre=np.linalg.norm(pre, axis=1), foot_x=pos[:, 0], foot_y=pos[:, 1], foot_z=pos[:, 2],
                    cmd_vx=cmd[:, 0], cmd_vy=cmd[:, 1], cmd_wz=cmd[:, 2],
                    is_second=(d['td_offset_ms'][:, s] > 0).astype(float), pair_spread_ms=spread)
        if 'grf' in d:
            rise, impulse = grf_window_features(d['grf'][:, :, s, 2], idx)
            cand.update(log_grf_rise20=np.log(np.maximum(rise, 1.)), log_grf_impulse20=np.log(np.maximum(impulse, 1e-2)))
        for k, name in enumerate(features):
            feats[:, s, k] = cand[name]
    out = dict(td=(d['td_offset_ms'] / 1e3).astype(np.float32), legs=d['leg_index'].astype(np.int64),
               feats=feats, trial=d['trial'].astype(str), dataset=d['dataset'].astype(str),
               walk=d['walk'].astype(str), diagonal=d['diagonal'].astype(str))
    if 'audio' in d:
        out['audio'] = d['audio'].astype(np.float32) / 32768.
        out['fs'] = int(d['fs'])
        out['post40'] = d['audio_post40_dbfs']
    out['pre'] = float(-t_ms[0]) / 1e3
    return out


def subset(d, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(mask) else v) for k, v in d.items()}


def to_dev(d, dev):
    return {k: torch.as_tensor(v, device=dev) for k, v in d.items() if k in ('td', 'legs', 'feats', 'audio')}


# ---------------------------------------------------------------- 학습
def psd_estimate(clips, seg, n_fft=2048):
    """σ²(f) 추정 (모델 noise0 정규화와 같음: E|X|² = σ²(f) Σw²). seg 길이 Hann, n_fft 로 0 채움."""
    w = np.hanning(seg)
    acc, cnt = 0., 0
    for c in clips:
        for s in range(0, len(c) - seg + 1, seg // 2):
            x = c[s:s + seg] - c[s:s + seg].mean()
            acc = acc + np.abs(np.fft.rfft(x * w, n_fft)) ** 2 / np.sum(w ** 2)
            cnt += 1
    return acc / cnt, cnt


def measured_noise(train_audio, pre, fs, motor_noise_path, cross=(700., 1400.)):
    """배경 소음 측정값: 1 kHz 위 = NORMAL_07 모터 소음 (stand+motor_off, outlier 제외),
    1 kHz 아래 = 보행 중 착지 전 구간 (-50 ~ -10 ms). 700~1400 Hz 에서 로그 주파수로 교차."""
    n0 = 0; n1 = int((pre - .01) * fs)
    walk_psd, n_walk = psd_estimate(train_audio[:, n0:n1], n1 - n0)
    with np.load(motor_noise_path) as z:
        chunks = z['audio'][~z['outlier']].astype(np.float64) / 32768.
    motor_psd, n_motor = psd_estimate(chunks, 2048)
    f = np.fft.rfftfreq(2048, 1 / fs)
    lf = np.log(np.maximum(f, 1.))
    w = np.clip((lf - np.log(cross[0])) / (np.log(cross[1]) - np.log(cross[0])), 0, 1)
    log_psd = (1 - w) * np.log(walk_psd + 1e-20) + w * np.log(motor_psd + 1e-20)
    info = dict(walk_pre_segments=n_walk, motor_noise_segments=n_motor,
                walk_pre_var_db=float(10 * np.log10(walk_psd.mean())),
                motor_var_db=float(10 * np.log10(motor_psd.mean())),
                combined_var_db=float(10 * np.log10(np.exp(log_psd).mean())))
    return log_psd.astype(np.float32), info


def init_noise(model, audio, dev, fixed_log_psd=None):
    """배경 소음: fixed_log_psd 가 있으면 그 값으로 고정 (학습 안 함). 없으면 학습 세트 파워 중앙값/ln2 로 시작해 학습.
    확률적 충격음 스펙트럼 S(f) 는 학습 세트 중앙값으로 시작."""
    with torch.no_grad():
        S = power_spec(torch.as_tensor(audio, device=dev), 2048, model.win2048)   # (B,1025,T)
        # 지수분포 중앙값 = ln2 × 평균 → 평균 파워로 환산
        med = S.permute(1, 0, 2).reshape(1025, -1).median(1).values / math.log(2) / model.win2048.pow(2).sum()
        model.stoch_log_psd.copy_(torch.log(med + 1e-20))
        if fixed_log_psd is None:
            model.noise0.copy_(torch.log(med + 1e-20))
        else:
            model.noise0.copy_(torch.as_tensor(fixed_log_psd, device=dev))
            model.noise0.requires_grad_(False)


def calibrate(model, batch, idx=None, feats=None):
    """모달 이득 크기를 실측 착지 구간(0~80 ms) 에너지에 맞춤."""
    with torch.no_grad():
        y = model(batch['td'], batch['legs'], idx, feats)['y']
        n0 = int(model.pre * model.fs); n1 = n0 + int(.08 * model.fs)
        x = batch['audio']
        target = (x[:, n0:n1].pow(2).mean() - x[:, :n0 - 480].pow(2).mean()).clamp_min(1e-12)
        model.gain.mul_(torch.sqrt(target / y[:, n0:n1].pow(2).mean().clamp_min(1e-20)))


def run_stage(model, data, dev, stage, steps, batch_size, lr, log_every=100, val=None):
    common = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith(('free_', 'feat_', 'log_freq'))]
    groups = [dict(params=common, lr=lr), dict(params=[model.log_freq], lr=lr * .2)]
    if stage == 'free':
        groups.append(dict(params=[model.free_log_amp, model.free_jitter], lr=lr * 5))
    else:
        groups.append(dict(params=[model.feat_w, model.feat_b], lr=lr * 2))
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=steps,
                                                pct_start=.05, anneal_strategy='cos')
    T = to_dev(data, dev)
    n = len(data['td'])
    gen = torch.Generator(device='cpu').manual_seed(0)
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randperm(n, generator=gen)[:batch_size].to(dev)
        feats = T['feats'][idx] if stage == 'feat' else None
        out = model(T['td'][idx], T['legs'][idx], idx if stage == 'free' else None, feats)
        is_div, lsd, sc = spectral_terms(out, T['audio'][idx], model)
        loss = is_div.mean() + 1e-3 * model.env_tail.pow(2).sum()
        if stage == 'feat':
            loss = loss + 1e-3 * model.feat_w.pow(2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step(); sched.step()
        if step % log_every == 0 or step == steps:
            row = dict(stage=stage, step=step, loss=loss.item(), IS=is_div.mean().item(), lsd=lsd.mean().item(), sc=sc.mean().item(),
                       sec=time.time() - t0)
            if val is not None:
                m = evaluate(model, val, dev, use_feats=True)
                row.update(val_IS=float(m['IS'].mean()), val_lsd=float(m['lsd'].mean()))
            hist.append(row)
            print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush=True)
    return hist


# ---------------------------------------------------------------- 평가
@torch.no_grad()
def evaluate(model, d, dev, use_feats=True, idx=None, noise_only=False, batch=64, draws=0):
    """스펙트럼 지표 + 결정적 파형 y + (draws>0) 한 번의 실현값 샘플 (draws, G, N)."""
    T = to_dev(d, dev)
    terms, ys, samples = [], [], [[] for _ in range(draws)]
    gens = [torch.Generator(device=dev).manual_seed(1000 + k) for k in range(draws)]
    for s in range(0, len(d['td']), batch):
        sl = slice(s, s + batch)
        out = None
        if not noise_only:
            out = model(T['td'][sl], T['legs'][sl], idx[sl] if idx is not None else None,
                        T['feats'][sl] if use_feats else None, jitter=idx is not None)
            ys.append(out['y'])
        for k in range(draws):
            if out is None:
                samples[k].append(model.sample_noise(len(T['td'][sl]), gens[k]))
            else:
                samples[k].append(model.sample(out, gens[k]))
        if 'audio' in T:
            terms.append(spectral_terms(out, T['audio'][sl], model))
    res = dict(y=torch.cat(ys).cpu().numpy() if ys else None)
    if draws:
        res['samples'] = np.stack([torch.cat(v).cpu().numpy() for v in samples])
    if terms:
        for k, v in zip(('IS', 'lsd', 'sc'), zip(*terms)):
            res[k] = torch.cat(v).cpu().numpy()
    return res


def dbfs_window(x, model, window=(0., .04)):
    n0 = int((model.pre + window[0]) * model.fs); n1 = int((model.pre + window[1]) * model.fs)
    seg = x[:, n0:n1] - x[:, n0:n1].mean(1, keepdims=True)
    return 10 * np.log10(np.mean(seg ** 2, 1) + 1e-20)


def predicted_post40(model, samples):
    """착지 후 0~40 ms 예측 레벨 [dBFS]: 실현값 샘플마다 dB → 평균 (실측과 같은 통계)."""
    return np.mean([dbfs_window(x, model) for x in samples], 0)


@torch.no_grad()
def calibrate_level(model, data, dev, draws=8, iters=14, span=2.0):
    """레벨 안정화: 학습 세트에서 착지 후 0~40 ms 예측 레벨의 평균 편향이 0 이 되도록
    모든 착지의 로그 충격 크기에 공통 상수 δ 를 더한다 (feat_b += δ, 파라미터 1개).

    IS 손실은 과소 예측 벌점이 커서 모델이 전반적으로 크게 예측하는 쪽으로 치우치고, 그 정도가 시드마다
    달라 (학습 편향 +0.5 ~ +2.3 dB) 검증 편향이 그대로 따라간다. 착지 간 상대 크기 순서는 바뀌지 않는다.
    실현값 샘플은 고정 시드(공통 난수)라 편향이 δ 에 대해 단조 → 이분법.
    """
    real = dbfs_window(data['audio'], model)
    base = model.feat_b.detach().clone()

    def bias(delta):
        model.feat_b.copy_(base + delta)
        return float(np.mean(predicted_post40(model, evaluate(model, data, dev, use_feats=True, draws=draws)['samples']) - real))
    before = bias(0.)
    lo, hi = -span, span
    for _ in range(iters):
        mid = (lo + hi) / 2
        if bias(mid) > 0:
            hi = mid
        else:
            lo = mid
    delta = (lo + hi) / 2
    after = bias(delta)
    return dict(delta_log_amp=float(delta), delta_db_impact=float(20 * delta / math.log(10)),
                train_bias_before_db=before, train_bias_after_db=after)


BANDS = [(50, 200), (200, 1000), (1000, 4000), (4000, 12000)]


@torch.no_grad()
def band_db(model, sig, dev, window=(0., .08), n_fft=1024):
    """착지 후 구간의 대역별 레벨 [dB]. 1024-STFT 파워를 구간 프레임·대역 bin 으로 합산."""
    win = getattr(model, f'win{n_fft}')
    out = []
    for s in range(0, len(sig), 64):
        S = power_spec(torch.as_tensor(sig[s:s + 64], device=dev), n_fft, win)
        times = torch.arange(S.shape[-1], device=dev) * (n_fft // 4) / model.fs - model.pre
        fm = (times >= window[0]) & (times < window[1])
        f = torch.arange(S.shape[1], device=dev) * model.fs / n_fft
        out.append(torch.stack([10 * torch.log10(S[:, (f >= lo) & (f < hi)][:, :, fm].sum((1, 2)) + 1e-20)
                                for lo, hi in BANDS], 1))
    return torch.cat(out).cpu().numpy()


def template_eval(train, val, dev, model):
    wins = {n: getattr(model, f'win{n}') for n in model.fft_sizes}
    with torch.no_grad():
        xt = torch.as_tensor(train['audio'], device=dev)
        temps = {n: power_spec(xt, n, wins[n]).mean(0, keepdim=True) for n in model.fft_sizes}
        r = template_terms(torch.as_tensor(val['audio'], device=dev), temps, fft_sizes=model.fft_sizes, windows=wins,
                           band_mats={n: getattr(model, f'band{n}') for n in model.fft_sizes})
    return {k: v.cpu().numpy() for k, v in zip(('IS', 'lsd', 'sc'), r)}


def corr(a, b):
    from scipy.stats import pearsonr, spearmanr
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


# ---------------------------------------------------------------- 그림·소리
@torch.no_grad()
def figures(out, model, val, y_val, s_val, post_pred, sim_table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs, pre = model.fs, model.pre
    (out / 'figures').mkdir(exist_ok=True)
    # 1. 검증 예시 스펙트로그램
    picks = np.linspace(0, len(val['td']) - 1, 4).astype(int)
    fig, ax = plt.subplots(3, 4, figsize=(18, 10))
    for c, j in enumerate(picks):
        x = val['audio'][j]; y = y_val[j]
        for r, (sig, name) in enumerate(((x, 'real'), (s_val[j], 'model sample (all parts)'), (y, 'model deterministic impact only'))):
            S = power_spec(torch.as_tensor(sig)[None], 512, torch.hann_window(512))[0].numpy()
            ext = [-pre * 1e3, (len(sig) / fs - pre) * 1e3, 0, fs / 2 / 1e3]
            ax[r, c].imshow(10 * np.log10(S + 1e-14), origin='lower', aspect='auto', extent=ext, vmin=-110, vmax=-40, cmap='magma')
            for s in range(2):
                ax[r, c].axvline(val['td'][j, s] * 1e3, color='c', ls=':', lw=.8)
            ax[r, c].set_ylim(0, 16)
            ax[r, c].set_title(f"{name}\n{val['trial'][j]} {val['diagonal'][j]}" if r == 0 else name, fontsize=8)
        ax[2, c].set_xlabel('ms from first touchdown')
    ax[0, 0].set_ylabel('kHz')
    fig.tight_layout(); fig.savefig(out / 'figures/val_spectrograms.png', dpi=90); plt.close(fig)

    # 2. 학습된 모달 응답 / 소음
    with torch.no_grad():
        h = model.modal_ir().cpu().numpy()
        f = np.fft.rfftfreq(8192, 1 / fs)
        n0 = np.exp(model.noise0.cpu().numpy()); n1 = np.exp(model.stoch_log_psd.cpu().numpy())
        fn = np.linspace(0, fs / 2, len(n0))
        env_t = np.linspace(-.02, .26, 300)
        env = model.stoch_envelope(torch.as_tensor(env_t, dtype=torch.float32, device=model.noise0.device)).cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, leg in enumerate(LEGS):
        ax[0].semilogx(f[1:], 20 * np.log10(np.abs(np.fft.rfft(h[i], 8192))[1:] + 1e-12), lw=.8, label=leg)
    ax[0].set_title('modal response per leg |H(f)|'); ax[0].set_xlabel('Hz'); ax[0].legend()
    ax[1].semilogx(fn[1:], 10 * np.log10(n0[1:]), label='background N0')
    if model.stochastic:
        g = np.exp(model.stoch_leg_gain.cpu().numpy())
        ax[1].semilogx(fn[1:], 10 * np.log10(n1[1:] * env.max() * g.mean()), label='stochastic impact S(f) at peak (A=1, mean leg)')
    ax[1].set_title('PSD (per-sample variance density)'); ax[1].legend(); ax[1].set_xlabel('Hz')
    ax[2].plot(env_t * 1e3, env); ax[2].set_title('stochastic impact envelope e(t - t_i)'); ax[2].set_xlabel('ms after impact')
    for x in ax:
        x.grid(alpha=.3, which='both')
    fig.tight_layout(); fig.savefig(out / 'figures/learned_components.png', dpi=90); plt.close(fig)

    # 3. 레벨 예측
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for w in np.unique(val['walk']):
        m = val['walk'] == w
        ax[0].scatter(val['post40'][m], post_pred[m], s=12, label=w)
    lim = [min(val['post40'].min(), post_pred.min()) - 1, max(val['post40'].max(), post_pred.max()) + 1]
    ax[0].plot(lim, lim, 'k--', lw=.8); ax[0].set_xlabel('real post-40ms [dBFS]'); ax[0].set_ylabel('predicted [dBFS]')
    ax[0].set_title('validation pairs'); ax[0].legend()
    if sim_table is not None and len(sim_table):
        ax[1].scatter(sim_table.real_post40_median, sim_table.sim_pred_post40_median)
        for r in sim_table.itertuples():
            ax[1].annotate(r.trial.replace('NORMAL_', ''), (r.real_post40_median, r.sim_pred_post40_median), fontsize=7)
        lo = min(sim_table.real_post40_median.min(), sim_table.sim_pred_post40_median.min()) - .5
        hi = max(sim_table.real_post40_median.max(), sim_table.sim_pred_post40_median.max()) + .5
        ax[1].plot([lo, hi], [lo, hi], 'k--', lw=.8)
        ax[1].set_xlabel('real median post-40ms [dBFS]'); ax[1].set_ylabel('model on SIM touchdowns, median')
        ax[1].set_title('per-recording: sim state -> predicted sound')
    for x in ax:
        x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'figures/level_prediction.png', dpi=90); plt.close(fig)

    # 4. 듣기용 wav (실측 / 모델+소음)
    (out / 'wav').mkdir(exist_ok=True)
    for c, j in enumerate(picks):
        for sig, name in ((val['audio'][j], 'real'), (s_val[j], 'model')):
            wavfile.write(out / f"wav/val{c}_{val['trial'][j]}_{name}.wav", fs,
                          np.clip(sig * 32768, -32768, 32767).astype(np.int16))


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=HERE / 'data/touchdown_w100_200hz')
    p.add_argument('--out', type=Path, required=True, help='실행 폴더 (예: runs/w100_200hz_kin_grf_s0)')
    p.add_argument('--modes', type=int, default=64)
    p.add_argument('--steps-free', type=int, default=3000)
    p.add_argument('--steps-feat', type=int, default=2000)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-stochastic', action='store_true', help='확률적 충격음 성분 없이 (원 DiffImpact 구성)')
    p.add_argument('--learn-noise', action='store_true', help='배경 소음을 학습 (기본: 측정값으로 고정)')
    p.add_argument('--motor-noise', type=Path, default=HERE / 'data/motor_noise/noise_chunks.npz')
    p.add_argument('--noise-data', type=Path, default=HERE / 'data/touchdown_w300_500hz/pairs_real.npz',
                   help='보행 중 착지 전 소음(1 kHz 이하) 추정용 세트. 짧은 창은 착지 전 구간이 짧아 300 ms 창 데이터를 씀')
    p.add_argument('--feature-set', choices=sorted(FEATURE_SETS), default='kin_grf')
    p.add_argument('--no-level-cal', action='store_true', help='학습 후 레벨 보정(학습 세트 평균 편향 0) 생략')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.out.mkdir(parents=True, exist_ok=True)

    global FEATURES
    FEATURES = FEATURE_SETS[a.feature_set]
    real = load_pairs(a.data / 'pairs_real.npz')
    is_val = np.array([(d, t) in VAL_TRIALS for d, t in zip(real['dataset'], real['trial'])])
    train, val = subset(real, ~is_val), subset(real, is_val)
    mu = train['feats'].reshape(-1, len(FEATURES)).mean(0)
    sd = train['feats'].reshape(-1, len(FEATURES)).std(0) + 1e-6
    for k, name in enumerate(FEATURES):
        if name in UNSCALED:
            mu[k], sd[k] = 0., 1.
    for d in (train, val):
        d['feats'] = ((d['feats'] - mu) / sd).astype(np.float32)
    for d in (train, val):                                           # 실측 레벨도 같은 함수로 (클립 0~40 ms)
        d['post40'] = dbfs_window(d['audio'], argparse.Namespace(pre=real['pre'], fs=real['fs']))
    print(f'train {len(train["td"])} pairs, val {len(val["td"])} pairs ({VAL_TRIALS})', flush=True)
    fs, N, pre = real['fs'], real['audio'].shape[1], real['pre']

    # ---- 단계 1: free
    clip_s = N / fs
    fft_sizes = (256, 512, 1024, 2048) if clip_s >= .2 else (128, 256, 512, 1024)
    stoch_span = (-.01, clip_s - pre)
    stoch_knots = int(round((stoch_span[1] - stoch_span[0]) / .01)) + 1          # 10 ms 간격
    model_cfg = dict(fs=fs, n_samples=N, pre=pre, n_modes=a.modes, n_free=len(train['td']), n_feat=len(FEATURES),
                     stochastic=not a.no_stochastic, fft_sizes=fft_sizes, stoch_span=stoch_span, stoch_knots=stoch_knots)
    print('model config:', model_cfg, flush=True)
    model = DiffImpact(seed=a.seed, **model_cfg).to(dev)
    noise_info = None
    fixed = None
    if not a.learn_noise:
        nd = load_pairs(a.noise_data, features=['cmd_vx'])
        nd_train = np.array([(d, t) not in VAL_TRIALS for d, t in zip(nd['dataset'], nd['trial'])])
        fixed, noise_info = measured_noise(nd['audio'][nd_train], nd['pre'], fs, a.motor_noise)
        noise_info['source'] = str(a.noise_data)
        print('fixed background noise:', noise_info, flush=True)
    init_noise(model, train['audio'], dev, fixed)
    T = to_dev(train, dev)
    calibrate(model, dict(td=T['td'][:64], legs=T['legs'][:64], audio=T['audio'][:64]),
              idx=torch.arange(64, device=dev))
    hist = run_stage(model, train, dev, 'free', a.steps_free, a.batch, a.lr)
    torch.save(model.state_dict(), a.out / 'model_free.pt')
    free_fit = evaluate(model, train, dev, use_feats=False, idx=torch.arange(len(train['td']), device=dev))

    # ---- 단계 2: feat (충격 크기 = 특징의 다리별 로그선형). 편향은 단계 1 크기의 다리별 평균으로 시작
    with torch.no_grad():
        la = model.free_log_amp.detach()
        legs = T['legs']
        for i in range(4):
            m = legs == i
            if m.any():
                model.feat_b[i] = la[m].mean()
    hist += run_stage(model, train, dev, 'feat', a.steps_feat, a.batch, a.lr * .3, val=val)
    torch.save(dict(state=model.state_dict(), feat_mu=mu, feat_sd=sd, features=FEATURES,
                    config=model_cfg), a.out / 'model_feat_raw.pt')
    level_cal = None
    if not a.no_level_cal:
        level_cal = calibrate_level(model, train, dev)
        print('level calibration:', level_cal, flush=True)
    torch.save(dict(state=model.state_dict(), feat_mu=mu, feat_sd=sd, features=FEATURES,
                    config=model_cfg, level_calibration=level_cal), a.out / 'model_feat.pt')
    pd.DataFrame(hist).to_csv(a.out / 'history.csv', index=False)

    # ---- 평가
    m_train = evaluate(model, train, dev, use_feats=True, draws=8)
    m_val = evaluate(model, val, dev, use_feats=True, draws=8)
    m_noise = evaluate(model, val, dev, noise_only=True, draws=8)
    m_tmpl = template_eval(train, val, dev, model)
    post_pred = predicted_post40(model, m_val['samples'])
    post_train = predicted_post40(model, m_train['samples'])
    mean_pred = np.full_like(val['post40'], train['post40'].mean())
    band_real = band_db(model, val['audio'], dev)
    band_pred = np.mean([band_db(model, x, dev) for x in m_val['samples']], 0)
    band_noise = np.mean([band_db(model, x, dev) for x in m_noise['samples']], 0)
    band_train_mean = band_db(model, train['audio'], dev).mean(0)
    summ = lambda m: {k: float(m[k].mean()) for k in ('IS', 'lsd', 'sc')}
    bname = [f'{lo}-{hi}Hz' for lo, hi in BANDS]
    metrics = dict(
        n_train=int(len(train['td'])), n_val=int(len(val['td'])), val_trials=VAL_TRIALS, features=FEATURES,
        feature_set=a.feature_set, data=str(a.data), window_ms=[-pre * 1e3, (clip_s - pre) * 1e3],
        level_calibration=level_cal,
        background_noise=dict(mode='learned' if a.learn_noise else 'fixed_measured', info=noise_info,
                              model_var_db=float(10 * np.log10(torch.exp(model.noise0.detach()).mean().item()))),
        spectral=dict(
            free_fit_train=summ(free_fit), feat_train=summ(m_train), feat_val=summ(m_val),
            noise_only_val=summ(m_noise), template_val=summ(m_tmpl),
            feat_val_by_walk={w: dict(IS=float(m_val['IS'][val['walk'] == w].mean()),
                                      template_IS=float(m_tmpl['IS'][val['walk'] == w].mean()),
                                      noise_only_IS=float(m_noise['IS'][val['walk'] == w].mean()))
                              for w in np.unique(val['walk'])}),
        band_0_80ms_val=dict(
            model_mae_db=dict(zip(bname, map(float, np.abs(band_pred - band_real).mean(0)))),
            model_bias_db=dict(zip(bname, map(float, (band_pred - band_real).mean(0)))),
            noise_only_mae_db=dict(zip(bname, map(float, np.abs(band_noise - band_real).mean(0)))),
            train_mean_mae_db=dict(zip(bname, map(float, np.abs(band_train_mean - band_real).mean(0)))),
            model_pearson=dict(zip(bname, [corr(band_pred[:, k], band_real[:, k])[0] for k in range(len(BANDS))]))),
        post40=dict(
            val_mae_db=float(np.abs(post_pred - val['post40']).mean()),
            val_mae_db_train_mean_baseline=float(np.abs(mean_pred - val['post40']).mean()),
            val_bias_db=float((post_pred - val['post40']).mean()),
            val_pearson_spearman=corr(post_pred, val['post40']),
            train_pearson_spearman=corr(post_train, train['post40'])),
        learned=dict(
            tau_ms={l: float(v) * 1e3 for l, v in zip(LEGS, model.tau().detach().cpu())},
            shift_ms={l: float(v) * 1e3 for l, v in zip(LEGS, model.shift().detach().cpu())},
            stoch_leg_gain={l: float(v) for l, v in zip(LEGS, model.stoch_leg_gain.detach().cpu())},
            free_jitter_ms_abs_median=float((8 * torch.tanh(model.free_jitter.detach()).abs()).median()),
            feat_bias={l: float(v) for l, v in zip(LEGS, model.feat_b.detach().cpu())},
            feat_weights={l: dict(zip(FEATURES, map(float, w))) for l, w in zip(LEGS, model.feat_w.detach().cpu())},
            modes_hz_sorted=sorted(float(f) for f in model.freqs().detach().cpu())))

    # ---- 시뮬 착지에 적용 (같은 특징 정규화)
    sim_table = None
    sim_path = a.data / 'pairs_sim.npz'
    if sim_path.exists():
        sim = load_pairs(sim_path)
        sim['feats'] = ((sim['feats'] - mu) / sd).astype(np.float32)
        sim_eval = evaluate(model, sim, dev, use_feats=True, draws=8)
        sim_post = predicted_post40(model, sim_eval['samples'])
        rows = []
        all_post_pred = np.r_[post_train, post_pred]
        all_real = dict(trial=np.r_[train['trial'], val['trial']], dataset=np.r_[train['dataset'], val['dataset']],
                        post=np.r_[train['post40'], val['post40']])
        for (ds, tr) in sorted(set(zip(sim['dataset'], sim['trial']))):
            ms = (sim['dataset'] == ds) & (sim['trial'] == tr)
            mr = (all_real['dataset'] == ds) & (all_real['trial'] == tr)
            if not mr.any():
                continue
            rows.append(dict(dataset=ds, trial=tr, split='val' if (ds, tr) in VAL_TRIALS else 'train',
                             walk=sim['walk'][ms][0], n_sim=int(ms.sum()), n_real=int(mr.sum()),
                             sim_pred_post40_median=float(np.median(sim_post[ms])),
                             real_model_post40_median=float(np.median(all_post_pred[mr])),
                             real_post40_median=float(np.median(all_real['post'][mr]))))
        sim_table = pd.DataFrame(rows)
        sim_table.to_csv(a.out / 'sim_vs_real_levels.csv', index=False, float_format='%.3f')
        metrics['sim_application'] = dict(
            per_recording_pearson_spearman=corr(sim_table.sim_pred_post40_median, sim_table.real_post40_median),
            median_bias_db=float((sim_table.sim_pred_post40_median - sim_table.real_post40_median).median()))
    (a.out / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n')
    figures(a.out, model, val, m_val['y'], m_val['samples'][0], post_pred, sim_table)
    print(json.dumps({k: metrics[k] for k in ('spectral', 'band_0_80ms_val', 'post40', 'learned')}, indent=1)[:6000])
    if 'sim_application' in metrics:
        print(json.dumps(metrics['sim_application']))


if __name__ == '__main__':
    main()
