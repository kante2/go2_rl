"""DiffImpact (Clarke et al., CoRL 2021) 구조를 Go2 트롯 착지 소리에 맞춘 미분 가능 합성 모델.

    소리(t) = [ Σ_i 펄스_i(t) * 모달응답_{leg_i}(t) ] * 공간응답(t)      ← 결정적 충격음 (원 DiffImpact)
            + Σ_i A_i · 확률적 충격음_{leg_i}(t - t_i)                  ← 착지가 일으키는 확률적 소리 (추가)
            + 배경소음(t)                                                ← 정상 소음 (원 DiffImpact)

  펄스_i       : 착지 i 의 접촉력. Hertz형 sin^1.5 펄스, 면적 = 충격 크기 A_i, 길이 τ_leg
                 시각 t_i = 착지 검출 시각 + 다리별 지연 shift_leg (+ 식별 단계에서는 세트별 미세조정)
  모달응답     : 감쇠 사인파 K개의 합. 주파수·감쇠율은 공유, 이득은 다리별 (마이크까지 경로가 다리마다 다름)
  공간응답     : δ + 학습 꼬리 (50 ms)
  확률적 충격음: 위상이 매번 다른 소리 (발 패드 미끄러짐, 기구 덜걱거림, 입각기 진동).
                 기대 파워 = A_i² · g_leg · S(f) · e(t - t_i).  e 는 착지 후 시간 포락선 (-10 ~ 250 ms).
                 충격 크기 A 를 결정적 성분과 공유 → 착지가 세면 둘 다 커짐 (식별성 유지).
  배경소음     : 정상 소음 N0(f).
  확률적 성분은 파형이 아니라 기대 파워로 스펙트로그램에 더하고, 손실은 파워의 지수분포 우도(IS 발산).

  충격 크기 A  : free = 세트·발마다 자유 파라미터 (식별, 원 DiffImpact)
                 feat = 착지 상태 특징의 다리별 로그선형 함수 (예측, 시뮬에도 적용 가능)

  stochastic=False 로 두면 원 DiffImpact 구성 (결정적 충격음 + 정상 소음) 과 같다.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PULSE_AREA = 0.5563706   # ∫_0^1 sin(πu)^1.5 du
TAU_MIN, TAU_SPAN = 0.3e-3, 19.7e-3
SHIFT_MAX = 0.015
JITTER_MAX = 0.008
FFT_SIZES = (256, 512, 1024, 2048)          # 300 ms 창 기본값. 짧은 창은 fft_sizes 인자로 지정
ALL_WINDOWS = (128, 256, 512, 1024, 2048)
STOCH_SPAN = (-0.01, 0.25)                  # 기본값. stoch_span 인자로 창 길이에 맞춤
BAND_RANGE = (40., 20000.)
BANDS_PER_OCTAVE = 6


def log_band_matrix(n_fft, fs, lo=BAND_RANGE[0], hi=BAND_RANGE[1], per_octave=BANDS_PER_OCTAVE):
    """STFT bin → 로그 간격 대역 합산 행렬 (bands, bins). bin 이 하나도 없는 대역은 이웃과 합침."""
    f = np.arange(n_fft // 2 + 1) * fs / n_fft
    edges = lo * 2 ** (np.arange(0, np.log2(hi / lo) * per_octave + 1) / per_octave)
    rows, start = [], lo
    for e in edges[1:]:
        m = (f >= start) & (f < e)
        if m.sum() >= 1:
            rows.append(m.astype(np.float32)); start = e
    return torch.tensor(np.stack(rows))


class DiffImpact(nn.Module):
    def __init__(self, fs=48000, n_samples=14400, pre=0.05, n_modes=64, n_legs=4, env_len=2400,
                 noise_bins=1025, stochastic=True, stoch_knots=27, n_free=0, n_feat=0, seed=0,
                 fft_sizes=FFT_SIZES, stoch_span=STOCH_SPAN):
        super().__init__()
        self.fs, self.N, self.pre = fs, n_samples, pre
        self.fft_sizes = tuple(fft_sizes)
        self.stoch_span = tuple(stoch_span)
        rng = np.random.default_rng(seed)
        f0 = np.exp(np.linspace(np.log(60.), np.log(16000.), n_modes) + rng.normal(0, .03, n_modes))
        d0 = 1. / np.exp(rng.uniform(np.log(.004), np.log(.08), n_modes))
        self.log_freq = nn.Parameter(torch.tensor(np.log(f0), dtype=torch.float32))
        self.log_decay = nn.Parameter(torch.tensor(np.log(d0), dtype=torch.float32))
        self.gain = nn.Parameter(torch.tensor(rng.normal(0, 1, (n_legs, n_modes)), dtype=torch.float32))
        self.tau_raw = nn.Parameter(torch.full((n_legs,), math.log(.137 / .863)))           # 3 ms
        self.shift_raw = nn.Parameter(torch.full((n_legs,), math.atanh(-.004 / SHIFT_MAX)))  # -4 ms
        self.env_tail = nn.Parameter(torch.zeros(env_len - 1))
        self.noise0 = nn.Parameter(torch.full((noise_bins,), -30.))
        self.stochastic = stochastic
        self.stoch_log_psd = nn.Parameter(torch.full((noise_bins,), -30.), requires_grad=stochastic)
        self.stoch_leg_gain = nn.Parameter(torch.zeros(n_legs), requires_grad=stochastic)
        knots = np.linspace(self.stoch_span[0], self.stoch_span[1], stoch_knots)
        e0 = np.where(knots < 0, .05, np.exp(-np.maximum(knots, 0) / .04))                   # 착지 후 감쇠로 시작
        self.stoch_env_raw = nn.Parameter(torch.tensor(np.log(np.expm1(np.maximum(e0, 1e-3))), dtype=torch.float32),
                                          requires_grad=stochastic)
        self.n_free, self.n_feat = n_free, n_feat
        if n_free:
            self.free_log_amp = nn.Parameter(torch.zeros(n_free, 2))
            self.free_jitter = nn.Parameter(torch.zeros(n_free, 2))
        if n_feat:
            self.feat_w = nn.Parameter(torch.zeros(n_legs, n_feat))
            self.feat_b = nn.Parameter(torch.zeros(n_legs))
        t = torch.arange(n_samples, dtype=torch.float32) / fs
        self.register_buffer('t_ir', t)
        self.register_buffer('t_clip', t - pre)
        self.nfft = 2 ** math.ceil(math.log2(2 * n_samples + env_len))
        for n in ALL_WINDOWS:
            self.register_buffer(f'win{n}', torch.hann_window(n))
            self.register_buffer(f'band{n}', log_band_matrix(n, fs), persistent=False)

    # ------------------------------------------------------------ 결정적 충격음
    def tau(self):
        return TAU_MIN + TAU_SPAN * torch.sigmoid(self.tau_raw)

    def shift(self):
        return SHIFT_MAX * torch.tanh(self.shift_raw)

    def freqs(self):
        return torch.exp(self.log_freq).clamp(20., .49 * self.fs)

    def decays(self):
        return torch.exp(self.log_decay)

    def modal_ir(self):
        basis = torch.exp(-self.decays()[:, None] * self.t_ir) * torch.sin(2 * math.pi * self.freqs()[:, None] * self.t_ir)
        return self.gain @ basis                                     # (legs, N)

    def env_ir(self):
        return torch.cat([torch.ones(1, device=self.env_tail.device), self.env_tail])

    def log_amp(self, legs, idx=None, feats=None):
        if feats is not None:
            return (self.feat_w[legs] * feats).sum(-1) + self.feat_b[legs]
        return self.free_log_amp[idx]

    def impact_times(self, td, legs, idx=None, jitter=True):
        t = td + self.shift()[legs]
        if jitter and idx is not None and self.n_free:
            t = t + JITTER_MAX * torch.tanh(self.free_jitter[idx])
        return t

    def pulses(self, times, legs, amps):
        tau = self.tau()[legs]                                       # (B,2)
        u = (self.t_clip - times[..., None]) / tau[..., None]        # (B,2,N)
        inside = (u > 0) & (u < 1)
        s = torch.sin(math.pi * u.clamp(0, 1)).clamp_min(1e-6) ** 1.5
        p = torch.where(inside, s, torch.zeros_like(s))
        return p * (amps / (tau * self.fs * PULSE_AREA))[..., None]

    def synth(self, times, legs, amps):
        P = torch.fft.rfft(self.pulses(times, legs, amps), n=self.nfft)
        H = torch.fft.rfft(self.modal_ir(), n=self.nfft)[legs]      # (B,2,F)
        E = torch.fft.rfft(self.env_ir(), n=self.nfft)
        return torch.fft.irfft((P * H).sum(1) * E, n=self.nfft)[:, :self.N]

    def forward(self, td, legs, idx=None, feats=None, jitter=True):
        """→ dict(y=결정적 파형 (B,N), times, legs, amps)  (확률적 성분은 spectral_terms/sample 에서 사용)"""
        amps = torch.exp(self.log_amp(legs, idx, feats))
        times = self.impact_times(td, legs, idx, jitter)
        return dict(y=self.synth(times, legs, amps), times=times, legs=legs, amps=amps)

    # ------------------------------------------------------------ 확률적 성분·소음
    def _interp(self, v, bins):
        pos = torch.linspace(0, len(v) - 1, bins, device=v.device)
        lo = pos.floor().long().clamp(max=len(v) - 2)
        w = pos - lo
        return v[lo] * (1 - w) + v[lo + 1] * w

    def stoch_envelope(self, rel):
        """착지 후 시각 rel [s] → e (0 밖은 0)."""
        k = len(self.stoch_env_raw)
        lo, hi = self.stoch_span
        pos = (rel - lo) / (hi - lo) * (k - 1)
        inside = (pos >= 0) & (pos <= k - 1)
        pos = pos.clamp(0, k - 1 - 1e-6)
        lo = pos.floor().long(); w = pos - lo
        e = self.stoch_env_raw[lo] * (1 - w) + self.stoch_env_raw[(lo + 1).clamp(max=k - 1)] * w
        return F.softplus(e) * inside

    def stoch_weight(self, out, times_grid):
        """Σ_i A_i² g_leg e(t - t_i) — (B, len(times_grid))"""
        rel = times_grid[None, None, :] - out['times'][..., None]
        w = (out['amps'] ** 2 * torch.exp(self.stoch_leg_gain)[out['legs']])[..., None]
        return (w * self.stoch_envelope(rel)).sum(1)

    def expected_power(self, n_fft, n_frames, out=None):
        """STFT 파워 기대값: 정상 소음 (bins,frames) + 확률적 충격음 (B,bins,frames).
        정규화: 분산 σ²(f) 소음 → E|X|² = σ²(f) Σw²."""
        hop = n_fft // 4
        bins = n_fft // 2 + 1
        winE = getattr(self, f'win{n_fft}').pow(2).sum()
        P = torch.exp(self._interp(self.noise0, bins))[None, :, None] * winE
        if self.stochastic and out is not None:
            times = torch.arange(n_frames, device=P.device) * hop / self.fs - self.pre
            S = torch.exp(self._interp(self.stoch_log_psd, bins))
            P = P + S[None, :, None] * self.stoch_weight(out, times)[:, None, :] * winE
        return P

    def _shaped_noise(self, n, log_psd, generator):
        bins = self.N // 2 + 1
        w = torch.randn(n, self.N, generator=generator, device=log_psd.device)
        return torch.fft.irfft(torch.fft.rfft(w) * torch.exp(self._interp(log_psd, bins)).sqrt(), n=self.N)

    def sample_noise(self, n, generator=None):
        """배경 소음 실현값 (n, N)."""
        return self._shaped_noise(n, self.noise0, generator)

    def sample(self, out, generator=None):
        """한 번의 실현값: 결정적 충격음 + 확률적 충격음 + 배경 소음 (B, N). 듣기·레벨 비교용."""
        x = out['y'] + self.sample_noise(len(out['y']), generator)
        if self.stochastic:
            s = self._shaped_noise(len(out['y']), self.stoch_log_psd, generator)
            x = x + s * self.stoch_weight(out, self.t_clip).sqrt()
        return x


# ---------------------------------------------------------------- 손실
def power_spec(x, n_fft, window):
    X = torch.stft(x, n_fft, hop_length=n_fft // 4, window=window, center=True, return_complex=True)
    return X.real ** 2 + X.imag ** 2                                 # (B,bins,frames)


def _terms(Sp, Sx, eps):
    """IS 발산 (지수분포 음의 로그우도 - 상수), 로그 스펙트럼 L1, spectral convergence — (B,) 값."""
    r = (Sx + eps) / (Sp + eps)
    is_div = (r - torch.log(r) - 1).mean((1, 2))
    lsd = torch.log(r).abs().mean((1, 2))
    sc = torch.linalg.norm((Sx.sqrt() - Sp.sqrt()).flatten(1), dim=1) / torch.linalg.norm(Sx.sqrt().flatten(1), dim=1)
    return is_div, lsd, sc


def spectral_terms(out, x, model, fft_sizes=None, bands=True):
    """해상도 평균 (IS, LSD, SC). out=None 이면 배경 소음만.

    bands=True (기본): 파워를 로그 간격 대역(40 Hz~20 kHz, 옥타브당 6개)으로 합산한 뒤 비교.
    선형 bin 으로 비교하면 bin 대부분이 소음이 지배하는 고주파라, 착지 소리 크기를 좌우하는
    50~1000 Hz 가 손실에서 거의 무시된다 (v3/v4 실험).

    확률적 성분을 기대 파워로 더하므로 IS(=파워의 지수분포 우도)가 맞는 손실이다.
    로그 L1 은 랜덤 소음의 중앙값(평균의 ln2배, -1.6 dB)을 맞추게 돼 레벨이 낮게 치우친다.
    """
    res = []
    for n in (fft_sizes or model.fft_sizes):
        win = getattr(model, f'win{n}')
        Sx = power_spec(x, n, win)
        Sp = model.expected_power(n, Sx.shape[-1], out)
        if out is not None:
            Sp = Sp + power_spec(out['y'], n, win)
        Sp = Sp.expand_as(Sx)
        if bands:
            M = getattr(model, f'band{n}')
            Sx = torch.einsum('kf,bft->bkt', M, Sx); Sp = torch.einsum('kf,bft->bkt', M, Sp)
        res.append(_terms(Sp, Sx, 1e-14 * win.pow(2).sum()))
    return tuple(torch.stack(v).mean(0) for v in zip(*res))


def template_terms(x, templates, fft_sizes=FFT_SIZES, windows=None, band_mats=None):
    """기준선: 학습 세트 평균 파워 스펙트로그램을 그대로 예측. band_mats 가 있으면 대역 합산 후 비교."""
    res = []
    for n in fft_sizes:
        Sx = power_spec(x, n, windows[n]); Sp = templates[n].expand_as(Sx)
        if band_mats is not None:
            Sx = torch.einsum('kf,bft->bkt', band_mats[n], Sx); Sp = torch.einsum('kf,bft->bkt', band_mats[n], Sp)
        res.append(_terms(Sp, Sx, 1e-14 * windows[n].pow(2).sum()))
    return tuple(torch.stack(v).mean(0) for v in zip(*res))
