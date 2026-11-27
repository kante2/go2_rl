"""[go2_noise_aware 8d4270d에서 그대로 복사] DiffImpact 앙상블 소음 모델 (RL 착지 소음 추정, 순수 torch · GPU 배치). 충돌 물리 법칙만 사용한다.

착지 한 번(발 i)의 A-가중 0~80 ms 기대 에너지(배경 제외, 단위 FS², 미보정)를
  E_i = A_i² · K_다리(Δ_i, τ_i)
로 계산한다. A 는 충격량(접촉력 펄스 넓이), τ 는 접촉 시간(펄스 폭)이다.

  접촉 속도   v = max( −(foot_vz_pre + body_vz_pre), 0.001 )                [m/s]  (바닥 기준 발 하강 속도)
  충격 크기   A = sqrt( (m_다리 · (v/v_ref)^α)² + A0_다리² )                충격량-운동량 J = m v (1+e) → A ∝ v
                                                                         + 하중이 실리는 소리 A0 (에너지 합)
  접촉 시간   τ = clamp( τ0_다리 · (v/v_ref)^(−β), 0.3 ms, 20 ms )          Hertz: t_c = 2.87 (m²/(R E*² v))^(1/5)
              τ0_다리 = 0.3 ms + 19.7 ms · sigmoid(τraw_다리)
  기본 표 (diffimpact_ensemble.pt, 실측 녹화 31개 · 11/26 데이터 포함): α = 1, β = 0.2 이론값 고정. 다리별 m, τ0, A0 만 실측 녹음으로 학습했다
  (m 에는 유효 질량·반발계수·마이크 감도, τ0 에는 유효 질량·발끝 반지름·탄성계수가 묶여 있다).
  Δ_i 는 세트(대각 쌍) 첫 착지 기준 그 발의 착지 시각 [s] (첫 발 = 0). 모델 소리가 힘 센서 착지보다 몇 ms 먼저
  시작해 첫 발은 창(0~80 ms) 앞부분이 잘리므로 Δ 에 따라 K 가 다르다. K 표는 학습된 모델의 진동 모드·반향으로
  미리 계산한 것이다 (크기 1 펄스, 폭 τ, 시각 Δ). 확률적 충격음 성분은 없다 (원 DiffImpact 구성). 파형 합성은 하지 않는다.
  두 발 에너지는 더하므로, 두 발이 5 ms 안에 거의 동시에 닿으면 원본 합성(모드 간섭)보다 약 0.2 dB 낮게 나온다.

입력 x 는 착지 특징 12개 (실로봇 학습 데이터와 같은 정의 · 단위, touchdown_features.py 가 만든다).
계산에는 foot_vz_pre, body_vz_pre 두 개만 쓰고 클리핑하지 않는다 (물리 법칙으로 외삽).
나머지 특징은 학습 범위(1~99 백분위) 밖 정도를 진단값(ood, ood_frac)으로만 돌려준다.
비용(보상) 변환은 이 모듈에 없다. db_ref (학습 실측 착지의 발별 예측 dB 백분위) 는 비용 설계 참고용이다.

실제 음압 (잠정): db_spl = db + SPL_OFFSET_DB  [dBA SPL, 마이크 위치, 0~80 ms 평균]
  UMIK-1 #7208479 교정 파일 Sens Factor −0.343 dB (94 dB SPL, 입력 볼륨 최대 = 디지털 +24 dB 기준) 에서
  C = 94 − (−0.343 − 24) − 0.3 (주파수 응답, A-가중 착지 소리) ≈ 118.0 dB.
  제조사 사양 (최대 133 dB SPL − 내부 게인 18 dB + 3.01) 과 일치. 녹음 조건: ALSA Mic Capture Volume 127 (0 dB),
  내부 게인 18 dB (기본) 가정. 교정기 미사용 → 불확실성 약 ±1~3 dB. 학습(상대 비교)에는 영향 없음.
"""

from pathlib import Path

import torch

DEFAULT_TABLE = Path(__file__).with_name("diffimpact_ensemble.pt")
DB_PER_NEPER = 10.0 / 2.302585092994046  # 10/ln(10)
OOD_FEATURES = ("foot_vx_pre", "foot_vy_pre", "foot_vz_pre", "body_vz_pre")
SPL_OFFSET_DB = 118.0  # 잠정 마이크 보정 상수: dB re FS² → dBA SPL (위 설명, ±1~3 dB)


class DiffImpactNoiseModel:
  def __init__(self, path: str | Path = DEFAULT_TABLE, device: str = "cpu",
               ood_features: tuple[str, ...] = OOD_FEATURES):
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("amp_mode") != "phys":
      raise ValueError(f"{path}: 물리 법칙 표(amp_mode='phys')만 지원한다.")
    self.amp_mode = "phys"
    self.features: list[str] = list(blob["features"])
    self.ood_mask = torch.tensor([f in ood_features for f in self.features], dtype=torch.float32, device=device)
    self.db_ref: dict[str, float] = dict(blob["db_ref"])
    self.meta = blob["meta"]
    self.device = device
    ens = blob["ensemble"]

    def stack(key):
      return torch.stack([m[key] for m in ens]).to(device)

    self.mu, self.sd, self.lo, self.hi = stack("mu"), stack("sd"), stack("lo"), stack("hi")  # (M, F), 진단용
    self.log_m, self.tau_raw = stack("log_m"), stack("tau_raw")  # (M, 4)
    self.alpha, self.beta, self.logA0 = stack("alpha"), stack("beta"), stack("logA0")  # (M, 4)
    v_idx = {tuple(m["v_idx"]) for m in ens}
    v_ref = {float(m["v_ref"]) for m in ens}
    if len(v_idx) != 1 or len(v_ref) != 1:
      raise ValueError("Ensemble members must share v_idx and v_ref.")
    self.v_idx, self.v_ref = v_idx.pop(), v_ref.pop()
    self.tau_min, self.tau_span = float(blob["tau_min"]), float(blob["tau_span"])
    self.log_tau_grid = torch.log(blob["tau_grid"].to(device))  # (T,) 등비 격자
    self.offset_grid = blob["offset_grid"].to(device)  # (O,) 등간격 [s]
    self.logK = blob["logK"].to(device)  # (M, 4, O, T)
    self.n_models = self.log_m.shape[0]

  def feature_index(self, name: str) -> int:
    return self.features.index(name)

  @staticmethod
  def _pos(v, g):
    p = ((v - g[0]) / (g[-1] - g[0]) * (len(g) - 1)).clamp(0, len(g) - 1)
    i = p.floor().long().clamp(max=len(g) - 2)  # 격자 끝(float32 반올림 포함)에서도 i+1 이 범위 안
    return i, p - i

  def _log_k(self, tau: torch.Tensor, legs: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """ln K_다리(Δ, τ) — (Δ, ln τ) 격자 쌍선형 보간. tau (M, N), legs/offset (N,) → (M, N)."""
    it, ft = self._pos(torch.log(tau), self.log_tau_grid)  # (M, N)
    io, fo = self._pos(offset, self.offset_grid)  # (N,)
    m_idx = torch.arange(self.n_models, device=tau.device)[:, None]
    L = legs[None, :]

    def at(o, t):
      return self.logK[m_idx, L, o[None, :].expand_as(t), t]

    k00, k01 = at(io, it), at(io, it + 1)
    k10, k11 = at(io + 1, it), at(io + 1, it + 1)
    fo = fo[None, :]
    return (k00 * (1 - ft) + k01 * ft) * (1 - fo) + (k10 * (1 - ft) + k11 * ft) * fo

  @torch.no_grad()
  def foot_energy(self, x: torch.Tensor, legs: torch.Tensor, offset: torch.Tensor | None = None,
                  mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """착지 N 개. x: (N, F) 원 단위 특징, legs: (N,) 0=FR 1=FL 2=RR 3=RL,
    offset: (N,) 세트 첫 착지 기준 이 발의 착지 시각 [s] (None = 0, 첫 발).
    mask: (N,) bool — 감시용 ood_frac 을 이 착지들로만 계산 (None = 전체). 에너지는 전체 계산.

    반환: energy (N,) 앙상블 선형 평균, db (N,), db_std (N,) 모델 간 dB 표준편차, tau_ms (N,), contact_speed (N,),
          ood (N,) OOD_FEATURES 의 학습 범위 밖 거리 합 [σ] (진단값), ood_frac (F,) 특징별 범위 밖 비율 (진단값).
    """
    x = x.to(self.device, torch.float32)
    legs = legs.to(self.device, torch.long)
    offset = (torch.zeros(len(x), device=self.device) if offset is None
              else offset.to(self.device, torch.float32).clamp(0, float(self.offset_grid[-1])))
    m_idx = torch.arange(self.n_models, device=x.device)[:, None]
    L = legs[None]
    i, j = self.v_idx
    v = (-(x[:, i] + x[:, j])).clamp_min(1e-3)  # (N,) 접촉 속도 [m/s], 클리핑 없음
    lv = torch.log(v / self.v_ref)[None]  # (1, N)
    log_imp = self.log_m[m_idx, L] + self.alpha[m_idx, L] * lv  # ln m + α ln(v/v_ref)
    log_a = 0.5 * torch.logaddexp(2 * log_imp, 2 * self.logA0[m_idx, L])  # ln sqrt(A_충격² + A0²)
    tau0 = self.tau_min + self.tau_span * torch.sigmoid(self.tau_raw[m_idx, L])
    tau = (tau0 * torch.exp(-self.beta[m_idx, L] * lv)).clamp(self.tau_min, self.tau_min + self.tau_span)
    log_e = 2 * log_a + self._log_k(tau, legs, offset)  # (M, N)
    db_m = DB_PER_NEPER * log_e
    energy = torch.exp(log_e).mean(0)
    xr = torch.maximum(torch.minimum(x[None], self.hi[:, None]), self.lo[:, None])  # 학습 범위 (진단용)
    outside = (x[None] - xr).abs() / self.sd[:, None]  # (M, N, F)
    return dict(
      energy=energy,
      db=DB_PER_NEPER * torch.log(energy.clamp_min(1e-30)),
      db_std=db_m.std(0) if self.n_models > 1 else torch.zeros_like(energy),
      ood=(outside.mean(0) * self.ood_mask).sum(-1),
      ood_frac=self._ood_frac(outside.mean(0) > 0, mask),
      tau_ms=tau.mean(0) * 1e3,
      contact_speed=v,
      db_spl=DB_PER_NEPER * torch.log(energy.clamp_min(1e-30)) + SPL_OFFSET_DB,  # 잠정 dBA SPL
    )


def _ood_frac_impl(out_mask, mask):
  if mask is None:
    return out_mask.float().mean(0) if len(out_mask) else torch.zeros(out_mask.shape[-1], device=out_mask.device)
  m = mask.float()[:, None]
  return (out_mask.float() * m).sum(0) / m.sum().clamp_min(1)


DiffImpactNoiseModel._ood_frac = staticmethod(_ood_frac_impl)
