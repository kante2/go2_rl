"""DiffImpact 앙상블 소음 모델 (RL 착지 소음 추정, 순수 torch · GPU 배치).

착지 한 번(발 i)의 A-가중 0~80 ms 기대 에너지(배경 제외, 단위 FS², 미보정)를
  E_i = A_i² · K_다리(Δ_i, τ_i)
로 계산한다. Δ_i 는 세트(대각 쌍) 첫 착지 기준 그 발의 착지 시각 [s] (첫 발 = 0). 모델 소리가 힘 센서
착지보다 몇 ms 먼저 시작해 첫 발은 창(0~80 ms) 앞부분이 잘리므로 Δ 에 따라 K 가 다르다. 표(diffimpact_ensemble.pt)는 학습된 모델에서 내보낸 것이다. 파형 합성은 하지 않는다.

입력 특징 x (착지 순간, 실로봇 학습 데이터와 같은 정의 · 단위):
  foot_v{x,y,z}_pre : 착지 직전 10 ms 평균, 몸 기준 발 속도(중력 정렬 좌표) [m/s]
  foot_{x,y,z}      : 착지 순간 몸체 좌표 발 위치 [m]
  cmd_{vx,vy,wz}    : 속도 명령
  body_vz_pre       : 착지 직전 10 ms 평균 몸통 수직 속도 [m/s]
  log_grf_impulse20 : ln(착지 후 20 ms 수직 접촉력 충격량 [N·s]) — 착지 1 스텝 뒤 확정
  is_second         : 대각 짝 발이 먼저 닿은 경우 1
특징은 학습 범위(기본 1~99 백분위)로 클리핑해 넣고, 벗어난 거리(σ 단위)는 ood 로 따로 돌려준다 (진단값).
예외: QUIET_EXTRAPOLATE_FEATURES (발·몸통 수직 속도) 는 **조용한 쪽만 클리핑하지 않고 외삽**한다.
  부호 제약으로 학습돼 범위 밖에서도 "느릴수록 조용" 방향이 유지되므로, 학습 데이터보다 조용한 착지도
  계속 낮게 예측한다 (클리핑하면 데이터의 가장 조용한 착지에서 예측이 멈춰 더 조용한 보행을 보상하지 못함).
  시끄러운 쪽은 그대로 클리핑한다. 조용한 방향은 앙상블 가중치 부호로 정한다 (모델·다리 모두 같은 부호여야 함).
  외삽 폭은 물리적으로 짧다: 발 vz 는 0 을 넘을 수 없다 (학습 범위 끝 −0.13 m/s → 0 에서 약 −0.8 dB).
  ood/ood_frac 은 외삽 구간도 학습 범위 밖으로 센다.
ood 는 OOD_FEATURES (착지 동작 = 속도) 만 합산한다. 자세·명령 특징은 시뮬-실제 차이가 원래 크다
(같은 정책에서 MuJoCo 착지 발 높이가 실로봇보다 ~3 cm 낮음). 특징별 범위 밖 비율은 ood_frac 으로 모두 돌려준다.
비용(보상) 변환은 이 모듈에 없다. db_ref (학습 실측 착지의 발별 예측 dB 백분위) 는 비용 설계 참고용으로 남긴다.
"""

from pathlib import Path

import torch

DEFAULT_TABLE = Path(__file__).with_name("diffimpact_ensemble.pt")
DB_PER_NEPER = 10.0 / 2.302585092994046  # 10/ln(10)
OOD_FEATURES = ("foot_vx_pre", "foot_vy_pre", "foot_vz_pre", "body_vz_pre")
QUIET_EXTRAPOLATE_FEATURES = ("foot_vz_pre", "body_vz_pre")


class DiffImpactNoiseModel:
  def __init__(self, path: str | Path = DEFAULT_TABLE, device: str = "cpu",
               ood_features: tuple[str, ...] = OOD_FEATURES,
               extrapolate_quiet: tuple[str, ...] = QUIET_EXTRAPOLATE_FEATURES):
    blob = torch.load(path, map_location=device, weights_only=False)
    self.features: list[str] = list(blob["features"])
    self.ood_mask = torch.tensor([f in ood_features for f in self.features], dtype=torch.float32, device=device)
    self.db_ref: dict[str, float] = dict(blob["db_ref"])
    self.meta = blob["meta"]
    self.device = device
    ens = blob["ensemble"]

    def stack(key):
      return torch.stack([m[key] for m in ens]).to(device)

    self.mu, self.sd, self.lo, self.hi = stack("mu"), stack("sd"), stack("lo"), stack("hi")  # (M, F)
    self.w, self.b = stack("feat_w"), stack("feat_b")  # (M, 4, F), (M, 4)
    self.tau_raw = stack("tau_raw")  # (M, 4)
    self.tau_feat = [m["tau_feat"] for m in ens]
    self.tau_w = torch.stack([
      m["tau_w"] if m["tau_w"] is not None else torch.zeros(4) for m in ens
    ]).to(device)  # (M, 4)
    if len(set(self.tau_feat)) != 1:
      raise ValueError("Ensemble members must share tau_feat.")
    self.tf = self.tau_feat[0]
    self.tau_min, self.tau_span = float(blob["tau_min"]), float(blob["tau_span"])
    self.log_tau_grid = torch.log(blob["tau_grid"].to(device))  # (T,) 등비 격자
    self.offset_grid = blob["offset_grid"].to(device)  # (O,) 등간격 [s]
    self.logK = blob["logK"].to(device)  # (M, 4, O, T)
    self.n_models = self.w.shape[0]

    # 클리핑 경계: 기본은 학습 범위 (lo, hi). 조용한 쪽 외삽 특징은 그 쪽 경계를 ±inf 로.
    self.clip_lo, self.clip_hi = self.lo.clone(), self.hi.clone()
    self.quiet_direction: dict[str, int] = {}  # +1: 값이 클수록 조용, -1: 작을수록 조용
    for name in extrapolate_quiet:
      if name not in self.features:
        continue
      k = self.features.index(name)
      wk = self.w[..., k]
      if torch.all(wk <= 0) and torch.any(wk < 0):
        self.clip_hi[:, k] = float("inf")
        self.quiet_direction[name] = +1
      elif torch.all(wk >= 0) and torch.any(wk > 0):
        self.clip_lo[:, k] = float("-inf")
        self.quiet_direction[name] = -1
      elif torch.all(wk == 0):
        continue  # 효과 없음
      else:
        raise ValueError(f"{name}: ensemble weights have mixed signs; cannot extrapolate one side.")
      if k == self.tf:  # τ 도 같은 특징을 쓰면 τ 방향도 조용한 쪽이어야 함 (τ 는 sigmoid 로 유계)
        if not (torch.all(self.tau_w * self.quiet_direction[name] >= 0)):
          raise ValueError(f"{name}: tau direction disagrees with amplitude direction.")

  def feature_index(self, name: str) -> int:
    return self.features.index(name)

  @staticmethod
  def _pos(v, g):
    p = ((v - g[0]) / (g[-1] - g[0]) * (len(g) - 1)).clamp(0, len(g) - 1 - 1e-6)
    i = p.floor().long()
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

    반환: energy (N,) 앙상블 선형 평균, db (N,), db_std (N,) 모델 간 dB 표준편차,
          ood (N,) OOD_FEATURES 의 범위 밖 거리 합 [σ] (진단값), ood_frac (F,) 특징별 범위 밖 비율 (전체 특징, 감시용).
    """
    x = x.to(self.device, torch.float32)
    legs = legs.to(self.device, torch.long)
    offset = (torch.zeros(len(x), device=self.device) if offset is None
              else offset.to(self.device, torch.float32).clamp(0, float(self.offset_grid[-1])))
    xc = torch.maximum(torch.minimum(x[None], self.clip_hi[:, None]), self.clip_lo[:, None])  # (M, N, F)
    z = (xc - self.mu[:, None]) / self.sd[:, None]
    m_idx = torch.arange(self.n_models, device=x.device)[:, None]
    log_a = (self.w[m_idx, legs[None]] * z).sum(-1) + self.b[m_idx, legs[None]]  # (M, N)
    zt = self.tau_raw[m_idx, legs[None]]
    if self.tf >= 0:
      zt = zt + self.tau_w[m_idx, legs[None]] * z[..., self.tf]
    tau = self.tau_min + self.tau_span * torch.sigmoid(zt)
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
    )


def _ood_frac_impl(out_mask, mask):
  if mask is None:
    return out_mask.float().mean(0) if len(out_mask) else torch.zeros(out_mask.shape[-1], device=out_mask.device)
  m = mask.float()[:, None]
  return (out_mask.float() * m).sum(0) / m.sum().clamp_min(1)


DiffImpactNoiseModel._ood_frac = staticmethod(_ood_frac_impl)
