"""논문 식 (1): 접촉력과 착지 속도로 만든 소음 비용. 실제 dB 예측 모델은 아니다."""

import math

import torch

from ..costs import NoiseCostCfg


def noise_cost(peak_forces: torch.Tensor, impact_speed_squared: torch.Tensor, cfg: NoiseCostCfg) -> torch.Tensor:
  """발별 [B, 4] 물리량 → 환경별 [B] 정규화 비용 ∈ [0, 1].

  식 (1)  S = Σ_i λ1·exp((Fmax − ‖F_i‖)/σF) + λ2·exp(−v_i²/σv)
  S는 힘과 속도가 0일 때 최대 S₀ = 4(λ1·e^{Fmax/σF} + λ2)이다. 논문은 [0, 1] 정규화만 밝히므로
  S/S₀로 정규화한다. 기본(increasing)은 1 − S/S₀ 이며 그림 2처럼 충격이 클수록 커진다.
  공통 인자 e^{Fmax/σF}로 나누어 큰 지수의 overflow를 피한다.
  """
  w_force = cfg.force_weight
  w_vel = cfg.velocity_weight * math.exp(-cfg.force_threshold / cfg.force_scale)
  force_term = torch.exp(-peak_forces.clamp_min(0.0) / cfg.force_scale)  # = e^{(Fmax−F)/σF} / e^{Fmax/σF}
  vel_term = torch.exp(-impact_speed_squared.clamp_min(0.0) / cfg.velocity_squared_scale)
  ratio = (w_force * force_term + w_vel * vel_term).mean(-1) / (w_force + w_vel)  # S/S₀
  cost = 1.0 - ratio if cfg.orientation == "increasing" else ratio
  return cost.clamp(0.0, 1.0)


class LandingMeasurements:
  """정책 한 스텝(물리 substep 여러 번) 안의 발별 최대 접촉력과 착지 직전 하강 속도²."""

  def __init__(self, num_envs: int, num_feet: int, device: str):
    shape = (num_envs, num_feet)
    self.in_contact = torch.zeros(shape, dtype=torch.bool, device=device)
    self.peak_forces = torch.zeros(shape, device=device)
    self.impact_speed_squared = torch.zeros(shape, device=device)
    self.landing_count = torch.zeros(shape, device=device)

  def begin_step(self):
    # 접촉 상태는 스텝 경계를 넘어 유지한다. 지속 접지를 재착지로 세지 않기 위해서다.
    self.peak_forces.zero_()
    self.impact_speed_squared.zero_()
    self.landing_count.zero_()

  def sample(self, force, pre_step_vertical_velocity, threshold: float):
    """force [B, 4, 3] (월드), pre_step_vertical_velocity [B, 4]: substep 시작 시 발 월드 vz."""
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    contact = magnitude > threshold
    touchdown = contact & ~self.in_contact
    impact_squared = pre_step_vertical_velocity.clamp_max(0.0).square()  # 하강 성분만
    self.peak_forces.copy_(torch.maximum(self.peak_forces, magnitude))
    # 한 스텝에 여러 번 착지하면 가장 큰 속도²를 쓴다.
    self.impact_speed_squared.copy_(torch.maximum(self.impact_speed_squared, impact_squared * touchdown))
    self.landing_count.add_(touchdown)
    self.in_contact.copy_(contact)

  def reset(self, env_ids):
    # reset 직전 스텝의 측정값은 extras로 넘길 때까지 유지한다.
    self.in_contact[env_ids] = False
