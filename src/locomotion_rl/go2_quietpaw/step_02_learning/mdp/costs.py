"""물리량에서 계산하는 소음 대리 비용. 실제 음압(dB)을 예측하는 모델은 아니다."""

import torch

from ..costs import NoiseCostCfg


def noise_cost(
  peak_forces: torch.Tensor,
  impact_speed_squared: torch.Tensor,
  cfg: NoiseCostCfg,
) -> torch.Tensor:
  """발별 [B, 4] 물리량을 환경별 [B] 비용으로 변환한다."""
  # PDF 식 (1)은 힘·속도가 클수록 감소한다. 여기서는 본문 의도에 맞는
  # 1-exp(-x)를 사용한다. 이 보완식과 threshold는 원 논문의 정확한 재현이 아니다.
  excess_force = (peak_forces - cfg.force_threshold).clamp_min(0.0)
  force_cost = -torch.expm1(-excess_force / cfg.force_scale)
  velocity_cost = -torch.expm1(
    -impact_speed_squared.clamp_min(0.0) / cfg.velocity_squared_scale
  )
  # 네 발 평균을 사용하므로 항상 [0, 1]. dt를 곱하지 않는 스텝당 비용이다.
  return (cfg.force_weight * force_cost + cfg.velocity_weight * velocity_cost).mean(-1)


class LandingMeasurements:
  """정책 한 스텝 안의 접촉력 peak와 착지 직전 하강 속도를 모은다."""

  def __init__(self, num_envs: int, num_feet: int, device: str):
    shape = (num_envs, num_feet)
    self.in_contact = torch.zeros(shape, dtype=torch.bool, device=device)
    self.peak_forces = torch.zeros(shape, device=device)
    self.impact_speed_squared = torch.zeros(shape, device=device)
    self.landing_count = torch.zeros(shape, device=device)

  def begin_step(self):
    # 접촉 상태는 스텝 경계를 넘어 유지해야 지속 접지를 재착지로 세지 않는다.
    self.peak_forces.zero_()
    self.impact_speed_squared.zero_()
    self.landing_count.zero_()

  def sample(self, force, pre_step_vertical_velocity, threshold: float):
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    contact = magnitude > threshold
    touchdown = contact & ~self.in_contact
    # 월드 z의 하강 성분을 사용한다. 험지의 정확한 접촉 법선 속도는 아니다.
    impact_squared = pre_step_vertical_velocity.clamp_max(0.0).square()
    self.peak_forces.copy_(torch.maximum(self.peak_forces, magnitude))
    self.impact_speed_squared.copy_(torch.maximum(
      self.impact_speed_squared, impact_squared * touchdown
    ))
    self.landing_count.add_(touchdown)
    self.in_contact.copy_(contact)

  def reset(self, env_ids):
    # reset 직전 스텝의 측정값은 extras로 전달할 때까지 유지한다.
    self.in_contact[env_ids] = False
