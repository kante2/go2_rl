"""착지 진단값(최대 접촉력, 착지 직전 하강 속도²). 비용은 소음 모델(noise_model/)이 만든다."""

import torch


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
