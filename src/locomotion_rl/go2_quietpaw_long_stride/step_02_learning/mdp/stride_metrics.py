"""동일 발의 연속 착지점 사이 수평 거리와 주기를 기록한다. 보상은 아니다."""

import torch


class StrideMeasurements:
  def __init__(self, num_envs, num_feet, device):
    shape = (num_envs, num_feet)
    self.previous_position = torch.zeros(*shape, 2, device=device)
    self.elapsed = torch.zeros(shape, device=device)
    self.has_previous = torch.zeros(shape, dtype=torch.bool, device=device)
    self.has_measurement = torch.zeros_like(self.has_previous)
    self.latest_distance = torch.zeros(shape, device=device)
    self.latest_period = torch.zeros(shape, device=device)
    self.distance_sum = torch.zeros(num_envs, device=device)
    self.period_sum = torch.zeros(num_envs, device=device)
    self.count = torch.zeros(num_envs, device=device)

  def begin_step(self):
    self.distance_sum.zero_()
    self.period_sum.zero_()
    self.count.zero_()

  def sample(self, positions, touchdown, command, dt):
    # 정지/제자리 회전을 제외한 직선 이동 구간의 진단값이다.
    active = ((command[:, :2].norm(dim=-1) > 0.1)
              & (command[:, 2].abs() < 0.2)).unsqueeze(-1)
    self.has_previous &= active
    self.has_measurement &= active
    self.elapsed += dt
    valid = touchdown & self.has_previous
    distance = (positions[..., :2] - self.previous_position).norm(dim=-1)
    self.distance_sum += (distance * valid).sum(-1)
    self.period_sum += (self.elapsed * valid).sum(-1)
    self.count += valid.sum(-1)
    self.latest_distance.copy_(torch.where(valid, distance, self.latest_distance))
    self.latest_period.copy_(torch.where(valid, self.elapsed, self.latest_period))
    self.has_measurement |= valid
    self.previous_position.copy_(torch.where(
      touchdown.unsqueeze(-1), positions[..., :2], self.previous_position
    ))
    self.elapsed.masked_fill_(touchdown, 0.0)
    self.has_previous |= touchdown & active

  def reset(self, env_ids=None):
    # reset 직전 스텝의 집계는 extras로 전달할 때까지 유지한다.
    self.has_previous[env_ids] = False
    self.has_measurement[env_ids] = False
    self.elapsed[env_ids] = 0.0

  def logs(self):
    # 발별 최근 완료 주기를 평균한다. 측정되지 않은 발은 제외한다.
    mask = self.has_measurement
    count = mask.sum().clamp_min(1)
    return {
      "LongStride/stride_length_m": (self.latest_distance * mask).sum() / count,
      "LongStride/stride_period_s": (self.latest_period * mask).sum() / count,
      "LongStride/measured_feet": mask.sum(),
    }
