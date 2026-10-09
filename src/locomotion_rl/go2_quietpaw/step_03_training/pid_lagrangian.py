"""환경별 평균 비용 위반으로 Lagrange multiplier를 조절한다."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PidLagrangianCfg:
  # 논문에 PID 계수가 없어 시작값을 명시한다. 비용 척도에 맞춰 조정할 수 있다.
  kp: float = 1.0
  ki: float = 0.05
  kd: float = 0.1
  smoothing: float = 0.95  # rollout 평균 비용의 EMA. 0이면 평활화하지 않는다.
  initial_multiplier: float = 0.0
  max_multiplier: float = 100.0


class PidLagrangian(nn.Module):
  def __init__(self, num_envs: int, cfg: PidLagrangianCfg, device: str):
    super().__init__()
    if min(cfg.kp, cfg.ki, cfg.kd, cfg.initial_multiplier) < 0:
      raise ValueError("PID gains and initial multiplier must be nonnegative.")
    if not 0 <= cfg.smoothing < 1 or cfg.max_multiplier <= 0:
      raise ValueError("Require smoothing in [0, 1) and positive max_multiplier.")
    if cfg.initial_multiplier > cfg.max_multiplier:
      raise ValueError("initial_multiplier exceeds max_multiplier.")
    self.cfg = cfg
    for name in ("integral", "smoothed_cost", "previous_cost", "multipliers"):
      self.register_buffer(name, torch.zeros(num_envs, device=device))
    self.register_buffer("initialized", torch.tensor(False, device=device))
    self.integral.fill_(cfg.initial_multiplier)
    self.multipliers.fill_(cfg.initial_multiplier)

  @torch.no_grad()
  def update(self, mean_cost, cost_limit):
    # 두 입력 모두 [B]. 누적 return과 스텝 평균 한계를 섞지 않는다.
    if mean_cost.shape != self.multipliers.shape or cost_limit.shape != mean_cost.shape:
      raise ValueError("PID expects one mean cost and limit per environment.")
    if not torch.isfinite(mean_cost).all() or not torch.isfinite(cost_limit).all():
      raise ValueError("PID received non-finite costs or limits.")
    if not self.initialized:
      self.smoothed_cost.copy_(mean_cost)
      self.previous_cost.copy_(mean_cost)
      self.initialized.fill_(True)
    else:
      self.smoothed_cost.lerp_(mean_cost, 1.0 - self.cfg.smoothing)
    error = self.smoothed_cost - cost_limit
    self.integral.add_(self.cfg.ki * error).clamp_(0.0, self.cfg.max_multiplier)
    # 비용이 증가할 때만 D항을 더하는 PID Lagrangian 변형이다.
    derivative = (self.smoothed_cost - self.previous_cost).clamp_min(0.0)
    self.multipliers.copy_(
      (self.cfg.kp * error + self.integral + self.cfg.kd * derivative)
      .clamp(0.0, self.cfg.max_multiplier)
    )
    self.previous_cost.copy_(self.smoothed_cost)
    return self.multipliers
