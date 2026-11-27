"""식 (4): 환경 i마다 비용 한계 ε_i를 [0, 1]에 균등 배정하고 학습 내내 고정한다.

quietness q는 사용자 입력이다. ε = max − q·(max − min), 기본 범위에서는 ε = 1 − q.
"""

from dataclasses import dataclass

import torch


@dataclass
class NoiseConditionCfg:
  # 논문: ε_i는 [0, 1]에서 균등 보간. ε는 (1−γ)로 정규화한 할인 비용 합과 비교한다.
  min_cost_limit: float = 0.0  # 가장 엄격한 ε
  max_cost_limit: float = 1.0  # 가장 느슨한 ε
  quietness: float | None = None  # None: 환경별 균등 배정, 숫자: 모든 환경에 같은 조건.

  def __post_init__(self):
    if not 0 <= self.min_cost_limit < self.max_cost_limit <= 1:
      raise ValueError("Require 0 <= min_cost_limit < max_cost_limit <= 1.")
    if self.quietness is not None and not 0 <= self.quietness <= 1:
      raise ValueError("quietness must be in [0, 1].")


class NoiseCondition:
  def __init__(self, cfg: NoiseConditionCfg, num_envs: int, device: str):
    cfg.__post_init__()  # CLI 수정 이후에도 유효 범위를 검사한다.
    self.cfg = cfg
    # 환경이 하나라면 중간 조건을 사용한다. reset 때 재배정하지 않으므로
    # 환경별 PID multiplier와 조건의 대응이 rollout 전체에서 유지된다.
    self.quietness = (
      torch.linspace(0.0, 1.0, num_envs, device=device)
      if num_envs > 1 else torch.full((1,), 0.5, device=device)
    )
    if cfg.quietness is not None:
      self.quietness.fill_(cfg.quietness)

  @property
  def cost_limit(self):
    return self.cfg.max_cost_limit - self.quietness * (
      self.cfg.max_cost_limit - self.cfg.min_cost_limit
    )

  def set_quietness(self, value: float):
    if not 0 <= value <= 1:
      raise ValueError("quietness must be in [0, 1].")
    self.quietness.fill_(value)


def noise_condition(env):
  # Actor와 두 Critic이 받는 조건은 비용 한계 ε이다. 작을수록 엄격하다.
  return env.noise_condition.cost_limit.unsqueeze(-1)
