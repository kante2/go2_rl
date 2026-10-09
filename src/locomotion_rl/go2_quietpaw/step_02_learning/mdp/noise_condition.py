"""사용자 quietness와 학습의 cost_limit을 분리한다."""

from dataclasses import dataclass

import torch


@dataclass
class NoiseConditionCfg:
  # 논문에 비용 정규화 기준이 없으므로 이 한계도 이식 구현의 조정 가능한 값이다.
  min_cost_limit: float = 0.02  # 가장 조용한 조건의 스텝 평균 비용 한계.
  max_cost_limit: float = 0.35  # 가장 느슨한 조건의 스텝 평균 비용 한계.
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
