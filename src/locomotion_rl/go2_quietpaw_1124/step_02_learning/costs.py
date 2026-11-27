"""논문 식 (1) 소음 비용의 설정값."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class NoiseCostCfg:
  # 식 (1)에 주어진 값: σF = 50 N, σv = 0.5 m²/s², λ1 = λ2 = 0.5.
  force_scale: float = 50.0  # σF [N]
  velocity_squared_scale: float = 0.5  # σv [m²/s²]
  force_weight: float = 0.5  # λ1
  velocity_weight: float = 0.5  # λ2
  # Fmax는 논문에 수치가 없다. 정규화 후에는 힘 항과 속도 항의 상대 비중만 바꾼다.
  force_threshold: float = 50.0  # Fmax [N]
  contact_force_threshold: float = 1.0  # 착지 판정 최소 접촉력 [N]
  # 인쇄된 식은 충격이 클수록 값이 작아진다. 본문·그림 2는 충격이 클수록 비용이 크다.
  # "increasing" = 1 − S/S₀ (기본), "printed" = S/S₀ (인쇄된 방향 그대로).
  orientation: Literal["increasing", "printed"] = "increasing"
  # 실제 실패 종료(넘어짐·불법 접촉) 뒤에는 매 스텝 이 비용이 영원히 이어진다고 보고
  # cost critic을 bootstrap한다. 일찍 넘어져서 비용 제약을 피하는 해를 막는다. (논문에 없음)
  failure_cost: float = 1.0

  def __post_init__(self):
    values = (self.force_scale, self.velocity_squared_scale, self.force_threshold,
              self.contact_force_threshold, self.force_weight, self.velocity_weight)
    if not all(math.isfinite(v) for v in values):
      raise ValueError("Noise cost parameters must be finite.")
    if min(self.force_scale, self.velocity_squared_scale) <= 0:
      raise ValueError("Noise cost scales must be positive.")
    if min(self.force_threshold, self.contact_force_threshold, self.force_weight, self.velocity_weight) < 0:
      raise ValueError("Thresholds and weights must be nonnegative.")
    if self.force_weight + self.velocity_weight <= 0:
      raise ValueError("At least one noise cost weight must be positive.")
    if self.orientation not in ("increasing", "printed"):
      raise ValueError("orientation must be 'increasing' or 'printed'.")
    if not 0 <= self.failure_cost <= 1:
      raise ValueError("failure_cost must be in [0, 1].")
