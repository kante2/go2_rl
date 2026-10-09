"""소음 비용 설정. 논문의 불명확한 부호·정규화 대신 명시적인 증가 함수를 사용한다."""

from dataclasses import dataclass


@dataclass
class NoiseCostCfg:
  # 이 값들은 MuJoCo 이식 구현의 시작값이며, 논문에서 확정된 파라미터가 아니다.
  force_threshold: float = 50.0  # 이 힘을 초과하는 접촉에 비용 [N].
  force_scale: float = 50.0  # 초과 접촉력의 지수 함수 척도 [N].
  velocity_squared_scale: float = 0.5  # 착지 수직 속도 제곱의 척도 [m²/s²].
  force_weight: float = 0.5
  velocity_weight: float = 0.5
  contact_force_threshold: float = 1.0  # 접촉 전환 판정의 최소 힘 [N].
  failure_cost: float = 1.0  # 넘어져 조기 종료하는 방식으로 비용을 피하지 않도록 적용.

  def __post_init__(self):
    if self.force_threshold < 0 or self.contact_force_threshold < 0:
      raise ValueError("Force thresholds must be nonnegative.")
    if self.force_scale <= 0 or self.velocity_squared_scale <= 0:
      raise ValueError("Noise cost scales must be positive.")
    if min(self.force_weight, self.velocity_weight) < 0:
      raise ValueError("Noise cost weights must be nonnegative.")
    if abs(self.force_weight + self.velocity_weight - 1.0) > 1e-6:
      raise ValueError("Noise cost weights must sum to one.")
    if not 0 <= self.failure_cost <= 1:
      raise ValueError("failure_cost must be in [0, 1].")
