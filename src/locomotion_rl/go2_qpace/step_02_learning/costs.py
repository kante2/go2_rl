"""소음 비용 설정: DiffImpact 소음 예측 모델 → 로봇 전체 착지 에너지."""

import math
from dataclasses import dataclass


@dataclass
class NoiseCostCfg:
  # 착지 판정: 소음 모델 학습 데이터(src/diffimpact/extract_touchdowns.py, 시뮬 녹화 기준)와 같은 규칙.
  touchdown_air_n: tuple[float, float, float, float] = (8.0, 10.0, 12.0, 9.0)  # 다리별 공중값 [N] (FR, FL, RR, RL)
  touchdown_on_delta: float = 10.0  # 접지: 힘 > air + on [N]
  touchdown_off_delta: float = 4.0  # 공중: 힘 < air + off [N]
  touchdown_rise_delta: float = 2.0  # 착지 시각: 힘이 air + rise 를 넘기 시작한 substep [N]
  touchdown_min_swing_s: float = 0.06  # 착지 직전 공중 최소 시간
  touchdown_min_stance_s: float = 0.04  # 착지 후 접지 최소 시간 (확정이 착지 후 35 ms로 늦어진다)
  contact_force_threshold: float = 1.0  # 진단값(최대 접촉력)용 접촉 기준 [N]. 소음 비용에는 쓰지 않는다.
  # 소음 모델은 몸통 수직 속도의 조용한 쪽(위로 움직이며 착지)을 무한히 외삽한다.
  # 학습 데이터에 없는 영역이므로 학습 범위 상단 + 이 값 [m/s]에서 자른다 (모델 속이기 방지).
  body_vz_extrapolation: float = 0.05
  # 실패 종료 뒤에는 학습 범위 상단(max_db)보다 이 값 [dB]만큼 시끄러운 소음이 영원히 이어진다고 보고
  # cost critic을 bootstrap한다. 넘어져서 제약을 피하는 해를 막는다.
  failure_margin_db: float = 3.0

  def __post_init__(self):
    if not all(math.isfinite(v) for v in (self.contact_force_threshold, self.body_vz_extrapolation,
                                          self.failure_margin_db)):
      raise ValueError("Noise cost parameters must be finite.")
    if not 0 <= self.touchdown_rise_delta < self.touchdown_off_delta < self.touchdown_on_delta:
      raise ValueError("Require 0 <= rise < off < on for touchdown detection.")
    if self.touchdown_min_swing_s < 0 or self.touchdown_min_stance_s < 0.025:
      raise ValueError("min_swing must be >= 0 and min_stance >= 25 ms (impulse window).")
    if self.contact_force_threshold < 0 or self.body_vz_extrapolation < 0:
      raise ValueError("contact_force_threshold and body_vz_extrapolation must be nonnegative.")
