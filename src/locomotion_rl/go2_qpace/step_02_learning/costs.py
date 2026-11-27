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
  # 착지 한 발의 소음 하한 [dB re FS²]. 물리 법칙 모델은 접촉 속도 → 0 에서 하중 소리 A0 까지 내려가고
  # A0 가 작은 RL 다리는 약 −106 dB 가 된다 (학습 데이터 밖). 이 값 아래는 같은 비용으로 본다 (모델 속이기 방지).
  # None: 모델 db_ref["p1"] (학습 실측 착지 예측 dB 1 백분위, 현재 표 −76.2 dB).
  foot_floor_db: float | None = None
  # 착지 규칙(유각기 ≥ 60 ms, 입각기 ≥ 40 ms)에서 빠진 짧은 접촉에도 접촉 속도로 소음 비용을 매긴다.
  # 끄면 발을 빠르게 두드려 비용을 0 으로 만드는 해가 생긴다 (2026-11-27 학습에서 확인, 접촉의 89% 탈락).
  count_rejected_contacts: bool = True
  # 불확실성 반영 비용: 접촉 소음 [dB] = 앙상블 평균 + k·앙상블 표준편차. 모델이 모르는 영역은 시끄럽다고 본다.
  # 0 이면 평균만 쓴다. (2026-11-27 학습: 데이터 밖으로 옮겨 간 보행이 "조용하다"로 예측돼 붕괴 → 1.0)
  uncertainty_k: float = 1.0
  uncertainty_max_db: float = 3.0  # 불확실성 가산의 상한 [dB] (극단 접촉 몇 개가 평균을 지배하지 않게)
  # 실패 종료 뒤에는 학습 범위 상단(max_db)보다 이 값 [dB]만큼 시끄러운 소음이 영원히 이어진다고 보고
  # cost critic을 bootstrap한다. 넘어져서 제약을 피하는 해를 막는다.
  failure_margin_db: float = 3.0

  def __post_init__(self):
    if not all(math.isfinite(v) for v in (self.contact_force_threshold, self.failure_margin_db, self.uncertainty_k)):
      raise ValueError("Noise cost parameters must be finite.")
    if self.foot_floor_db is not None and (math.isnan(self.foot_floor_db) or self.foot_floor_db == math.inf):
      raise ValueError("foot_floor_db must be a number below +inf (use -inf to disable).")
    if not 0 <= self.touchdown_rise_delta < self.touchdown_off_delta < self.touchdown_on_delta:
      raise ValueError("Require 0 <= rise < off < on for touchdown detection.")
    if self.touchdown_min_swing_s < 0 or self.touchdown_min_stance_s < 0.025:
      raise ValueError("min_swing must be >= 0 and min_stance >= 25 ms (impulse window).")
    if self.uncertainty_k < 0 or not self.uncertainty_max_db >= 0:
      raise ValueError("uncertainty_k and uncertainty_max_db must be nonnegative.")
    if self.contact_force_threshold < 0:
      raise ValueError("contact_force_threshold must be nonnegative.")
