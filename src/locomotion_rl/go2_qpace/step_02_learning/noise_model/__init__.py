"""소음 모델(torch, GPU): MuJoCo 착지 상태 → DiffImpact 앙상블 → 정책 스텝별 A-가중 착지 소음 dB.

diffimpact.py, diffimpact_ensemble.pt/.json: go2_noise_aware 8d4270d (물리 법칙 모델) 그대로.
touchdown_features.py: go2_qpace 버전 (착지 판정을 학습 데이터 규칙으로, 조용한 쪽 발별 하한 추가).
"""

from .diffimpact import DEFAULT_TABLE, SPL_OFFSET_DB, DiffImpactNoiseModel
from .touchdown_features import (
  LEGS,
  DiffImpactStepNoise,
  TouchdownFeatureTracker,
  leg_kinematics,
  torque_grf_z,
  unitree_joint_order,
)

__all__ = [
  "DEFAULT_TABLE", "SPL_OFFSET_DB", "DiffImpactNoiseModel", "LEGS",
  "DiffImpactStepNoise", "TouchdownFeatureTracker", "leg_kinematics", "torque_grf_z", "unitree_joint_order",
]
