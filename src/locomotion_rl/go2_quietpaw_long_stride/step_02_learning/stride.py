"""긴 보폭을 유도하는 설정. CLI 변경 후 환경 생성 시 한 번에 적용한다."""

from dataclasses import dataclass
import math


@dataclass
class StrideCfg:
  gait_period: float = 0.70  # 같은 발의 착지 간 목표 주기 [s].
  thigh_pose_std: float = 0.45  # 보행/달리기 자세 보상의 허용 폭 [rad].
  calf_pose_std: float = 0.60  # 실제 관절 한계를 바꾸지는 않는다.

  def __post_init__(self):
    for name in ("gait_period", "thigh_pose_std", "calf_pose_std"):
      value = getattr(self, name)
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"stride.{name} must be finite and positive.")


def apply_stride_settings(cfg):
  """등록 시점뿐 아니라 CLI/체크포인트 복원 이후에도 호출해야 한다."""
  stride = cfg.stride
  stride.__post_init__()
  cfg.rewards["foot_gait"].params["period"] = stride.gait_period
  for group in ("actor", "critic"):
    cfg.observations[group].terms["phase"].params["period"] = stride.gait_period
  for regime in ("std_walking", "std_running"):
    values = cfg.rewards["pose"].params[regime]
    values[r".*(FR|FL|RR|RL)_thigh_joint.*"] = stride.thigh_pose_std
    values[r".*(FR|FL|RR|RL)_calf_joint.*"] = stride.calf_pose_std
