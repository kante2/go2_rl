"""Reset states, randomization and external disturbances."""

from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


def make_events() -> dict[str, EventTermCfg]:
  # mode: startup=환경 생성 시, reset=각 에피소드 초기화 시, interval=지정 간격마다 실행.
  # 각 범위에서 균등 샘플링해 환경마다 다른 초기 조건·마찰·센서 오차를 경험하게 한다.
  return {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),  # 기본 위치에서 전후 위치 오프셋 [m].
          "y": (-0.5, 0.5),  # 기본 위치에서 좌우 위치 오프셋 [m].
          "z": (0.0, 0.0),  # 기본 높이는 유지한다. 월드 높이를 0으로 만드는 설정은 아니다.
          "yaw": (-3.14, 3.14),  # 초기 몸체 방향 오프셋 [rad], 거의 한 바퀴 전체 범위.
        },
        "velocity_range": {},  # 기본 root 속도에 추가할 랜덤 속도가 없다.
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),  # 기본 관절각에 더할 오프셋 [rad]; 현재는 0.
        "velocity_range": (-0.0, 0.0),  # 기본 관절 속도에 더할 오프셋 [rad/s]; 현재는 0.
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),  # 전체 관절 대상.
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 6.0),  # 시뮬레이션 시간 5~6초마다 속도 외란을 추가한다.
      params={
        "velocity_range": {
          # 현재 root 속도에 더하는 월드 좌표계 외란: x/y/z [m/s], roll/pitch/yaw [rad/s].
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",  # 기존 값에 곱하지 않고 샘플링한 마찰계수로 대체한다.
        "ranges": (0.3, 1.6),  # 접선 마찰계수 범위(무차원), 작을수록 잘 미끄러진다.
        "shared_random": True,  # 한 환경의 네 발은 같은 값, 서로 다른 환경은 별도 샘플.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),  # 관절 엔코더 영점 오차 [rad], 약 ±0.86도.
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",  # 기본 몸체 질량중심 위치에 아래 오프셋을 더한다.
        "ranges": {
          0: (-0.05, 0.05),  # 몸체 로컬 x축 질량중심 오프셋 [m].
          1: (-0.05, 0.05),  # 몸체 로컬 y축 질량중심 오프셋 [m].
          2: (-0.05, 0.05),  # 몸체 로컬 z축 질량중심 오프셋 [m].
        },
      },
    ),
  }
