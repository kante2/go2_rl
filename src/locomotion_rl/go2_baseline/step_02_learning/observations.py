"""Actor and critic inputs, including ordering, noise and history."""

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp


def make_observations(
  terrain_scan: RayCastSensorCfg,
) -> dict[str, ObservationGroupCfg]:
  # dict 순서가 연결된 관측 벡터의 순서이다. 바꾸면 기존 체크포인트/실로봇 관측 매핑도 맞춰야 한다.
  # Go2 Flat Actor: 각속도3 + 중력3 + 명령3 + 위상2 + 관절각12 + 관절속도12 + 직전행동12 = 47.
  # Rough에는 height_scan이 추가된다. B=병렬 환경 수, 관측 텐서 첫 축은 항상 B이다.
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},  # 몸체 기준 각속도 xyz [rad/s].
      noise=Unoise(n_min=-0.2, n_max=0.2),  # 각 성분에 더하는 균등 노이즈 [rad/s].
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),  # 몸체 좌표로 투영한 단위 중력 방향의 노이즈.
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},  # 목표 [vx, vy, wz]; 실제 측정 속도와 구분한다.
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},  # 주기 0.6 s를 sin/cos 2개 값으로 표현.
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),  # 기본 자세 대비 관절각 오차에 더하는 노이즈 [rad].
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),  # 기본 속도 대비 관절 속도에 더하는 노이즈 [rad/s].
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),  # 직전 정책 출력 12개; 변환 후 목표각이 아니다.
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),  # 지형 높이 관측에 더하는 노이즈 [m], scale 적용 전.
      scale=1 / terrain_scan.max_distance,  # 스캔 거리로 나눠 입력 크기를 줄인다.
    ),
  }

  critic_terms = {
    # Critic은 학습 중 가치 추정에만 쓰며, 실로봇 Actor에 없는 추가 상태도 입력받는다.
    # Flat에서는 아래 추가 항목 3+4+4+4+12=27개를 더해 총 74차원이다.
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},  # 몸체 기준 선속도 [m/s].
      noise=Unoise(n_min=-0.5, n_max=0.5),  # 아래 Critic 그룹은 노이즈가 꺼져 있어 적용되지 않는다.
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      # 발 site의 월드 z 좌표 [m]. 로컬 지면과의 상대 높이를 직접 계산한 값은 아니다.
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      # 발마다 현재 연속해서 공중에 떠 있는 시간 [s].
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      # 발마다 접촉하면 1, 아니면 0인 값.
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      # 네 발의 월드 힘 xyz [N]를 펼쳐 sign(F)*log(1+|F|)로 압축한 12개 값.
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,  # 항목을 위 순서대로 이어 하나의 관측 벡터로 만든다.
      enable_corruption=True,  # 학습 Actor에는 지정한 관측 노이즈를 적용한다.
      history_length=1,  # 현재 시점만 입력한다. 과거 관측 여러 개를 쌓지 않는다.
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,  # Critic에는 깨끗한 시뮬레이션 상태를 제공한다.
      history_length=1,  # Critic도 현재 시점만 사용한다.
    ),
  }
