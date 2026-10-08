"""Unitree Go2 velocity environment configurations."""

from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .base_env import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]  # 지형 이름의 타입 후보. 현재 이 파일의 함수에서는 참조하지 않는다.


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  # play=False는 학습, True는 평가용 설정. 공통 기본값을 만든 뒤 아래에서 로봇별 값을 채운다.
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500  # 충돌 검출 알고리즘의 최대 반복 수.
  cfg.sim.contact_sensor_maxmatch = 500  # 접촉 센서의 매칭 버퍼 용량.

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")  # 오른앞/왼앞/오른뒤/왼뒤. 보행 위상 offset도 이 순서.
  site_names = ("FR", "FL", "RR", "RL")  # 발 위치·속도를 읽는 XML site 이름.
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)  # 발의 충돌 형상 이름.

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),  # 접촉 여부/개수와 접촉력 [N]을 저장한다.
    reduce="netforce",  # 각 발의 여러 접촉을 하나의 합력으로 묶는다.
    # -> 이 센서 설정으로 인해, 각 발의 접촉력을 합산한 WORLD 좌표계 벡터를 받는다.
    # X,Y는 월드에 수평방향, Z은 수직 방향 = 로봇이 회전해도 축은 그대로
    num_slots=1,  # 발마다 합산 결과 한 슬롯.
    track_air_time=True,  # 접촉/공중 체류 시간과 착지 순간 판정에 필요하다.
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",  # 발 이외 형상별 접촉을 합력으로 합치지 않는다.
    num_slots=1,  # 형상별 저장 접촉 슬롯 수.
    history_length=4,  # 최근 접촉력 이력 길이; 순간 접촉을 종료 판정에 활용한다.
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  # 현재는 설정 타입만 확인한다. scale=0.25 등의 값은 actions.py에서 가져온다.
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5  # Go2 평가 화면의 카메라 거리 [m].
  cfg.viewer.elevation = -10.0  # 카메라 고도각 [deg].

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # 자세 보상 허용 폭 [rad]: 값이 작으면 기본 자세에서 조금만 벗어나도 보상이 크게 감소한다.
  # 키는 관절 이름 정규식, hip/thigh/calf는 각각 엉덩이 벌림/허벅지/종아리 관절이다.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  # 보행 때는 정지 때보다 관절을 크게 움직일 수 있도록 허용 폭을 늘린다.
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  # 현재 달리기의 허용 폭은 보행과 같다. 속도 구간 기준은 rewards.py에 있다.
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  # 주기 비율 0~1: FR/RL과 FL/RR 대각선 발 쌍을 반 주기 차이로 움직이도록 유도한다.
  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    # 여기의 mdp는 mjlab.tasks.velocity.mdp. 로컬 mdp/terminations.py의 함수를 직접 쓰지 않는다.
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},  # 힘 임계값 [N].
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)  # 평가 중 시간 초과를 사실상 해제하는 초 단위 길이.

    cfg.observations["actor"].enable_corruption = False  # 추론 입력의 관측 노이즈를 끈다.
    cfg.events.pop("push_robot", None)  # 주기적인 밀기 외란을 끈다.
    cfg.curriculum = {}  # 평가에서는 속도/지형 난이도를 학습 진행도에 따라 바꾸지 않는다.
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5  # 평가용 지형 타일의 열 수.
        cfg.scene.terrain.terrain_generator.num_rows = 5  # 평가용 지형 타일의 행 수.
        cfg.scene.terrain.terrain_generator.border_width = 10.0  # 외곽 경계 폭 [m].

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  # 센서/행동/보상 구성은 험지와 공유하고, 지형과 높이 스캔 등 평지 차이만 수정한다.
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300  # 평지의 환경당 제약 버퍼 용량.
  cfg.sim.mujoco.ccd_iterations = 50  # 평지의 충돌 검출 최대 반복 수.
  cfg.sim.contact_sensor_maxmatch = 64  # 평지의 접촉 센서 매칭 버퍼 용량.
  cfg.sim.nconmax = None  # 접촉 버퍼 크기를 mjlab이 추정하도록 한다.

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # play.py의 평지 평가 전용 범위. 학습용 commands.py의 ±1.5 rad/s와 별도이다.
    # lin_vel_x/y는 [m/s], ang_vel_z는 [rad/s]. 외부 키보드/UDP 명령은 해당 실행기가 관리한다.
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
