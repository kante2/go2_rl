"""Assemble environment defaults from the preparation and learning stages."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.viewer import ViewerConfig

from .actions import make_actions
from .commands import make_commands
from .events import make_events
from .scene import make_terrain_scan_sensor
from ..step_02_learning.curriculum import make_curriculum
from ..step_02_learning.metrics import make_metrics
from ..step_02_learning.observations import make_observations
from ..step_02_learning.rewards import make_rewards
from ..step_02_learning.terminations import make_terminations


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create fresh defaults before applying the Go2 terrain-specific settings."""
  # 각 make_*는 새 설정 객체를 만든다. 이후 go2_env.py에서 Go2 이름과 평지/험지 차이를 덮어쓴다.
  terrain_scan = make_terrain_scan_sensor()
  observations = make_observations(terrain_scan)
  metrics = make_metrics()
  actions = make_actions()
  commands = make_commands()
  events = make_events()
  rewards = make_rewards()
  terminations = make_terminations()
  curriculum = make_curriculum()

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",  # 험지 생성기를 사용하며, Flat 환경은 plane으로 변경한다.
        terrain_generator=replace(ROUGH_TERRAINS_CFG),  # 공통 지형 설정을 복사해 구성한다.
        max_init_terrain_level=5,  # 처음 배치할 지형 난이도 레벨의 상한.
      ),
      sensors=(terrain_scan,),  # 높이 스캔 센서. Go2 설정에서 발 접촉 센서도 추가한다.
      num_envs=1,  # 기본 병렬 환경 수. 학습 CLI의 --env.scene.num-envs 1024 등으로 변경한다.
      extent=2.0,  # MuJoCo stat.extent의 장면 크기 척도 [m]; 환경 사이 간격은 env_spacing이다.
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,  # 지정한 로봇 body를 카메라 기준점으로 삼는다.
      entity_name="robot",
      body_name="",  # Go2 설정에서 base_link로 채운다.
      distance=3.0,  # 카메라와 기준점 거리 [m]; Go2 설정에서 다시 조정한다.
      elevation=-5.0,  # 카메라 고도각 [deg].
      azimuth=90.0,  # 카메라 방위각 [deg].
    ),
    sim=SimulationCfg(
      nconmax=35,  # 환경당 접촉 버퍼 할당 기준. Flat에서는 자동 추정(None)을 사용한다.
      njmax=1500,  # 환경당 물리 제약 버퍼 용량; 관절 개수를 뜻하지 않는다.
      mujoco=MujocoCfg(
        timestep=0.005,  # 물리 적분 간격 [s]: 200 Hz.
        iterations=10,  # 물리 제약 솔버의 최대 반복 수.
        ls_iterations=20,  # 솔버 내부 선 탐색의 최대 반복 수.
      ),
    ),
    decimation=4,  # 정책 행동 하나를 물리 4스텝 유지: 정책 주기 0.02 s, 50 Hz.
    episode_length_s=20.0,  # 에피소드 시간 제한 [s]: 정책 스텝 기준 1,000회.
  )
