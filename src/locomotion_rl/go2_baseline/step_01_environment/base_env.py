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
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=1,
      extent=2.0,
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
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
