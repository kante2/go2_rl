from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

# 아래 cast는 타입 설명용이다. 외부 mjlab에서 만든 command_term.cfg도 실행 중 변환하지 않는다.
from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")  # 이동 거리/속도를 평가할 로봇 선택자.

'''
 curriculum : 학습 과정 관리
-> 학습 중에 다양한 조건과 목표를 설정하여 에이전트가 점차적으로 더 어려운 과제를 해결하도록 유도
'''

class VelocityStage(TypedDict):
  step: int  # 환경 공통 정책 스텝 수가 이 값을 초과하면 적용한다.
  lin_vel_x: tuple[float, float] | None  # 전후 명령 범위 [m/s]; None/키 생략 시 기존 범위를 유지.
  lin_vel_y: tuple[float, float] | None  # 좌우 명령 범위 [m/s].
  ang_vel_z: tuple[float, float] | None  # 회전 명령 범위 [rad/s].


class RewardWeightStage(TypedDict):
  step: int  # 새 보상 가중치를 적용하기 시작하는 정책 스텝 기준.
  weight: float  # RewardTermCfg.weight를 대체할 값.


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  # distance는 에피소드 시작 지형 원점으로부터의 수평 변위 [m], 이동 경로 전체 길이가 아니다.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2  # 타일 길이의 절반보다 멀리 원점에서 벗어나면 상승.

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  # 목표 선속도로 에피소드 내 이동할 거리의 절반에 못 미치면 난이도를 낮춘다.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up  # 상승/하강이 동시에 선택되지 않도록 상승 조건을 우선한다.

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())  # 로그로 기록할 전체 환경의 평균 지형 난이도.


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # 이 함수는 일부 환경만의 값이 아니라 공유 명령 설정을 변경하므로 사용하지 않는다.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  # 조건을 만족한 단계를 순서대로 적용하므로 같은 축은 뒤 단계가 우선한다.
  # 재개 시 common_step_counter가 복원되면 해당 학습 단계의 범위가 다시 적용된다.
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  # 선택적 보상 커리큘럼 도우미. 현재 make_curriculum()에는 이 함수를 등록하지 않았다.
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)  # rewards.py 항목 이름으로 조회.
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
