"""Terrain and command difficulty schedules."""

from dataclasses import dataclass

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from . import mdp


@dataclass
class GaitCurriculumCfg:
  # 논문의 약 2k 전환을 학습 iteration 기준으로 해석한 설정이다.
  center_iteration: float = 2000.0
  width_iterations: float = 250.0
  minimum_scale: float = 0.2  # 후반에도 긴 주기 보상의 20%를 유지한다.
  steps_per_iteration: int = 24  # Runner가 실제 rollout 길이로 맞춘다.
  enabled: bool = True


def make_curriculum() -> dict[str, CurriculumTermCfg]:
  # command_name은 commands.py의 키와 같아야 한다. 여기서는 twist 속도 명령을 조정한다.
  return {
    "terrain_levels": CurriculumTermCfg(
      # 이동 거리와 명령 속도를 보고 지형 난이도를 조절한다. Flat에서는 이 항목을 제거한다.
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          # step은 모든 환경이 함께 진행한 정책 스텝 수로, num_envs를 곱한 경험 수가 아니다.
          # 각 리셋에서 counter > step인 단계를 차례로 적용한다. 속도 단위는 x/y [m/s], z [rad/s].
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.5, 1.5)},
          # 24스텝/학습회 기준 약 5,000회 이후 이동 속도 범위를 확장한다. 회전 범위는 위 값을 유지.
          {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
        ],
      },
    ),
  }
