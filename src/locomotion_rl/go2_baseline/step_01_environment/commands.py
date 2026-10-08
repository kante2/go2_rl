"""Velocity commands sampled during training."""

import math

from mjlab.managers.command_manager import CommandTermCfg
# 실제 명령 생성은 설치된 mjlab 구현을 사용한다. step_02_learning/mdp의 동명 클래스와 구분한다.
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def make_commands() -> dict[str, CommandTermCfg]:
  # twist = [vx, vy, wz]. vx/vy는 몸체 기준 [m/s], wz는 몸체 z축 회전 속도 [rad/s].
  # 각 환경에서 독립적으로 샘플링하며, 속도 범위는 step_02_learning/curriculum.py도 덮어쓴다.
  return {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",  # 자세와 실제 속도를 읽을 로봇 이름.
      resampling_time_range=(3.0, 8.0),  # 시뮬레이션 시간 3~8초 간격으로 새 목표를 뽑는다.
      rel_standing_envs=0.05,  # 샘플링 시 정지 명령 [0, 0, 0]을 선택할 확률 5%.
      heading_command=True,  # 목표 방위각과 현재 방위각의 차이로 wz를 계산한다.
      heading_control_stiffness=0.5,  # wz = 0.5 × 방향 오차 [rad]; 이후 ang_vel_z 범위로 제한.
      debug_vis=True,  # 뷰어에서 목표/실제 속도를 화살표로 표시할 수 있도록 한다.
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 2.0),  # 후진 -1 ~ 전진 +2 m/s.
        lin_vel_y=(-1.0, 1.0),  # 오른쪽 -1 ~ 왼쪽 +1 m/s.
        ang_vel_z=(-1.5, 1.5),  # 좌우 회전 한계 [rad/s]; 양수는 z축 기준 반시계 방향.
        heading=(-math.pi, math.pi),  # 월드 기준 목표 방위각 -180~180도 [rad].
      ),
    )
  }
