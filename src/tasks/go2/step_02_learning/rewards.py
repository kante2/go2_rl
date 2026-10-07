"""Reward weights and parameters; formulas live in mdp/rewards.py."""

# 사용할 보상함수/ 가중치/ 파리미터 설정
# /home/kante/unitree_rl_mjlab/src/tasks/go2/step_02_learning/mdp/rewards.py
#   여기에, 실제 점수를 계산하는 수식

import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

# mdp : 보상/관측/종료 조건 등을 계산하는 함수들을 모아놓은 py패키지
'''
step_02_learning/
├── rewards.py            ← 보상 설정 파일
└── mdp/                  ← 실제 계산 함수들의 묶음
    ├── __init__.py
    ├── rewards.py        ← 보상 계산
    ├── observations.py   ← 관측 계산
    ├── terminations.py   ← 종료 조건 계산
    ├── curriculums.py    ← 난이도 조정
    └── velocity_command.py
'''
from . import mdp


# 보상 설정을 담은, 딕셔너리 함수
'''
| 항목 | 가중치 | 유도하는 행동 |
|---|---:|---|
| `track_linear_velocity` | +1.0 | 목표 이동 속도 추종, 수직 출렁임 억제 |
| `track_angular_velocity` | +1.0 | 목표 회전 속도 추종 |
| `body_orientation_l2` | -1.0 | 몸통을 수평에 가깝게 유지 |
| `pose` | +1.0 | 명령 속도에 맞는 범위에서 기본 관절 자세 유지 |
| `body_ang_vel` | -0.05 | 몸통의 앞뒤·좌우 회전 흔들림 억제 |
| `angular_momentum` | -0.025 | 각운동량 억제 |
| `is_terminated` | -200.0 | 실패 종료 회피 |
| `joint_acc_l2` | -0.00000025 | 급격한 관절 가속 억제 |
| `joint_pos_limits` | -10.0 | 관절의 소프트 제한 범위 이탈 억제 |
| `action_rate_l2` | -0.05 | 연속된 행동 명령의 급격한 변화 억제 |
| `foot_gait` | +0.5 | 정해진 발 접촉 리듬 유도 |
| `foot_clearance` | -1.0 | 움직이는 발의 높이를 목표에 가깝게 유도 |
| `foot_slip` | -0.25 | 지면에 닿은 발의 미끄러짐 억제 |
| `soft_landing` | -0.001 | 착지 순간의 접촉력 억제 |
| `stand_still` | -1.0 | 정지 명령일 때 기본 자세 유지 |

 -> 정리
1. track_linear_velocity, track_angular_velocity
    - 목표 속도를 잘 따라가도록 유도
2. body_orientation_l2, pose, body_ang_vel, angular_momentum
    - 몸통과 관절이 안정적으로 움직이도록 유도
3. is_terminated, joint_acc_l2, joint_pos_limits, action_rate_l2
    - 실패 종료, 관절 급격한 움직임, 행동 급격한 변화 억제
4. foot_gait, foot_clearance, foot_slip, soft_landing, stand_still
    - 발이 지면과 잘 접촉하도록 유도
5. 그 결과가 Critic의 예상보다 좋았는지 판단

'''
def make_rewards() -> dict[str, RewardTermCfg]:
  return {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      }
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
  }
