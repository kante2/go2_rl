"""Reward weights and parameters; formulas live in mdp/rewards.py."""

# 사용할 보상함수/ 가중치/ 파리미터 설정
# /home/kante/go2_rl/src/locomotion_rl/go2_quietpaw/step_02_learning/mdp/rewards.py
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
| `track_linear_velocity`   | +1.0 | 목표 이동 속도 추종, 수직 출렁임 억제 |
| `track_angular_velocity`  | +1.0 | 목표 회전 속도 추종 |
| `body_orientation_l2`     | -1.0 | 몸통을 수평에 가깝게 유지 |
| `pose`                    | +1.0 | 명령 속도에 맞는 범위에서 기본 관절 자세 유지 |
| `body_ang_vel`            | -0.05 | 몸통의 앞뒤·좌우 회전 흔들림 억제 |
| `angular_momentum`        | -0.025 | 각운동량 억제 |
| `is_terminated`           | -200.0 | 실패 종료 회피 |
| `joint_acc_l2`            | -0.00000025 | 급격한 관절 가속 억제 |
| `joint_pos_limits`        | -10.0 | 관절의 소프트 제한 범위 이탈 억제 |
| `action_rate_l2`          | -0.05 | 연속된 행동 명령의 급격한 변화 억제 |
| `foot_gait`               | +0.5 | 정해진 발 접촉 리듬 유도 |
| `foot_clearance`          | -1.0 | 움직이는 발의 높이를 목표에 가깝게 유도 |
| `foot_slip`               | -0.25 | 지면에 닿은 발의 미끄러짐 억제 |
| `soft_landing`            | 0.0 | 별도 noise cost를 사용하므로 비활성 |
| `stand_still`             | -1.0 | 정지 명령일 때 기본 자세 유지 |

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
  # func의 환경별 결과에 weight를 곱해 보상으로 합산한다. 음수 weight는 비용을 페널티로 만든다.
  # mjlab의 기본 scale_by_dt=True에서는 정책 주기 dt도 곱한다. weight 자체가 스텝당 최종 점수는 아니다.
  # std는 측정 노이즈가 아니라 exp(-오차²/std²) 보상의 허용 폭이다. 작을수록 오차에 엄격하다.
  # command_threshold는 |선속도|+|회전속도|를 더한 활동도 판정값이며 순수 m/s 단위는 아니다.
  return {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},  # std=0.5, 제곱오차 분모=0.25.
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},  # 회전 추종 오차의 허용 폭.
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
        "std_standing": {},  # 정지 시 관절별 허용 폭 [rad]; go2_env.py가 채운다.
        "std_walking": {},  # 보행 시 관절별 허용 폭 [rad].
        "std_running": {},  # 달리기 시 관절별 허용 폭 [rad].
        "walking_threshold": 0.1,  # 활동도 < 0.1이면 정지 자세 기준을 적용한다.
        "running_threshold": 1.5,  # 활동도 >= 1.5이면 달리기 자세 기준을 적용한다.
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,  # 월드 x/y 각속도 제곱에 페널티를 주어 몸통 흔들림을 억제한다.
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,  # 전신 각운동량 크기의 제곱에 주는 페널티.
      params={"sensor_name": "robot/root_angmom"},  # 로봇 XML의 각운동량 센서 이름.
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),  # 시간 초과를 제외한 실패 종료.
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),  # 관절 가속도 제곱합 억제.
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),  # 소프트 관절 한계 위반 억제.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),  # 현재/직전 행동 차이 제곱합 억제.
    "foot_gait": RewardTermCfg(
      func=mdp.quietpaw_gait,
      weight=0.5,
      params={
        "period": 0.6,  # 보행 한 주기의 길이 [s]; 관측 phase의 주기와 맞춘다.
        "offset": [0.0, 0.5],  # 공통 기본값. Go2에서는 네 발 [0, 0.5, 0.5, 0]으로 덮어쓴다.
        "threshold": 0.56,  # 발 위상 0~1 중 앞 56%를 접지 구간으로 정한다.
        "command_threshold": 0.1,  # 활동도가 이 값보다 커야 보행 리듬 보상을 적용한다.
        "command_name": "twist",  # 목표 명령을 읽는 command manager 키.
        "sensor_name": "feet_ground_contact",  # 실제 발 접촉과 목표 리듬을 비교할 센서.
      }
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,  # 움직이는 발의 목표 월드 z 높이 [m]; 지형 상대 높이는 아니다.
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
      weight=0.0,  # 접촉력은 별도 noise cost로 학습하므로 고정 중복 페널티를 끈다.
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
