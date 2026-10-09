"""QuietPaw의 Actor·분해 Critic·PPO-Lagrangian 설정."""

'''

ppo - PROXIMAL POLICY OPTIMIZATION
- RL알고리즘 중 하나, actor-critic을 학습하는 알고리즘

policy - 현재 상황에서 어떤 행동을 선택할 것인가

Go2를 예로 들면 다음처럼 학습합니다.
1. Actor가 관절 행동을 선택합니다.
2. 시뮬레이션에서 로봇이 움직입니다.
3. 목표 속도를 잘 따라가고 자세가 안정적이면 좋은 보상을 받습니다.
4. 그 결과가 Critic의 예상보다 좋았는지 판단합니다.
5. 예상보다 좋은 결과를 낸 행동이 더 잘 나오도록 Actor를 수정합니다.
6. Reward/Cost Critic도 각각 보행 보상과 소음 비용을 더 정확히 예측하도록 수정합니다.

'''

# 설정 클래스 : RL 설정을 정의하는 클래스들
from dataclasses import dataclass, field

from .pid_lagrangian import PidLagrangianCfg

from mjlab.rl import (
    RslRlModelCfg,          # Actor·Critic 신경망 설정
    RslRlOnPolicyRunnerCfg, # 전체 학습 설정을 묶음
    RslRlPpoAlgorithmCfg,   # PPO 학습 방법 설정
)


@dataclass
class QuietPawCriticCfg(RslRlModelCfg):
  feature_dim: int = 64
  condition_hidden_dims: tuple[int, ...] = (64, 64)
  class_name: str = "src.locomotion_rl.go2_quietpaw.step_03_training.critics:QuietPawCritics"


@dataclass
class QuietPawPpoCfg(RslRlPpoAlgorithmCfg):
  cost_value_loss_coef: float = 1.0
  pid: PidLagrangianCfg = field(default_factory=PidLagrangianCfg)
  class_name: str = "src.locomotion_rl.go2_quietpaw.step_03_training.ppo_lagrangian:QuietPawPPO"


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """baseline의 PPO 기본값에 두 Critic과 PID 설정을 더한다."""
  # 이 함수는 설정만 만든다. 실제 제약 PPO 업데이트는 ppo_lagrangian.py에서 수행한다.
  # CLI --agent.* 값이 아래 기본값을 덮어쓴다. 체크포인트 재개 시에는 저장된 모델/옵티마이저도 복원한다.
  return RslRlOnPolicyRunnerCfg(
    # PPO는, actor-critic을 학습하는 알고리즘
    # 신경망은 간단하게, 숫자 묶음 --> 여러 계산을 거쳐 --> 새로운 숫자 묶음을 만드는 함수
    # 학습을 통해, 가중치-편향이 좋은 행동이 나오도록 조정하는 구조
    # <1> actor -> 로봇이 어떻게 움직이는 지 결정하는 신경망
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),  # 은닉층 3개의 뉴런 수. Flat 입력 48, 행동 출력 12는 환경이 정한다.
      activation="elu", # 계산층 사이에 사용하는 비선형 계산함수/ 신경망의 복잡한 상태, 행동관계를 표현하도록 돕는다
      obs_normalization=True, # 센서마다 숫자 크기가 달라도, 신경망이 비교적 안정적으로 학습하도록 도와주는 전처리
      distribution_cfg={
         # 확률적 출력을 위한 설정 -> 이는, 학습 중 여러 행동을 시도하는 방식
         # 현재 actor가 계산한 행동을 평군삼음/ 그 주변에서 행동을 무작위로 뽑음
        "class_name": "GaussianDistribution",  # 정책 평균 주변에서 정규분포로 행동을 샘플링한다.
        "init_std": 1.0,  # 새 학습의 행동 표준편차 초기값; 재개 시 체크포인트의 학습값을 복원한다.
        "std_type": "scalar",  # 관측과 무관한 행동 차원별 표준편차 파라미터를 학습한다.
      },
    ),
    # <2> critic -> 앞으로의 보행 보상과 소음 비용을 각각 예측한다.
    # 각 critic은 상태 특징 ξ(s)와 조건 가중치 w(ε)의 내적으로 value를 만든다.
    critic=QuietPawCriticCfg(
      hidden_dims=(512, 256, 128), # 중간 계산층을3개 만들고, 각 층의 출력 크기를 512-256-128로 설정
      activation="elu",            # ELU 비선형 활성화 함수 사용
      obs_normalization=True,      # 관측값의 크기를 정규화해서 입력
    ),
    # ---------
    algorithm=QuietPawPpoCfg(
      value_loss_coef=1.0,  # Critic 가치 오차가 전체 손실에 기여하는 비중.
      use_clipped_value_loss=True,  # Critic 값의 과도한 갱신을 제한하는 clipped value loss 사용.
      clip_param=0.2,  # PPO 정책 확률비 클립 폭: 목적함수에서 1±0.2 범위를 사용한다.
      entropy_coef=0.01,  # 행동 다양성/탐색을 유지하는 엔트로피 보너스 계수.
      num_learning_epochs=5,  # 수집한 rollout 데이터를 학습에 재사용하는 횟수.
      num_mini_batches=4,  # rollout을 epoch마다 4개 소배치로 분할: 반복당 기본 5×4회 업데이트.
      learning_rate=1.0e-3,  # 새 학습 옵티마이저의 초기 학습률 0.001.
      schedule="adaptive",  # 정책 변화량(KL)에 따라 학습률을 조정한다.
      gamma=0.99,  # 미래 보상의 할인율. 1에 가까울수록 먼 미래의 보상을 더 반영한다.
      lam=0.95,  # GAE advantage 추정의 lambda; 편향과 분산 사이의 균형을 조절한다.
      desired_kl=0.01,  # adaptive 학습률 조절이 기준으로 삼는 정책 간 KL 크기.
      max_grad_norm=1.0,  # 전체 gradient norm의 클립 상한, 과도한 갱신을 억제한다.
    ),
    # 경험 수집/ 실행 주기를 설정
    experiment_name="go2_quietpaw",  # baseline과 체크포인트를 분리한다.
    logger="tensorboard",  # 기본 실행은 로컬 기록. 필요 시 CLI에서 wandb를 선택한다.
    upload_model=False,
    save_interval=100,    # 학습 반복 100회마다, 체크포인트를 저장
    num_steps_per_env=24, # 환경당 rollout 길이. 1,024개 환경이면 학습 1회당 경험 24,576개 수집.
    max_iterations=10001, # 이번 실행의 학습 반복 수. 재개 시에도 복원 시점부터 이 횟수만큼 추가한다.
  )
