"""RL configuration for Unitree Go2 velocity task."""

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
6. Critic도 다음에는 결과를 더 정확히 예측하도록 수정합니다.

'''

# 설정 클래스 : RL 설정을 정의하는 클래스들
from mjlab.rl import (
    RslRlModelCfg,          # Actor·Critic 신경망 설정
    RslRlOnPolicyRunnerCfg, # 전체 학습 설정을 묶음
    RslRlPpoAlgorithmCfg,   # PPO 학습 방법 설정
)

def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    # PPO는, actor-critic을 학습하는 알고리즘
    # 신경망은 간단하게, 숫자 묶음 --> 여러 계산을 거쳐 --> 새로운 숫자 묶음을 만드는 함수
    # 학습을 통해, 가중치-편향이 좋은 행동이 나오도록 조정하는 구조
    # <1> actor -> 로봇이 어떻게 움직이는 지 결정하는 신경망
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu", # 계산층 사이에 사용하는 비선형 계산함수/ 신경망의 복잡한 상태, 행동관계를 표현하도록 돕는다
      obs_normalization=True, # 센서마다 숫자 크기가 달라도, 신경망이 비교적 안정적으로 학습하도록 도와주는 전처리
      distribution_cfg={
         # 확률적 출력을 위한 설정 -> 이는, 학습 중 여러 행동을 시도하는 방식
         # 현재 actor가 계산한 행동을 평군삼음/ 그 주변에서 행동을 무작위로 뽑음
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    # <2> critic -> 지금 상황에서 앞으로 받을 보상을 예측하는 신경망
    # 현재 자세가 안정적/ 목표 속도를 잘 따라간다면 -> 앞으로 좋은 보상을 받을 가능성이 높음
    # 예측값, VALUE
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), # 중간 계산층을3개 만들고, 각 층의 출력 크기를 512-256-128로 설정
      activation="elu",            # ELU 비선형 활성화 함수 사용
      obs_normalization=True,      # 관측값의 크기를 정규화해서 입력
    ),
    # ---------
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    # 경험 수집/ 실행 주기를 설정
    experiment_name="go2_velocity",
    save_interval=100,    # 학습 반복 100회마다, 체크포인트를 저장
    num_steps_per_env=24, # PPO업데이트 전에, 각 환경에서 행동 24번 수행
    max_iterations=10001, # 경험 수집과 PPo업데이트를 10,001회 반복
  )
