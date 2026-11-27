# Go2 QuietPaw 1124 (CNCP)

[QuietPaw 논문](../../../quietpaw_.pdf)(arXiv:2503.05035)의 **Conditional Noise-Constrained Policy (CNCP)** 를
Go2 MuJoCo/mjlab에서 학습한다. 다른 locomotion 패키지(`go2_quietpaw`, `go2_baseline` 등)를 import하지 않는
독립 패키지다. 외부 의존은 `mjlab`, `rsl_rl`, `src/assets`(Go2 모델)뿐이다.

태스크: `Unitree-Go2-QuietPaw-1124-Flat`, `Unitree-Go2-QuietPaw-1124-Rough`
로그·체크포인트: `logs/rsl_rl/go2_quietpaw_1124/<날짜>_<run-name>/`

## 실행

```bash
# 학습 (논문 기본 4,000 env / 10k iter, GPU에 맞게 env 수 조절)
bash scripts/run_go2_docker.sh python scripts/train.py Unitree-Go2-QuietPaw-1124-Flat \
  --env.scene.num-envs 2048 --agent.max-iterations 10000 --agent.save-interval 100 \
  --agent.logger tensorboard --agent.run-name cncp_flat

# 관찰 (quietness q → ε = 1 − q). 브라우저 http://localhost:8080
bash scripts/run_go2_docker.sh python scripts/play.py Unitree-Go2-QuietPaw-1124-Flat \
  --checkpoint-file logs/rsl_rl/go2_quietpaw_1124/RUN/model_9999.pt --viewer viser --quietness 0.8

# 표 I 형식 평가 (ε × 속도 → CSV)
bash scripts/run_go2_docker.sh python -m src.locomotion_rl.go2_quietpaw_1124.step_04_evaluation.evaluate \
  --checkpoint-file logs/rsl_rl/go2_quietpaw_1124/RUN/model_9999.pt

# 테스트 (CPU / GPU 포함)
bash scripts/run_go2_docker.sh python -m unittest src.locomotion_rl.go2_quietpaw_1124.tests.test_cncp -v
bash scripts/run_go2_docker.sh env QUIETPAW_1124_GPU_TESTS=1 MUJOCO_GL=egl \
  python -m unittest src.locomotion_rl.go2_quietpaw_1124.tests.test_cncp -v
```

## 폴더 구조

| 위치 | 역할 |
|---|---|
| `registry.py` | Flat/Rough 태스크 등록 |
| `step_01_environment/base_env.py` | 환경 설정 조립, `only_positive_rewards` |
| `step_01_environment/go2_env.py` | Go2 센서·보상 대상·종료, 환경 수, play 설정 |
| `step_01_environment/cncp_env.py` | substep 착지 측정, 식 (1) 비용, 보상 자르기, reset 전 관측 |
| `step_01_environment/{scene,actions,commands,events}.py` | 높이 스캔, 관절 위치 행동, 속도 명령, reset·DR |
| `step_02_learning/costs.py`, `mdp/costs.py` | 식 (1) 설정과 계산 |
| `step_02_learning/mdp/noise_condition.py` | 식 (4): 환경별 ε 배정 |
| `step_02_learning/observations.py` | Actor 48 / Critic 75 차원 (Flat), 마지막 값 ε |
| `step_02_learning/rewards.py`, `mdp/rewards.py` | 보행 보상 |
| `step_02_learning/curriculum.py`, `mdp/curriculums.py` | 2k 지점 gait 보상 sigmoid 감소, 속도 명령 단계 |
| `step_03_training/critics.py` | 식 (6): V_f(s, ε) = ξ_f(s)ᵀ w_f(ε), reward·cost 독립 |
| `step_03_training/storage.py` | reward/cost 두 신호 rollout 저장, GAE |
| `step_03_training/pid_lagrangian.py` | 환경별 λ_i PID |
| `step_03_training/cncp_ppo.py` | 식 (7)~(10), Algorithm 1, 실패 bootstrap |
| `step_03_training/ppo.py` | 네트워크·PPO·PID 하이퍼파라미터 |
| `step_03_training/{runner,train}.py` | 저장·재개·ONNX, 학습 CLI |
| `step_04_evaluation/{play,evaluate,checkpoint}.py` | 관찰, 표 I 평가, 체크포인트 설정 복원 |
| `tests/test_cncp.py` | 식 (1), 할인 비용, 실패 bootstrap, 식 (8) 순서, 기본값, 패키지 분리 |

## 논문 ↔ 구현

| 논문 | 구현 |
|---|---|
| 식 (1) λ1Σexp((Fmax−‖F‖)/σF) + λ2Σexp(−v²/σv), σv=0.5, σF=50, λ=0.5, [0,1] 정규화 | `mdp/costs.py`. 정규화 = 1 − S/S₀ (S₀ = 힘·속도 0일 때 값) |
| 식 (2)(3) J_c ≤ ε | J_c = (1−γ)·에피소드 시작부터 할인 비용 합 ∈ [0, 1] |
| 식 (4) ε_i 를 [0,1]에 균등 배정, 환경 N개 | `noise_condition.py`, 학습 중 고정 |
| 식 (5)(6) ξ_f(s)ᵀ w_f(ε), 64차원, 은닉층 [512,256,128] | `critics.py`, `ppo.py` |
| 식 (7) critic MSE | value clipping 끔 |
| 식 (8) Â = (A_r − λ_i A_c)/(1+λ_i), 환경별 λ_i | `cncp_ppo.py` |
| 식 (9)(10) 환경 평균 클립 surrogate | `cncp_ppo.py` |
| Algorithm 1: advantage 계산 → λ 갱신 → actor·critic 갱신 | 같은 순서 |
| λ PID ([53]) | `pid_lagrangian.py` |
| 4,000 env, 10k iter, 50 Hz | `go2_env.py`, `ppo.py`, decimation 4 × 5 ms |
| gait 보상 2k 지점 sigmoid 감소 | `curriculum.py` |
| 평가 ε ∈ {0,.1,.2,.3,.4,.6,.8,1}, 0.5~2.25 m/s, E[max(0, c−ε)] | `evaluate.py` |

## 논문에 없거나 다르게 정한 것

| 항목 | 선택 | 이유 |
|---|---|---|
| 식 (1) 부호 | 기본 `increasing`(충격이 클수록 비용 증가). `--env.noise-cost.orientation printed`로 인쇄된 방향 선택 가능 | 인쇄된 식은 충격이 클수록 값이 작아져 본문·그림 2와 반대 |
| Fmax | 50 N. 정규화 뒤에는 힘·속도 항의 상대 비중만 바꾼다 | 논문에 수치 없음 |
| 힘 F_i | 정책 한 스텝(4 substep) 안의 발별 최대 접촉력 | 정의 없음. 서 있는 하중도 비용에 들어가므로 ε가 매우 작으면 만족이 불가능할 수 있다 |
| v_impact | 착지 substep 직전 발 site의 월드 수직 하강 속도 | 정의 없음 |
| J_c 정규화 (1−γ) | ε ∈ [0,1]과 할인 비용 합의 척도를 맞춘다 | 논문은 c ∈ [0,1], ε ∈ [0,1]만 밝힘 |
| **실패 종료** | cost critic을 failure_cost(=1)가 영원히 이어지는 값 1/(1−γ)로 bootstrap | 넘어져서 비용 제약을 피하는 해 방지. 1124 1차 시도가 이 문제로 실패 |
| **보상 자르기** | 종료 페널티를 뺀 스텝 보상을 0 이상으로 자름 (`only_positive_rewards`) | WTW/legged_gym 방식. 음수 보상이 쌓여 일찍 끝내는 것이 이득이 되는 해 방지 |
| PD 게인 | Go2 자산 기본값 hip·thigh 20/1, calf 40/2 (실로봇 실행기와 같음) | 논문 25/0.6은 실기 하드웨어 설정. MuJoCo 이상 액추에이터에서는 다리가 처져 학습이 무너졌다 |
| 보상 | 기존 Go2 보행 보상 + 고정 trot gait 보상 | 논문은 WTW 보상([3])을 쓴다. WTW의 gait 명령(주파수·위상·발 높이 등)은 구현하지 않았다 |
| PID 계수, 조건 MLP [64,64], gait sigmoid 폭 250 iter | 구현 선택 | 논문에 수치 없음 |
| 시뮬레이터 | MuJoCo(mjlab) | 논문은 Isaac Gym |

## 확인할 지표 (tensorboard)

- `Train/mean_episode_length`: 1000(20초)에 가까워야 한다. 1124 1차 시도는 약 9였다.
- `Episode_Termination/*`: time_out이 대부분이어야 한다.
- `Loss/normalized_cost_return`, `Loss/cost_violation`, `Loss/lagrange_multiplier(_max)`: 제약 상태.
- `CNCP/noise_cost`, `CNCP/peak_force`, `CNCP/impact_speed_squared`: 소음 대리 지표.

ONNX 정책의 마지막 입력은 ε다. 실로봇 실행기는 이 입력을 추가해야 이 정책을 쓸 수 있다.
