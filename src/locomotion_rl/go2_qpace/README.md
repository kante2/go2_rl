# Go2 QPace

QuietPaw(CNCP) 구조로 **목표 소음 dB에 맞춰 걷는 정책**을 학습한다. 소음 비용은 실측 녹음으로 학습한
DiffImpact 착지 소음 예측 모델에서 만든다. 다른 locomotion 패키지를 import하지 않는 독립 패키지다.

태스크: `Unitree-Go2-QPace-Flat`, `Unitree-Go2-QPace-Rough`
로그·체크포인트: `logs/rsl_rl/go2_qpace/<날짜>_<run-name>/`

## 실행

```bash
# 학습: 환경마다 목표 dB를 [min_db, max_db]에 균등 배정해 한 정책이 전 범위를 배운다.
bash scripts/run_go2_docker.sh python scripts/train.py Unitree-Go2-QPace-Flat \
  --env.scene.num-envs 2048 --agent.max-iterations 10000 --agent.save-interval 100 \
  --agent.logger tensorboard --agent.run-name qpace_flat

# 관찰: 목표 dB 지정 (학습 범위 안). 브라우저 http://localhost:8080
bash scripts/run_go2_docker.sh python scripts/play.py Unitree-Go2-QPace-Flat \
  --checkpoint-file logs/rsl_rl/go2_qpace/RUN/model_9999.pt --viewer viser --target-db -60

# 평가: 목표 dB × 속도 → 실제 로봇 dB, 초과량, 추종 오차, Pareto (CSV)
bash scripts/run_go2_docker.sh python -m src.locomotion_rl.go2_qpace.step_04_evaluation.evaluate \
  --checkpoint-file logs/rsl_rl/go2_qpace/RUN/model_9999.pt

# 테스트
bash scripts/run_go2_docker.sh env QPACE_GPU_TESTS=1 MUJOCO_GL=egl python -m unittest \
  src.locomotion_rl.go2_qpace.tests.test_cncp src.locomotion_rl.go2_qpace.tests.test_noise_model -v
```

## QuietPaw 대비 바뀐 것

| | QuietPaw (논문) | QPace |
|---|---|---|
| 착지 비용 c_t | 식 (1) 대리 비용 (접촉력·착지 속도), [0, 1] | **소음 모델이 예측한 로봇 전체 착지 에너지** E_t / (Δt·P_ref), ≥ 0 |
| 한도 | ε ∈ [0, 1] (단위 없음) | **목표 dB** [min_db, max_db] |
| 정책·critic 조건 입력 | ε | (목표 dB − min_db)/(max_db − min_db) |
| 제약 | (1−γ)Σγᵗc_t ≤ ε | **L_robot = 10·log10((1−γ)Σγᵗc_t) + reference_db ≤ 목표 dB** |
| λ PID 오차 | 비용 − ε | **L_robot − 목표 dB** (dB 차이) |
| cost critic | 대리 비용의 할인 합 | 구조 동일, 소음 에너지 비율의 할인 합을 예측 |
| actor, reward critic, PPO, 식 (6)(8) 분해·결합 | | 동일 |

### 비용 계산 (`step_01_environment/cncp_env.py`)

```
물리 substep(5 ms)마다: 발 속도·위치, 접촉력, 관절각, 액추에이터 토크, 자세, 명령 → 착지 추적기
착지 후 20 ms에 그 착지의 특징 12개 확정 → 앙상블 모델 → 발별 에너지 E_i (A-가중 0~80 ms)
정책 스텝(20 ms) 끝: E_t = Σ 이번 스텝에 확정된 착지 E_i   (로봇 전체, 착지 없으면 0)
c_t = E_t / (Δt · 10^(reference_db/10))
```

- 에너지를 쓰는 이유: 시간에 따라 더해야 하는 비용(할인 합, GAE)에는 선형 에너지가 맞다. dB는 더할 수 없고
  착지가 없는 스텝에서 −inf가 된다. 한도·PID·로그는 dB로 다루며, dB 비교와 에너지 비교는 같은 결과다.
- 발·관절 순서: 소음 모델은 Unitree 순서(FR, FL, RR, RL). 접촉 센서(geom 순서), 관절(FL, FR, RL, RR),
  액추에이터(관절 종류별)를 각각 재배열한다. 토크는 mjlab 1.2에 `joint_torques`가 없어 `actuator_force`를 쓴다.
- 실패 종료: 학습 범위 상단 + `failure_margin_db`(3 dB) 파워가 영원히 이어진다고 보고 cost critic을 bootstrap한다.
- 몸통 vz 외삽 상한: 원본 모델은 몸통이 위로 움직이며 착지하면 무한히 조용하다고 외삽한다(+0.2 m/s에서 약 10 dB).
  학습 범위 끝 + `body_vz_extrapolation`(0.05 m/s)에서 자른다.
- 착지가 아닌 소리(모터, 발 끌기, 미끄러짐)는 포함하지 않는다. 기존 `foot_slip`·`foot_clearance` 보상이 맡는다.

## dB 단위와 범위

- 단위: **dB re FS²/s, A-가중, 미보정 상대값**. 소음 모델 출력(녹음 풀스케일 기준)을 시간으로 나눈 로봇 소음 파워.
- 기본 범위 `min_db=-64`, `max_db=-49`, `reference_db=-56`: 모델 db_ref(학습 실측 착지의 발별 1/50/99 백분위
  −72.6/−63.7/−56.9 dB)에 걸음 주기 0.6 s의 착지 빈도(초당 6.7회, +8.2 dB)를 더한 추정값. 실제 정책으로 측정해 조정한다.
- 실제 방의 dB(A)로 쓰려면 보정 상수 C(마이크 교정 + 시뮬-실로봇 차이)가 필요하다.
  `src/sim_to_real_data/real_sim_straight`의 실측-시뮬 쌍으로 측정할 수 있다.

## 설정 (CLI)

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--env.noise-condition.min-db / max-db` | −64 / −49 | 학습 목표 dB 범위 |
| `--env.noise-condition.reference-db` | −56 | 비용 정규화 기준 |
| `--env.noise-cost.body-vz-extrapolation` | 0.05 | 몸통 vz 외삽 상한 [m/s] |
| `--env.noise-cost.failure-margin-db` | 3 | 넘어짐 비용 (max_db + 이 값) |
| `--agent.algorithm.pid.max-multiplier` | 100 | PID λ 절대 상한 (커리큘럼 `lambda_max`가 이보다 작으면 그 값) |

## 단계 학습 (`step_02_learning/curriculum.py`의 `QPaceCurriculumCfg`)

사전학습 보행(기본: 실로봇에 배포된 `src/go2_autonomous_driving/model_pt/model_10000.pt`)에서 시작한다.

| 구간 (PPO iteration) | actor | λ | 옵션 (기본값) |
|---|---|---|---|
| 0 ~ critic_warmup | **고정** (critic만 학습) | 0 | `--env.qpace-curriculum.critic-warmup-iterations` (100) |
| ~ noise_start | 학습 | 0 (소음 제약 없음) | `--env.qpace-curriculum.noise-start-iteration` (300) |
| ~ noise_start + ramp | 학습 | PID, 상한 0 → lambda_max 선형 증가 | `--env.qpace-curriculum.lambda-ramp-iterations` (500) |
| 이후 | 학습 | PID, 상한 lambda_max | `--env.qpace-curriculum.lambda-max` (20, 낮추면 속도 우선) |

- 가중치 이식 (`step_03_training/warm_start.py`): actor 47→48 입력 (조건 열 가중치 0 → 시작 시 baseline과 같은 행동),
  baseline critic의 정규화와 앞 3층을 reward·cost critic의 상태 인코더에 복사. optimizer·λ는 새로 시작.
- 처음부터 학습: `--env.qpace-curriculum.init-checkpoint ""` 와 함께 `noise-start-iteration`을 보행이 자리 잡는 시점(예: 2000)으로 늦춘다.
- 재개(`--agent.resume`)할 때는 이식하지 않고 체크포인트를 그대로 읽는다. iteration도 이어서 센다.
- 속도 명령 단계 `VELOCITY_STAGES`: 처음 vx −0.5~0.7 m/s (소음 모델이 아는 범위), 4000회 이후 −0.7~1.0 m/s.
- 보행 리듬 보상 감소 `GaitCurriculumCfg`: 2000회 중심 sigmoid (논문).

## 확인할 지표 (tensorboard)

- `Train/mean_episode_length`: 1000에 가까워야 한다.
- `QPace/robot_power_db`, `QPace/target_db`: 배치 평균 로봇 소음과 목표.
- `Loss/noise_excess_db`(L_robot − 목표), `Loss/noise_violation_db`, `Loss/noise_satisfied_frac`, `Loss/lagrange_multiplier(_max)`.
- `QPace/ood_frac/*`: 소음 모델 학습 범위 밖 비율. 계속 커지면 모델을 속이는 보행일 수 있다.
- `Metrics/twist/error_vel_xy`: 속도 추종. mjlab 정의상 에피소드 동안 |오차|/400을 누적한 값이라 m/s가 아니다
  (1000 step 에피소드면 평균 오차 × 2.5). 같은 지표로 baseline `model_10000` 학습 끝값은 1.54.
- `Loss/lambda_cap`, `Loss/actor_frozen`: 단계 학습 진행 상태.

## 폴더 구조

| 위치 | 역할 |
|---|---|
| `step_01_environment/cncp_env.py` | 소음 모델 연결, 비용 c_t, 보상 자르기, reset 전 관측 |
| `step_02_learning/noise_model/` | DiffImpact 소음 예측 모델 (go2_noise_aware 0044f61에서 복사, `extrapolation_limits`만 추가) |
| `step_02_learning/costs.py` | 비용 설정 |
| `step_02_learning/mdp/noise_condition.py` | 목표 dB 조건 |
| `step_02_learning/mdp/costs.py` | 착지 진단값 (최대 접촉력) |
| `step_03_training/cncp_ppo.py` | dB PID, 실패 bootstrap, 식 (7)~(10) |
| `step_03_training/critics.py` | V_f(s, 조건) = ξ_f(s)ᵀ w_f(조건), reward·cost 독립 |
| `step_04_evaluation/{play,evaluate}.py` | `--target-db` 관찰, 목표 dB × 속도 평가 |

## 한계

- 소음 모델은 명령 속도 0.7 m/s를 넘으면 그 값을 0.7로 잘라 쓴다. 빠른 속도의 소음 증가를 모른다.
- 목표 dB가 낮고 속도가 빠르면 둘 다 지킬 수 없다. 이때 λ가 커져 소음을 우선하고 속도 추종이 나빠진다.
- 소음 모델은 녹음한 바닥 하나 기준이다. 바닥이 다르면 보정이 필요하다.
