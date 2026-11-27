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
  --checkpoint-file logs/rsl_rl/go2_qpace/RUN/model_9999.pt --viewer viser --target-db -52

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
착지(학습 데이터 규칙) 후 35 ms에 특징 12개 확정 → 앙상블 모델 → 발별 에너지 E_i (A-가중 0~80 ms)
탈락 접촉(유각기 < 60 ms 또는 입각기 < 40 ms): 탈락한 substep에 그 접촉 속도로 E_i 계산 (세트 첫 발 기준)
정책 스텝(20 ms) 끝: E_t = Σ 이번 스텝의 착지 E_i + 탈락 접촉 E_i   (로봇 전체, 없으면 0)
c_t = E_t / (Δt · 10^(reference_db/10))
```

- **탈락 접촉도 비용에 넣는 이유** (`count_rejected_contacts`, 기본 켬): 착지 규칙은 소음 모델 학습 데이터에서
  정상 착지만 골라낸 기준이다. RL에서 이 규칙 밖 접촉을 0으로 두면 정책이 발을 빠르게 두드려 비용을 지운다.
  2026-11-27 학습(model_10000 시작, 탈락 접촉 미포함)에서 확인: 800회째 접촉 초당 26회 중 89%가 탈락,
  행동 노이즈 std 0.39 → 0.84, 속도 추종 악화. 물리 법칙 모델은 접촉 속도만 쓰므로 짧은 접촉에도 쓸 수 있다.

- 에너지를 쓰는 이유: 시간에 따라 더해야 하는 비용(할인 합, GAE)에는 선형 에너지가 맞다. dB는 더할 수 없고
  착지가 없는 스텝에서 −inf가 된다. 한도·PID·로그는 dB로 다루며, dB 비교와 에너지 비교는 같은 결과다.
- 발·관절 순서: 소음 모델은 Unitree 순서(FR, FL, RR, RL). 접촉 센서(geom 순서), 관절(FL, FR, RL, RR),
  액추에이터(관절 종류별)를 각각 재배열한다. 토크는 mjlab 1.2에 `joint_torques`가 없어 `actuator_force`를 쓴다.
- 실패 종료: 학습 범위 상단 + `failure_margin_db`(3 dB) 파워가 영원히 이어진다고 보고 cost critic을 bootstrap한다.
- 소음 모델 (go2_noise_aware 8d4270d, 물리 법칙): 접촉 속도 v = −(발 vz + 몸통 vz)로
  충격 크기 A = √((m·v/0.5)² + A0²), 접촉 시간 τ = τ0·(v/0.5)^−0.2 (지수는 이론값 고정, 다리별 m·τ0·A0만 실측 학습).
  접촉 속도는 클리핑하지 않는다 (빠른 착지는 계속 시끄러워짐).
- 발별 하한 `foot_floor_db` (기본: 모델 db_ref p1 = −76.2 dB): v → 0 이면 A0가 작은 RL 다리가 약 −106 dB까지 내려간다.
  학습 데이터 밖이라 이 값 아래는 같은 비용으로 본다. 하한에 걸린 착지 비율은 `QPace/floored_frac`.
- 착지가 아닌 소리(모터, 발 끌기, 미끄러짐)는 포함하지 않는다. 기존 `foot_slip`·`foot_clearance` 보상이 맡는다.

## dB 단위와 범위

- 단위: **dB re FS²/s, A-가중, 미보정 상대값**. 소음 모델 출력(녹음 풀스케일 기준)을 시간으로 나눈 로봇 소음 파워.
- 기본 범위 `min_db=-58`, `max_db=-45`, `reference_db=-52.5`: **측정값 기준** (2026-11-27, 탈락 접촉 포함).
  baseline 보행(model_34997, actor 고정, 2048 env, 60 iter)의 로봇 소음 파워가 −50.1 dB(착지 초당 6.1회 + 탈락 접촉
  5.0회, 탈락 접촉의 에너지 몫 6%), 불확실성 반영 비용(k = 1)을 켜면 −49.3 dB. baseline 대비 약 −8 ~ +5 dB가 학습 범위.
  하한 −60 → −58: 측정 하한(약 −62.5 dB)에 너무 가깝고, −60 목표에서 λ 가 상한에 붙은 채 보행이 무너졌다.
  탈락 접촉은 대부분 입각 중 힘이 잠깐 빠졌다 돌아오는 느린 접촉이라 29%가 발별 하한에 걸린다.
- 실제 음압: 소음 모델의 착지 한 발 `db_spl = db + 118` (잠정 dBA SPL, 마이크 위치, 0~80 ms 평균, ±1~3 dB).
  로봇 소음 파워(초당 합)를 방의 dBA 한도로 쓰려면 기준 시간(예: 1초 Leq)을 정해 변환식을 따로 정해야 한다.
- MuJoCo 착지는 실측보다 빠르다 (평균 접촉 속도 0.86 m/s, 소음 모델 README 기준 시뮬 적용 편향 +1.4 dB).

## 설정 (CLI)

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--env.noise-condition.min-db / max-db` | −58 / −45 | 학습 목표 dB 범위 |
| `--env.noise-condition.reference-db` | −52.5 | 비용 정규화 기준 |
| `--env.noise-cost.foot-floor-db` | None (= db_ref p1, −76.2) | 착지 한 발 소음 하한 [dB] |
| `--env.noise-cost.count-rejected-contacts` | True | 착지 규칙에서 빠진 짧은 접촉도 비용에 넣음 |
| `--env.noise-cost.failure-margin-db` | 3 | 넘어짐 비용 (max_db + 이 값) |
| `--env.noise-cost.uncertainty-k` | 1.0 | 접촉 소음 = 앙상블 평균 + k·표준편차 [dB] (0 이면 평균만) |
| `--env.noise-cost.uncertainty-max-db` | 3.0 | 불확실성 가산 상한 [dB] |
| `--agent.algorithm.anchor-coef` | 1.0 | baseline 보행 앵커 세기 (0 이면 끔) |
| `--agent.algorithm.max-std` | 0.5 | 행동 표준편차 상한 |
| `--env.qpace-curriculum.target-gate-tracking-error` | 0.20 | 목표 ramp 게이트 [m/s] (None 이면 끔) |
| `--agent.algorithm.pid.max-multiplier` | 100 | PID λ 절대 상한 (커리큘럼 `lambda_max`가 이보다 작으면 그 값) |

## 단계 학습 (`step_02_learning/curriculum.py`의 `QPaceCurriculumCfg`)

사전학습 보행(기본: `src/go2_autonomous_driving/model_pt/model_34997.pt`)에서 시작한다.
실로봇 기본 보행이자 소음 모델 학습 녹음에 쓴 정책이라, 시작 시점의 소음 예측이 가장 믿을 만하다.
다른 보행에서 시작하려면 `--env.qpace-curriculum.init-checkpoint <경로>` (같은 47/74 입력 baseline 체크포인트).

| 구간 (PPO iteration) | actor | λ | 옵션 (기본값) |
|---|---|---|---|
| 0 ~ critic_warmup | **고정** (critic만 학습) | 0 | `--env.qpace-curriculum.critic-warmup-iterations` (100) |
| ~ noise_start | 학습 | 0 (소음 제약 없음) | `--env.qpace-curriculum.noise-start-iteration` (300) |
| ~ noise_start + ramp | 학습 | PID, 상한 0 → lambda_max 선형 증가 | `--env.qpace-curriculum.lambda-ramp-iterations` (2000) |
| 이후 | 학습 | PID, 상한 lambda_max | `--env.qpace-curriculum.lambda-max` (5, 낮추면 속도 우선) |

**목표 dB 범위 커리큘럼** (`target_start_min_db` −52, `target_ramp_iterations` 3000): 가장 엄격한 목표를
baseline 근처(−52 dB)에서 시작해 noise_start부터 3000회 동안 `min_db`(−60)까지 선형으로 내린다.
환경별 목표 = 현재 하한 + u_i·(max_db − 현재 하한), u_i는 환경마다 고정. 정책 입력 정규화는 전체 범위 기준이라
목표 dB의 의미는 바뀌지 않는다. 목표는 rollout 사이에만 바뀐다. 현재 하한은 `Loss/target_min_db`.

- 왜 넣었나 (2026-11-27 확인 학습, model_34997 시작, 탈락 접촉 포함, 목표 −60 ~ −45 처음부터, λ 상한 500회에 20):
  정책이 목표 조건을 배우기 전에 엄격한 목표 환경의 압력으로 모든 환경이 "거의 안 걷기"로 무너졌다
  (500회째 목표와 상관없이 약 −61 dB, 1.0 m/s 명령에 추종 오차 0.8 m/s). 그래서 λ 상한도 20 → 5,
  ramp 500 → 2000회로 낮추고, 엔트로피 계수를 0.01 → 0.005로 줄였다 (행동 노이즈 증가 억제).

- 가중치 이식 (`step_03_training/warm_start.py`): actor 47→48 입력 (조건 열 가중치 0 → 시작 시 baseline과 같은 행동),
  baseline critic의 정규화와 앞 3층을 reward·cost critic의 상태 인코더에 복사. optimizer·λ는 새로 시작.
- 처음부터 학습: `--env.qpace-curriculum.init-checkpoint ""` 와 함께 `noise-start-iteration`을 보행이 자리 잡는 시점(예: 2000)으로 늦춘다.
- 재개(`--agent.resume`)할 때는 이식하지 않고 체크포인트를 그대로 읽는다. iteration도 이어서 센다.
- 속도 명령 단계 `VELOCITY_STAGES`: 처음 vx −0.5~0.7 m/s (소음 모델이 아는 범위), 5000회 이후 −0.7~1.0 m/s
  (목표 ramp 가 끝난 뒤. 소음 압박과 겹치지 않게).
- 보행 리듬 보상 `GaitCurriculumCfg`: **끝까지 유지** (`minimum_scale = 1`). 논문의 2000회 sigmoid 감소는 처음부터 학습할 때용.

### C 방식: 상한 + 여유 활용 (2026-11-27 붕괴 대책)

제약은 그대로 **로봇 소음 ≤ 목표 dB** 이다 (λ ≥ 0, 일부러 시끄럽게 걷는 법은 배우지 않는다).
남는 여유를 baseline 보행 유지에 쓰게 해서 소음이 목표 근처에 머물게 한다.

| 장치 | 위치 | 하는 일 |
|---|---|---|
| baseline 앵커 | `cncp_ppo.py` (`anchor_coef`) | actor 손실 + anchor_coef·Σ(μ − μ_base)². μ_base 는 warm start 직후 actor 의 고정 복사본(model_34997). λ = 0 인 느슨한 환경은 baseline 보행에 머물고, 엄격한 환경만 λ 가 미는 만큼 벗어난다. baseline 은 소음 데이터를 녹음한 보행이라 모델의 데이터 범위 안에 묶는 효과도 있다. 지표 `Loss/anchor_mse`. |
| 보행 리듬 유지 | `GaitCurriculumCfg.minimum_scale = 1` | trot 리듬 보상을 끄지 않는다. |
| 불확실성 반영 비용 | `DiffImpactStepNoise` (`uncertainty_k`, 상한 3 dB) | 접촉 소음 = 평균 + k·앙상블 표준편차. 모델이 모르는 영역은 시끄럽다고 본다. |
| 표준편차 상한 | `cncp_ppo.py` (`max_std` 0.5) | 행동 노이즈가 커져 발 떨기 해로 가는 것을 막는다. |
| 목표 ramp 게이트 | `runner.py` (`target_gate_tracking_error` 0.20 m/s) | rollout 평균 추종 오차가 한계를 넘으면 그 iteration 은 목표를 더 내리지 않는다. 지표 `Loss/target_progress`. |

이전 학습(`qpace_34997_full`, 2026-11-27)이 무너진 과정: 리듬 보상이 사라진 2500회부터 trot 이 풀림 → 목표 −58 ~ −60 dB 압박
→ 접촉 초당 11 → 19.5회(탈락 72%), 행동 표준편차 0.30 → 1.40, 소음 모델 데이터 밖 접촉 25 → 95%, 앙상블 σ 0.8 → 2.3 dB
→ 속도 추종 붕괴 (평균 보상 57 → 0). 무너지기 전 쓸 만한 체크포인트는 2000 ~ 2300회.

## 확인할 지표 (tensorboard)

- `Train/mean_episode_length`: 1000에 가까워야 한다.
- `QPace/robot_power_db`, `QPace/target_db`: 배치 평균 로봇 소음과 목표.
- `Loss/noise_excess_db`(L_robot − 목표), `Loss/noise_violation_db`, `Loss/noise_satisfied_frac`, `Loss/lagrange_multiplier(_max)`.
- `QPace/ood_frac/*`: 소음 모델 학습 범위 밖 비율 (진단값, 예측에는 접촉 속도만 쓴다).
- `QPace/contacts_per_s`, `QPace/touchdowns_per_s`: 모든 접촉 / 규칙에 맞는 착지 (baseline 11.1 / 6.1).
- `QPace/rejected_frac`, `QPace/rejected_energy_frac`: 탈락 접촉 비율과 소음 에너지 몫 (baseline 45% / 6%).
  접촉 수와 탈락 비율이 같이 크게 오르면 두드리기 보행이다.
- `QPace/floored_frac`: 발별 하한에 걸린 접촉 비율 (baseline 29%, 대부분 느린 탈락 접촉).
- `QPace/contact_speed`: 모든 접촉의 평균 접촉 속도 [m/s] (baseline 0.54). 소음이 줄 때 이 값이 같이 줄어야 정상.
- `Loss/entropy`: 행동 노이즈. 계속 오르면 정책이 불안정하다 (탈락 접촉 미포함 학습에서 3 → 14).
- `QPace/tracking_error_mps`: 속도 추종 오차 [m/s] (baseline 0.166, 목표 게이트 0.20).
- `Loss/anchor_mse`, `Loss/target_progress`, `Policy/mean_std`: baseline 이탈 정도, 게이트를 통과한 목표 진행, 행동 표준편차.
- 학습 모니터: `python src/locomotion_rl/go2_qpace/step_04_evaluation/monitor.py` → http://localhost:6007
- `Metrics/twist/error_vel_xy`: 속도 추종. mjlab 정의상 에피소드 동안 |오차|/400을 누적한 값이라 m/s가 아니다
  (1000 step 에피소드면 평균 오차 × 2.5). 같은 지표로 baseline `model_10000` 학습 끝값은 1.54.
- `Loss/lambda_cap`, `Loss/actor_frozen`: 단계 학습 진행 상태.

## 폴더 구조

| 위치 | 역할 |
|---|---|
| `step_01_environment/cncp_env.py` | 소음 모델 연결, 비용 c_t, 보상 자르기, reset 전 관측 |
| `step_02_learning/noise_model/` | DiffImpact 소음 예측 모델 (모델·표는 go2_noise_aware 8d4270d 그대로, 착지 판정은 학습 데이터 규칙, 발별 하한·탈락 접촉 비용 추가) |
| `step_02_learning/costs.py` | 비용 설정 |
| `step_02_learning/mdp/noise_condition.py` | 목표 dB 조건 |
| `step_02_learning/mdp/costs.py` | 착지 진단값 (최대 접촉력) |
| `step_03_training/cncp_ppo.py` | dB PID, 실패 bootstrap, 식 (7)~(10) |
| `step_03_training/critics.py` | V_f(s, 조건) = ξ_f(s)ᵀ w_f(조건), reward·cost 독립 |
| `step_04_evaluation/{play,evaluate}.py` | `--target-db` 관찰, 목표 dB × 속도 평가 |

## 한계

- 소음 모델은 접촉 속도만 본다. 같은 접촉 속도면 다리별로 같은 값이고, 발 끌기·튐·재접촉·입각기 소리는 모른다.
- 조용한 쪽(접촉 속도 약 0.5 m/s 아래)은 실측 녹음의 배경 소음 바닥(약 −62.5 dB) 아래라 물리식 외삽에 기댄다.
- 목표 dB가 낮고 속도가 빠르면 둘 다 지킬 수 없다. 이때 λ가 커져 소음을 우선하고 속도 추종이 나빠진다.
- 소음 모델은 녹음한 바닥 하나 기준이다. 바닥이 다르면 보정이 필요하다.
