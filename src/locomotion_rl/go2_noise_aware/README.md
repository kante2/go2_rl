# Go2 NoiseAware

학습된 소음 모델(DiffImpact 앙상블)로 MuJoCo 착지마다 소음을 추정해, 저소음 보행을 학습하는 패키지입니다. `go2_quietpaw`와 같은 구조(보상/비용 분리, quietness 조건, PPO-Lagrangian)를 쓰고, **소음 추정만** QuietPaw의 물리량 대리식에서 실측 녹음으로 학습한 모델로 바꿉니다.

`noise_model/`의 출력은 **정책 스텝마다 환경별 소음 dB 하나**입니다. 이 값을 비용(보상)으로 바꾸는 식은 아직 정하지 않았고, 이 모듈에 넣지 않습니다.

## 현재 상태

| 단계 | 상태 |
|---|---|
| `noise_model/`: 소음 예측 모델 + MuJoCo 입력 모듈 | **완료, 검증됨** |
| `step_01_environment/`: 환경에 착지 추적 연결 (QuietPaw 환경 상속) | 예정 |
| `step_02_learning/`: 소음 dB → 비용 설계, quietness → cost_limit 재조정 | 예정 (재설계) |
| `step_03_training/`, `step_04_evaluation/`: QuietPaw 재사용 | 예정 |

## 소음 모델 (`noise_model/`)

| 파일 | 역할 |
|---|---|
| `diffimpact_ensemble.pt` (+ `.json`) | **물리 법칙 모델** 3개(시드): 다리별 상수(m, τ₀, A₀)·K 표 (`DEFAULT_TABLE`). α = 1, β = 0.2 이론값 고정 |
| `diffimpact.py` | `DiffImpactNoiseModel`: 착지 특징 → 발별 A-가중 에너지·dB |
| `touchdown_features.py` | `TouchdownFeatureTracker`: 물리 substep 상태 → 착지 특징 (학습 데이터와 같은 정의). `DiffImpactStepNoise`: 추적 + 모델 → 정책 스텝별 소음 dB (env용 래퍼) |

### 무엇을 예측하나

착지 한 번(발 i)마다 **착지 후 0~80 ms A-가중 기대 에너지**를 계산합니다. 배경소음은 빠집니다.

```
E_i = A_i² · K_다리(Δ_i, τ_i)          파형 합성 없이 표 보간 (A = 충격량 = 접촉력 펄스 넓이, τ = 접촉 시간)
앙상블: E = 모델 3개의 E_i 선형 평균,  dB = 10·log10(E)
```

충격 크기 A와 접촉 시간 τ는 충돌 물리 법칙으로 계산합니다.

```
접촉 속도   v = max( −(foot_vz_pre + body_vz_pre), 0.001 )              [m/s]  바닥 기준 발 하강 속도, 클리핑 없음
충격 크기   A = √( (m_다리 · (v/0.5))² + A0_다리² )                      충격량-운동량 J = m v (1+e) → A ∝ v (α = 1)
                                                                    + 하중이 실리는 소리 A0 (에너지 합)
접촉 시간   τ = clamp( τ0_다리 · (v/0.5)^(−0.2), 0.3 ms, 20 ms )         Hertz t_c = 2.87 (m²/(R E*² v))^(1/5) (β = 0.2)
            τ0_다리 = 0.3 ms + 19.7 ms · sigmoid(τraw_다리)
```

| 항목 | 내용 |
|---|---|
| 고정 (이론) | 지수 α = 1 (충격량 ∝ v), β = 0.2 (Hertz 접촉 시간 ∝ v^−1/5) |
| 학습 (다리별, 실측 녹음) | log m (유효 질량·반발계수·마이크 감도 묶음), τ0 (0.5 m/s 접촉 시간: 유효 질량·발끝 반지름·탄성계수 묶음), A0 (하중 소리) → 12개 |
| 입력 | 특징 12개 중 `foot_vz_pre`, `body_vz_pre` 두 개만 계산에 씀. 나머지는 학습 범위 밖 진단(`ood`, `ood_frac`)용 |
| 근사 | 접촉력 파형 sin^1.5 (Hunter–Reed), K 표 보간 (원본 합성과 0.12 dB 이내) |

검증 (학습에 안 쓴 실측 녹화 4개 / sport mode 녹화 29개, 시드 3개 앙상블):

| 지표 | 값 |
|---|---|
| 실측 검증: 레벨 MAE / 녹화 내 순위 상관 | 1.01 dB / 0.36 |
| MuJoCo 착지 적용: 녹화별 중앙값 상관 | 0.92 |
| sport mode (다른 보행, 재학습 없음): MAE / 편향 / 녹화 내 순위 / 속도별 중앙값 상관 | 2.58 / −2.3 dB / 0.48 / 0.60 |
| sport mode, 접촉 속도를 자르지 않을 때 최대 과대 예측 | +6.8 dB (폭주 없음) |
| α, β 를 다리별로 학습한 모델과 비교 | 모든 지표 ±0.1 이내 → 이론값 고정 |

- **한계:**
  - 같은 접촉 속도의 착지는 다리별로 같은 값을 냅니다. 그래서 녹화 안에서 착지별 순위를 잘 가리지 못합니다(0.36).
  - sport mode로 따로 학습하면 소리 크기와 속도의 관계(하중 소리 A0, 학습한 α)가 크게 달랐습니다. 소리 크기와 접촉 속도의 관계는 보행(제어기)에 따라 달라집니다.
  - 접촉 속도 → 0에서는 하중 소리 A0에서 멈춥니다. A0가 작은 다리는 더 낮게 내려갑니다.

- **단위:** dB re FS²입니다. 녹음 디지털 풀스케일 기준이고 마이크 보정이 없는 **상대값**입니다. 실로봇 RL 보행 착지 한 발이 대략 -73 ~ -57 dB입니다. 절대 dBA는 UMIK-1 교정 상수를 더하면 됩니다.
- **Δ:** 세트(대각 쌍) 첫 착지 기준 이 발의 착지 시각입니다. 모델 소리가 힘 센서 착지보다 몇 ms 먼저 시작해서, 첫 발은 창 앞부분이 잘리기 때문에 필요합니다.
- **원본 모델과의 일치:** 로컬 검증 스크립트로 확인했습니다. 세트 dB 중앙값 차이가 실로봇·MuJoCo·sport mode에서 −0.12 ~ +0.02 dB, 상관 0.994 이상입니다.
- **속도:** 노트북 RTX 2060에서 착지 16k개 추론이 1.35 ms입니다. CPU 로봇 1대 기준 정책 스텝당 착지 추적 5.9 ms + 추론 1.0 ms입니다.

### 입력: MuJoCo에서 넘길 값

물리 substep(5 ms)마다 `tracker.sample(...)`에 넘깁니다. **발·관절은 Unitree 순서(FR, FL, RR, RL)**입니다.

| 인자 | 모양 | MuJoCo (mjlab) 값 |
|---|---|---|
| `foot_vel_rel_w` | (B, 4, 3) | `site_lin_vel_w[:, 발]` − `root_link_lin_vel_w[:, None]` (월드) |
| `foot_pos_b` | (B, 4, 3) | R_baseᵀ(`site_pos_w[:, 발]` − `root_link_pos_w`) (몸체 좌표) |
| `contact_force` | (B, 4) | 발 접촉 센서 힘 크기 (착지·입각 판정, > 1 N) |
| `joint_pos` | (B, 12) | 관절 각, `unitree_joint_order(joint_names)`로 재배열 |
| `joint_torque` | (B, 12) | 액추에이터 토크, 같은 순서로 재배열 |
| `base_quat_w` | (B, 4) | `root_link_quat_w` (wxyz) |
| `command` | (B, 3) | 속도 명령 (vx, vy, wz) |
| `base_vz_w` | (B,) | 선택. `body_vz_source="true"`일 때만 |

- **발 site:** `go2.xml`의 `FR`, `FL`, `RR`, `RL` site(calf + 0.213 m)는 학습 특징의 URDF 발 위치와 같은 점입니다. baseline처럼 `site_names=("FR","FL","RR","RL")`, `preserve_order=True`로 읽으면 Unitree 순서가 됩니다.
- **관절 순서 주의:** `go2.xml` 관절 순서는 FL, FR, RL, RR입니다. 그대로 넘기면 다리가 뒤바뀌므로 `unitree_joint_order`로 재배열해야 합니다.
- **mjlab 속성 이름:** 표의 이름은 baseline·QuietPaw 코드에서 쓰는 것 기준입니다. 관절 토크 속성 이름은 mjlab 1.2.0에서 확인이 필요합니다.

### 특징을 어떻게 만드나 (학습 데이터와 같은 정의)

모듈 안에서 아래 12개 특징을 학습 때와 **같은 식**으로 만듭니다. 학습 데이터는 실로봇 200 Hz였습니다.

| 특징 | 계산 |
|---|---|
| foot_v{x,y,z}_pre | 몸 기준 발 속도, 착지 전 -1 ~ -10 ms 보간 평균 = 200 Hz 샘플 (-10, -5, 0 ms)에 가중치 (0.3, 0.5, 0.2) |
| foot_{x,y,z} | 착지 순간 몸체 좌표 발 위치 |
| cmd_{vx,vy,wz} | 착지 순간 명령 |
| body_vz_pre | **반대 대각 쌍 입각 발 운동학 추정** −평균(몸 기준 발 vz). MuJoCo 정답이 있어도 학습과 같은 추정식을 씀 |
| log_grf_impulse20 | **관절 토크 기반 접촉력** F = −J⁻ᵀτ의 월드 z, 착지 후 0~20 ms에서 착지 전 평균을 뺀 적분(가중치 3,5,5,5,3 ms)의 로그 |
| is_second | 대각 짝이 80 ms 안쪽 앞에 먼저 닿음 |

학습과 같은 추정식을 쓰는 이유는, MuJoCo 정답과 추정값이 다르기 때문입니다. 녹화된 MuJoCo 세션으로 확인한 차이는 이렇습니다.
- 충격량: 접촉 센서 정답이 토크 추정보다 약 2.4배 큽니다.
- 몸통 vz: 중앙값 0.02 m/s 차이가 나고, 이 때문에 예측이 -0.3 dB 치우칩니다.

모델은 추정값으로 학습됐으므로 같은 추정값을 넣어야 합니다.

- **착지 판정:** 접촉력이 1 N을 넘는 순간입니다(QuietPaw와 같음). reset 직후 이미 닿아 있는 발도 첫 접촉으로 셉니다.
- **확정 시점:** 착지 특징은 **착지 후 4 substep(20 ms) 뒤**에 확정됩니다. 충격량 때문입니다. 정책 스텝(20 ms) 기준으로 최대 1스텝 늦게 소음 값이 나옵니다.
- **검증:** 로컬 검증 스크립트로 MuJoCo 녹화 세션 20개를 substep 단위로 재생했습니다.
  - 학습 데이터 착지 928개가 모두 같은 시각에 검출됐습니다.
  - 특징 12개의 중앙값 차이가 0입니다(몸통 vz 상관 0.95, 나머지 1.00).
  - 예측 dB 차이는 95%가 0.04 dB 안입니다.

### 출력: 정책 스텝 소음 (에너지 합 → dB)

```
스텝 에너지 = Σ_i E_i        (그 정책 스텝에 확정된 착지들, 발별 앙상블 에너지의 선형 합)
스텝 dB    = 10·log10(스텝 에너지)     착지가 없으면 no_touchdown_db (기본 -inf)
```

`end_step()` 반환 dict (모두 환경별):

| 키 | 모양 | 의미 |
|---|---|---|
| `db` | (B,) | 스텝 소음 [dB re FS², A-가중, 0~80 ms, 배경 제외] |
| `energy` | (B,) | 스텝 에너지 [FS²], 착지 없으면 0 |
| `touchdown` | (B,) bool | 이번 스텝에 확정된 착지가 있는지 |
| `foot_energy`, `foot_db` | (B, 4) | 발별 값 (확정 안 된 발은 0, -inf) |
| `ood` | (B,) | 확정 착지의 속도 특징 범위 밖 거리 합 [σ] (진단값) |
| `log` | dict | 배치 요약: 착지 수, `step_db`(착지 있는 스텝 평균), 앙상블 dB 표준편차, 특징별 범위 밖 비율 |

- **dB를 더하지 않고 에너지를 더합니다.** −62 dB 두 발 = −59 dB입니다. 대각 두 발이 같은 스텝에 확정되면 학습 정답(세트 0~80 ms 소리)과 같은 정의가 됩니다. 두 발 시간차가 커서 다른 스텝에 확정되면 각 스텝에 나뉘어 들어갑니다.
- **착지가 없는 스텝은 -inf입니다.** 정책 스텝의 약 93%가 이렇습니다(트롯 0.6 s 주기). 보상에 쓸 때는 `touchdown`으로 거르거나 `energy`를 쓰거나 `no_touchdown_db`를 유한값으로 지정하세요.
- **입력 범위:** 접촉 속도는 클리핑하지 않습니다(물리 법칙으로 외삽). 학습 범위 밖 정도는 `ood`(속도 특징 거리 합), `log`(특징별 비율)로 볼 수 있습니다.
- **비용 설계 참고값:** `model.db_ref`에 학습 실측 착지의 발별 예측 dB 백분위가 있습니다(p1 −78.1, p5 −68.3, p50 −63.2, p95 −58.9, p99 −57.5).

### 사용 예 (env step 안)

```python
from src.locomotion_rl.go2_noise_aware.noise_model import DiffImpactStepNoise, unitree_joint_order

# load_managers
self.noise = DiffImpactStepNoise(self.num_envs, self.device)
self.unitree_j = unitree_joint_order(robot.joint_names)

# step
self.noise.begin_step()
for _ in range(self.cfg.decimation):
  ...  # apply_action, sim.step, scene.update
  d = robot.data
  feet = self.foot_site_ids                                  # ("FR","FL","RR","RL"), preserve_order
  rel_v = d.site_lin_vel_w[:, feet] - d.root_link_lin_vel_w[:, None]
  pos_b = quat_rotate_inverse(d.root_link_quat_w, d.site_pos_w[:, feet] - d.root_link_pos_w[:, None])
  self.noise.sample(rel_v, pos_b, contact_force_norm, d.joint_pos[:, self.unitree_j],
                    joint_torque[:, self.unitree_j], d.root_link_quat_w, command)
noise = self.noise.end_step()
noise["db"], noise["touchdown"]                             # (B,) 스텝 소음 dB, 착지 여부

# _reset_idx
self.noise.reset(env_ids)
```

`noise["log"]`에는 확정된 착지 수, 착지 있는 스텝의 평균 dB, 앙상블 모델 간 dB 표준편차, 특징별 범위 밖 비율이 들어 있습니다.

### 한계

- **접촉력을 따로 쓰지 않습니다.** 충격량은 접촉 속도로부터 충격량-운동량 관계로 계산합니다. 실측 데이터에서 토크 기반 충격량과 발 하강 속도의 상관이 0.82라 따로 구분할 정보가 없었고, MuJoCo 접촉력은 실제와 2배 이상 다릅니다.
- **착지 이벤트만 봅니다.** 발 끌기, 미끄러짐, 입각기 소리는 예측하지 않으므로 보행 보상 쪽 제약(발 들기 높이, 미끄러짐 페널티)이 필요합니다.
- **보행 방식이 크게 바뀌면 절대 레벨이 어긋납니다.** sport mode에서 순위는 유지됐지만(녹화 내 상관 0.48) 레벨은 2.3 dB 낮았습니다. 튐·재접촉 소리는 모델에 없습니다.
- **앙상블은 시드만 다릅니다.** 모델 간 표준편차가 불확실성을 잘 나타내지 못합니다. 불확실성을 비용에 반영하려면 녹화 단위 부트스트랩 앙상블이 필요합니다.

## 테스트

```bash
python3 -m unittest src.locomotion_rl.go2_noise_aware.tests.test_noise_model   # 저장소 루트, docker (mjlab 필요)
```

단위 테스트(13개)는 물리 법칙 모델(이론 지수, 접촉 속도 = 발 + 몸통, 범위 밖 단조 증가, Hertz 접촉 시간 비율, 하중 소리 하한, 다른 특징 무관), 스텝 소음(에너지 합 → dB), 다리 기구학, 토크 기반 접촉력, 착지 시점·세트 판정을 검사합니다. `src/locomotion_rl/__init__.py`가 mjlab 태스크를 등록하므로 mjlab이 있는 환경에서 실행합니다.

## 모델 출처

`diffimpact_ensemble.pt`는 실로봇 녹음으로 학습한 물리 법칙 DiffImpact 모델 3개(`w100_200hz_body_physfix_1band_sign_s0~s2`, α = 1, β = 0.2 고정)를 RL용 표로 내보낸 파일입니다. 이전의 1차식 모델(특징 12개 회귀)과 α·β 다리별 학습 모델은 제거했습니다(git 기록에 남아 있음). 이 저장소에는 **모델 파일과 예측 코드만** 올립니다. 학습·내보내기·검증 코드(`src/diffimpact/`의 `train_diffimpact.py`, `export_noise_model.py`, `validate_noise_model.py`, `validate_tracker.py`)와 원본 체크포인트는 로컬에 보관합니다. 위에 적은 일치 검증 수치는 그 로컬 스크립트로 얻은 값입니다.
