# Go2 NoiseAware

학습된 소음 모델(DiffImpact 앙상블)로 MuJoCo 착지마다 소음을 추정해, 저소음 보행을 학습하는 패키지입니다. `go2_quietpaw`와 같은 구조(보상/비용 분리, quietness 조건, PPO-Lagrangian)를 쓰고, **소음 비용 함수만** QuietPaw의 물리량 대리식에서 실측 녹음으로 학습한 모델로 바꿉니다.

## 현재 상태

| 단계 | 상태 |
|---|---|
| `noise_model/`: 소음 예측 모델 + MuJoCo 입력 모듈 | **완료, 검증됨** |
| `step_01_environment/`: 환경에 착지 추적 연결 (QuietPaw 환경 상속) | 예정 |
| `step_02_learning/`: 비용 설정, quietness → cost_limit 재조정 | 예정 |
| `step_03_training/`, `step_04_evaluation/`: QuietPaw 재사용 | 예정 |

## 소음 모델 (`noise_model/`)

| 파일 | 역할 |
|---|---|
| `diffimpact_ensemble.pt` (+ `.json`) | 모델 3개(시드)의 계수·정규화·입력 범위·K 표, 213 KB (아래 "모델 출처") |
| `diffimpact.py` | `DiffImpactNoiseModel`: 착지 특징 → A-가중 에너지. `touchdown_cost`: 에너지 → 비용 |
| `touchdown_features.py` | `TouchdownFeatureTracker`: 물리 substep 상태 → 착지 특징 (학습 데이터와 같은 정의). `DiffImpactNoiseCost`: 추적 + 모델 + 비용을 묶은 env용 래퍼 |

### 무엇을 예측하나

착지 한 번(발 i)마다 **착지 후 0~80 ms A-가중 기대 에너지**를 계산합니다. 배경소음은 빠집니다.

```
E_i = A_i² · K_다리(Δ_i, τ_i)                     파형 합성 없이 표 보간
log A_i = w_다리 · z_i + b_다리                    z_i = (clip(x_i) − μ) / σ
τ_i = 0.3 ms + 19.7 ms · sigmoid(c_다리 + k_다리 · z_i[발 vz])   접촉 시간, 빠른 하강일수록 짧음
앙상블: E = 모델 3개의 E_i 선형 평균,  dB = 10·log10(E)
```

- **단위:** dB re FS²입니다. 녹음 디지털 풀스케일 기준이고 마이크 보정이 없는 **상대값**입니다. 실로봇 RL 보행 착지 한 발이 대략 -73 ~ -57 dB입니다. 절대 dBA는 UMIK-1 교정 상수를 더하면 됩니다.
- **Δ:** 세트(대각 쌍) 첫 착지 기준 이 발의 착지 시각입니다. 모델 소리가 힘 센서 착지보다 몇 ms 먼저 시작해서, 첫 발은 창 앞부분이 잘리기 때문에 필요합니다.
- **원본 모델과의 일치:** 로컬 검증 스크립트로 확인했습니다. 세트 dB 중앙값 차이가 실로봇·MuJoCo·sport mode에서 +0.05 ~ +0.11 dB, 상관 0.997 이상입니다.
- **속도:** RTX 5060 Ti에서 착지 4,096개에 1.7 ms입니다.

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
- **확정 시점:** 착지 특징은 **착지 후 4 substep(20 ms) 뒤**에 확정됩니다. 충격량 때문입니다. 정책 스텝(20 ms) 기준으로 최대 1스텝 늦게 비용이 들어옵니다.
- **검증:** 로컬 검증 스크립트로 MuJoCo 녹화 세션 20개를 substep 단위로 재생했습니다.
  - 학습 데이터 착지 928개가 모두 같은 시각에 검출됐습니다.
  - 특징 12개의 중앙값 차이가 0입니다(몸통 vz 상관 0.95, 나머지 1.00).
  - 예측 dB 차이는 95%가 0.04 dB 안입니다.

### 비용

```
cost_착지 = max(dB − floor, 0) / (ref − floor)  +  ood_weight · Σ_속도특징 |x − clip(x)| / σ
스텝 비용 = 그 정책 스텝에 확정된 착지들의 cost 합 (착지 없으면 0)
```

| 설정 (`DiffImpactCostCfg`) | 기본값 | 의미 |
|---|---|---|
| `floor_db` | 학습 실측 발별 예측의 1 백분위 (-72.6 dB) | 이보다 조용해도 추가 이득 없음 (모델이 모르는 영역 추구 방지) |
| `ref_db` | 50 백분위 (-63.7 dB) | (ref − floor)가 비용 1 단위 |
| `ood_weight` | 0.1 | 범위 밖 거리 [σ]당 비용. 범위 안으로 돌아오는 신호 |
| `failure_cost` | 1.0 | 넘어짐 종료 |

- **상한은 두지 않습니다.** 시끄러울수록 비용이 계속 커져서, 줄이는 방향의 신호가 끊기지 않습니다.
- **입력 클리핑:** 특징을 학습 범위(1~99 백분위)로 자른 뒤 모델에 넣습니다. 범위 밖에서 직선 외삽으로 폭주하는 것을 막기 위해서입니다. sport mode 데이터에서 클리핑 없이 최대 20 dB 과대 예측한 적이 있습니다.
- **범위 밖 페널티는 속도 특징(발 vx·vy·vz, 몸통 vz)에만 겁니다.** 같은 정책에서도 MuJoCo 착지 발 높이가 실로봇보다 약 3 cm 낮아 시뮬 착지의 72%가 foot_z 범위 밖입니다. 여기에 페널티를 걸면 소음과 무관한 자세 변경을 유도하기 때문입니다. 모든 특징의 범위 밖 비율은 감시용으로 로그에 남깁니다.
- **기준 비용 수준:** 착지당 비용의 5/50/95%는 실로봇 착지 0.22 / 0.93 / 1.66, MuJoCo 착지 0.57 / 1.21 / 1.87입니다. QuietPaw cost_limit 범위는 이 수준과 착지 빈도에 맞춰 다시 정해야 합니다.

### 사용 예 (env step 안)

```python
from src.locomotion_rl.go2_noise_aware.noise_model import DiffImpactNoiseCost, unitree_joint_order

# load_managers
self.noise = DiffImpactNoiseCost(self.num_envs, self.device)
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
noise_cost, log = self.noise.end_step()                     # (B,), 진단값

# _reset_idx
self.noise.reset(env_ids)
```

로그(`log`)에는 확정된 착지 수, 평균 예측 dB, 앙상블 모델 간 dB 표준편차, 특징별 범위 밖 비율이 들어 있습니다.

### 한계

- **접촉력 효과가 거의 없습니다.** 학습 데이터에서 충격량과 발 하강 속도가 함께 움직여(상관 0.82) 충격량 가중치가 대부분 0입니다. 소음은 주로 발 하강 속도와 몸통 낙하 속도로 결정됩니다.
- **착지 이벤트만 봅니다.** 발 끌기, 미끄러짐, 입각기 소리는 예측하지 않으므로 보행 보상 쪽 제약(발 들기 높이, 미끄러짐 페널티)이 필요합니다.
- **보행 방식이 크게 바뀌면 절대 레벨이 어긋납니다.** sport mode에서 순위는 맞지만(녹화 내 상관 0.57) 레벨은 3 dB 낮았습니다. 튐·재접촉 소리는 모델에 없습니다.
- **앙상블은 시드만 다릅니다.** 모델 간 표준편차가 불확실성을 잘 나타내지 못합니다. 불확실성 기반 비관적 비용을 쓰려면 녹화 단위 부트스트랩 앙상블이 필요합니다.

## 테스트

```bash
python3 -m unittest src.locomotion_rl.go2_noise_aware.tests.test_noise_model   # 저장소 루트, docker (mjlab 필요)
```

단위 테스트(11개)는 방향, 클리핑, 비용 경계, 다리 기구학, 토크 기반 접촉력, 착지 시점·세트 판정을 검사합니다. `src/locomotion_rl/__init__.py`가 mjlab 태스크를 등록하므로 mjlab이 있는 환경에서 실행합니다.

## 모델 출처

`diffimpact_ensemble.pt`는 실로봇 녹음으로 학습한 DiffImpact 모델 3개(`w100_200hz_body_tau_1band_sign_s0~s2`)를 RL용 표로 내보낸 파일입니다. 이 저장소에는 **모델 파일과 예측 코드만** 올립니다. 학습·내보내기·검증 코드(`src/diffimpact/`의 `train_diffimpact.py`, `export_noise_model.py`, `validate_noise_model.py`, `validate_tracker.py`)와 원본 체크포인트는 로컬에 보관합니다. 위에 적은 일치 검증 수치는 그 로컬 스크립트로 얻은 값입니다.
