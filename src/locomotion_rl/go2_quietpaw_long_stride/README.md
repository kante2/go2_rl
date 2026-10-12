# Go2 QuietPaw Long Stride

현재 작업 중인 `go2_quietpaw`를 복사한 긴 보폭 실험용 태스크다.
소음 제약 PPO는 유지하면서 보행 주기와 보행/달리기 자세 허용 폭을 조정한다.
원본 태스크의 기본 설정과 로그 경로는 그대로 유지한다.

## 보폭 조절

`step_02_learning/stride.py`의 `StrideCfg`가 아래 값을 관리한다.
환경 생성 직전에 CLI 설정을 보상과 Actor/Critic 관측에 함께 반영한다.
하위 `foot_gait.period`, `phase.period`, thigh/calf 자세 허용 폭은 이 설정으로
덮어쓰므로 해당 값을 바꿀 때는 `--env.stride.*`를 사용한다.

| 인자 | 기본값 | 의미 |
|---|---:|---|
| `--env.stride.gait-period` | 0.70 | 같은 발의 착지 간 목표 주기 [s] |
| `--env.stride.thigh-pose-std` | 0.45 | 보행/달리기 thigh 자세 보상의 허용 폭 [rad] |
| `--env.stride.calf-pose-std` | 0.60 | 보행/달리기 calf 자세 보상의 허용 폭 [rad] |
| `--env.gait-curriculum.minimum-scale` | 0.20 | 학습 후반에 남길 gait 보상의 비율 |

정지 자세 허용 폭, hip 허용 폭, 관절 한계, 행동 배율, 발 높이 목표는 원본과 같다.
고정 리듬을 후반에도 일부 유지하므로 quietness에 따른 자유로운 보행 적응과
상충할 수 있다. 이 값들은 실험 시작값이며 보폭 향상을 보장하지 않는다.
새로운 보폭 보상이나 Actor 입력은 추가하지 않았다. Flat 관측은 여전히 48/75차원이다.

저장소 루트의 mjlab 환경에서 새 학습을 시작한다.

```bash
python scripts/train.py Unitree-Go2-QuietPaw-LongStride-Flat \
  --env.scene.num-envs 1024 \
  --env.stride.gait-period 0.70 \
  --env.stride.thigh-pose-std 0.45 \
  --env.stride.calf-pose-std 0.60 \
  --env.gait-curriculum.minimum-scale 0.2 \
  --agent.run-name long_stride_v1
```

호스트의 conda `(base)`에는 학습 의존성이 없으므로, 호스트에서 시작할 때는
위 명령의 `python scripts/train.py`를 `bash scripts/run_go2_training.sh`로
바꾼다. 이 스크립트가 기존 학습 컨테이너에서 실행한다.
공통 `scripts/train.py`와 `scripts/play.py`는 직접 실행 시 저장소 루트를
자동으로 Python 경로에 추가한다.
험지 태스크는 `Unitree-Go2-QuietPaw-LongStride-Rough`다.
로그는 `logs/rsl_rl/go2_quietpaw_long_stride`에 저장한다.

체크포인트에 `stride` 설정을 저장하며 play/evaluate에서 학습 때의 주기와 자세
설정을 복원한다. ONNX metadata에는 `stride_gait_period` [s]를 기록한다.
외부 실행기도 이 주기로 phase 입력을 생성해야 한다.
일반 resume은 stride 설정도 같아야 한다. 원본 QuietPaw 체크포인트에는 이 설정이
없으므로 이 복제본의 평가/재개 체크포인트로 사용하지 않는다.

## 보폭 측정

물리 substep에서 같은 발의 연속 착지점 사이 월드 수평 거리와 경과 시간을
측정한다. 직선 이동 명령(`|v_xy| > 0.1 m/s`, `|wz| < 0.2 rad/s`) 구간만
포함하며 첫 착지, reset을 가로지르는 쌍, 정지/회전 구간을 가로지르는 쌍은 제외한다.
거리는 수평 변위의 크기이며 목표 방향 투영이나 발의 스윙 궤적 길이가 아니다.
접촉력 1 N 전환을 사용하므로 접촉 튐이나 미끄러짐도 함께 살펴봐야 한다.

- TensorBoard `LongStride/stride_length_m`, `LongStride/stride_period_s`:
  현재 유효한 발별 최근 완료 주기의 평균. `LongStride/measured_feet`로 표본 유무 확인.
- 평가 CSV `stride_length_m`, `stride_period_s`, `stride_samples`:
  평가 구간의 모든 유효 착지 쌍을 같은 비중으로 평균. 표본이 없으면 평균 칸은 비워둔다.

같은 목표 속도와 quietness에서 속도 추종 오차, 소음 비용, 실패 횟수를 함께 비교한다.

아래는 복제한 QuietPaw 학습 구조와 실행 설명이다.

`go2_baseline`의 환경·학습·평가 구조를 기반으로 만든 QuietPaw/CNCP의
MuJoCo(mjlab) 이식 구현이다. 기존 baseline 파일은 수정하지 않고, 공통
`scripts/train.py`, `scripts/play.py`에서 태스크 이름에 따라 실행기를 선택한다.

# command
bash /home/kante/go2_rl/scripts/enter_go2_docker.sh

cd /workspace/unitree_rl_mjlab

python scripts/train.py Unitree-Go2-QuietPaw-LongStride-Flat \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.save-interval 100 \
  --agent.logger tensorboard \
  --agent.run-name quietpaw_flat \
  --video True \
  --video-interval 2400 \
  --video-length 200 \
  --gpu-ids '[0]'

`--gpu-ids`는 리스트 형식(`'[0]'`)으로 지정한다. 생략해도 기본값 `[0]`을 사용한다.

## 파일을 읽는 순서

| 위치 | 역할 |
|---|---|
| `registry.py` | Flat/Rough 환경과 QuietPaw Runner 등록 |
| `step_01_environment/base_env.py` | 환경·관측·보상·비용 설정 조립 |
| `step_01_environment/go2_env.py` | baseline의 Go2와 지형별 설정 |
| `step_01_environment/quietpaw_env.py` | 물리 substep의 착지 측정, reset 전 비용·관측 전달 |
| `step_02_learning/rewards.py`, `mdp/rewards.py` | 보행 보상 설정과 계산 |
| `step_02_learning/costs.py`, `mdp/costs.py` | 소음 비용 설정과 계산 |
| `step_02_learning/mdp/noise_condition.py` | quietness → 비용 한계 변환 |
| `step_02_learning/observations.py` | 마지막 scalar로 비용 한계 추가 |
| `step_02_learning/curriculum.py`, `mdp/curriculums.py` | 고정 gait 보상 감소 |
| `step_03_training/ppo.py` | 네트워크·PPO·PID 하이퍼파라미터 |
| `step_03_training/critics.py` | 독립적인 Reward/Cost 분해 critic |
| `step_03_training/storage.py` | Reward/Cost GAE, rollout 저장·minibatch |
| `step_03_training/pid_lagrangian.py` | 환경별 Lagrange multiplier |
| `step_03_training/ppo_lagrangian.py` | 두 advantage 결합과 PPO 업데이트 |
| `step_03_training/runner.py` | 저장·재개·ONNX export |
| `step_03_training/train.py` | 학습 CLI |
| `step_04_evaluation/play.py` | 특정 quietness로 관찰 |
| `step_04_evaluation/evaluate.py` | 조건별 정량 평가와 CSV 출력 |
| `tests/test_learning.py` | 비용·접촉 전환·GAE·PID·업데이트 검증 |

Actor는 baseline의 RSL-RL `MLPModel`을 그대로 활용한다. 별도의 actor 구현은
필요하지 않다. Flat 관측은 Actor 48차원, Critic 75차원이며 마지막 값이
`cost_limit`이다. Rough에는 기존 height scan이 추가된다. 행동은 기존과 같은
12개 관절 위치 offset이다.

## 학습 신호

- `reward`: baseline의 속도 추종·자세·안정성 보상. 기존처럼 정책 dt를 곱한다.
- `noise_cost`: 아래 식으로 계산하는 `[0, 1]` 값. dt를 곱하지 않는다.
- `cost_limit`: rollout의 **정책 스텝 평균 raw cost**에 적용하는 한계.
- `quietness`: 사용자 입력 `[0, 1]`. 클수록 비용 한계가 작아진다.

기본 변환은 `cost_limit = 0.35 - quietness * (0.35 - 0.02)`이다.
학습 중에는 환경별 quietness를 0~1에 균등 배정하고 reset 때 유지한다.
환경이 하나면 중간 조건을 사용한다. 따라서 여러 조건을 학습하려면 병렬 환경을
여러 개 사용해야 한다. 배포·평가에서는 원하는 quietness를 지정한다.

비용 계산은 다음과 같다. `mean_feet`는 네 발 평균이다.

```text
force_cost    = 1 - exp(-max(peak_force - 50 N, 0) / 50 N)
velocity_cost = 1 - exp(-impact_vertical_speed² / 0.5 m²/s²)
noise_cost    = mean_feet(0.5 * force_cost + 0.5 * velocity_cost)
```

`peak_force`는 정책 한 스텝에 포함된 물리 substep들의 최대 접촉력이다.
속도 항은 접촉이 시작되는 substep 직전의 월드 z 하강 속도를 사용한다.
지속 접촉은 반복 착지로 세지 않는다. 넘어짐 등 실제 실패 종료에는 비용 1을
적용한다. 비용을 보상에 중복 반영하지 않도록 기존 `soft_landing` 가중치는 0이다.

두 critic은 각각 `V(s, ε) = ξ(s)ᵀ w(ε)`를 계산한다. 상태 인코더에 ε가
들어가지 않으며, 기본 특징 차원은 64다. PPO advantage는
`(A_reward - λ * A_cost) / (1 + λ)`이고, 결합한 뒤 정규화한다.
PID는 환경별 rollout 평균 비용과 한계의 차이로 λ를 갱신한다.

timeout은 reset **이전** 관측의 reward/cost value로 bootstrap한다.
실제 실패 종료에는 bootstrap하지 않는다. checkpoint에는 두 critic,
Actor, optimizer, PID 이력, 학습률, 환경 공통 스텝 수가 저장된다.

## 논문과 구분되는 구현 선택

분석 기준: 저장소의 `quietpaw_.pdf`, arXiv:2503.05035v1.

1. PDF 식 (1)의 부호와 본문 설명이 불일치하여 증가형 `1-exp(-x)`를 사용했다.
   위 힘 threshold와 비용 한계는 조정 가능한 시작값이며 논문의 확정 수치가 아니다.
2. 논문에 부족한 비용 누적 단위는 rollout 스텝 평균으로 명확히 정했다.
   critic은 할인 return을 예측하지만 PID 한계는 할인 return과 비교하지 않는다.
3. PID 계수, 조건 MLP `[64, 64]`, gait sigmoid 폭 250 iterations는 구현 선택이다.
   gait 보상은 기본 2,000 iterations를 중심으로 줄어든다.
4. Isaac Gym/Walk These Ways 대신 baseline의 MuJoCo, 보상, 관측과 PD gain을 유지한다.
5. 발 속도는 월드 수직 방향 근사다. 험지의 정확한 접촉 법선이나 음향을 모델링하지 않는다.
6. 초기 버전은 단일 학습 프로세스/GPU를 지원한다. RND·대칭 증강·순환 정책과
   관측 history 적층·delay는 지원하지 않는다.
7. 실제 dB 감소나 제약 만족은 보장되지 않는다. 짧은 실행 검증과 충분한 학습 후의
   보행·소음 성능 검증은 별도로 수행해야 한다.

## 실행

저장소의 mjlab/RSL-RL 환경 또는 `go2-rl:local` 컨테이너 안에서 실행한다.
명령의 작업 디렉터리는 저장소 루트다.

```bash
# 평지 학습: 현재 GPU 메모리에 맞게 환경 수를 조정한다.
python scripts/train.py Unitree-Go2-QuietPaw-LongStride-Flat \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard

# 험지 학습
python scripts/train.py Unitree-Go2-QuietPaw-LongStride-Rough \
  --env.scene.num-envs 1024

# 짧은 연결 확인. 보행 성능을 평가할 수 있는 학습량은 아니다.
python scripts/train.py Unitree-Go2-QuietPaw-LongStride-Flat \
  --env.scene.num-envs 8 \
  --agent.max-iterations 2 \
  --agent.num-steps-per-env 4 \
  --agent.algorithm.num-mini-batches 2 \
  --agent.algorithm.num-learning-epochs 1

# 조용함 조건을 지정한 실행
python scripts/play.py Unitree-Go2-QuietPaw-LongStride-Flat \
  --checkpoint-file logs/rsl_rl/go2_quietpaw_long_stride/RUN/model_1000.pt \
  --quietness 0.8

# 속도·조건별 비용, 속도 오차, 충격, 실패 횟수와 Pareto 여부를 CSV로 저장
python -m src.locomotion_rl.go2_quietpaw_long_stride.step_04_evaluation.evaluate \
  --checkpoint-file logs/rsl_rl/go2_quietpaw_long_stride/RUN/model_1000.pt \
  --num-envs 32 \
  --quietness '(0.0,0.5,1.0)' \
  --velocities '(1.0,1.5)' \
  --output logs/quietpaw_evaluation.csv

# 물리 시뮬레이션 없이 CPU에서 실행되는 학습 구성요소 테스트
python -m unittest discover -s src/locomotion_rl/go2_quietpaw_long_stride/tests -v

# GPU 통합 테스트도 포함: timeout의 실제 관측, 저장·재개, ONNX 출력, play 경로
QUIETPAW_GPU_TESTS=1 MUJOCO_GL=egl \
  python -m unittest discover -s src/locomotion_rl/go2_quietpaw_long_stride/tests -v
```

학습 재개는 baseline처럼 `--agent.resume True --agent.load-run RUN
--agent.load-checkpoint model_1000.pt`를 사용한다. 환경 수, 비용·조건 설정,
제어 주기, gait curriculum은 저장 시점과 같아야 한다. 네트워크 크기를 바꾸어
학습했다면 재개 CLI에도 같은 설정을 준다. 평가에서는 저장된 모델 설정과
비용 범위를 자동으로 읽고 Actor만 복원하므로 환경 수를 바꿀 수 있다.

각 run에는 `params/env.yaml`, `params/agent.yaml`, `model_*.pt`,
`policy.onnx`가 저장된다. ONNX의 마지막 관측은 quietness가 아니라 위 식으로
변환한 **cost_limit**이다. 47차원 baseline 실로봇 실행기는 이 입력을 추가하기
전까지 QuietPaw 정책과 호환되지 않는다.
