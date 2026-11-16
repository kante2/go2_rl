# Go2 offline real2sim

실제 NPZ의 LowState와 LowCmd로 MuJoCo 자유 예측 오차를 계산하고, 작은 actuator 파라미터 집합을 식별한다. 실제 로봇 연결, ROS, 정책 추론은 사용하지 않는다.

## 실행

저장소 루트에서 실행한다. Python 3.10 이상과 CPU만 필요하다. 기존 ML 환경 또는 별도 가상환경에 의존성을 설치한다.

```bash
python3 -m pip install -r src/sim_to_real_data/scripts/requirements.txt
python3 src/sim_to_real_data/scripts/run_pipeline.py
python3 -m pytest -q src/sim_to_real_data/scripts/tests
```

기본 입력은 `real/NORMAL_20261019_155919.npz`, 기본 출력은 이 디렉터리의 `outputs/`다. 출력 디렉터리를 다시 사용하면 이전 실행 결과를 덮어쓴다. 입력과 출력은 명시할 수도 있다.

```bash
python3 src/sim_to_real_data/scripts/run_pipeline.py \
  src/sim_to_real_data/real/NORMAL_20261019_155919.npz \
  --output-dir /tmp/go2-real2sim-run --iterations 20 --max-clips 12
```

`--max-clips`는 계산량을 줄이기 위한 학습/validation clip 수 상한이다. test는 모든 clip을 평가한다. `--no-plots`로 그림을 생략할 수 있다.

기본 실행은 0.2초 식별·평가 후 파라미터를 고정한 채 **1초 horizon 진단**도 수행한다. 결과는 `outputs/horizon_1s/`에 저장한다. 긴 horizon에서 오차가 악화될 수도 있으며, 이 결과에 맞춰 자동 재식별하지 않는다.

## 구성과 출력

| 파일 | 역할 |
|---|---|
| `inspect_dataset.py` | 원본 shape, 수신 주기, 중복 tick, clock drift, 명령 갱신율, 관절 움직임 및 발 센서 진단 |
| `prepare_dataset.py` | tick 시간축 복원, quaternion 보간, clip 생성, 시간 블록 분할 |
| `mujoco_replay.py` | 명시적 관절 매핑, 물리 모델 구성, 초기 상태 추정, 자유 예측 엔진 |
| `replay_mujoco.py` | baseline 또는 파라미터 파일 기반 예측 저장 |
| `identify_dynamics.py` | train 목적함수를 최소화하는 CMA-ES 식별 |
| `evaluate_real2sim.py` | 상태 RMSE, 움직임별 오차, 잠정 접촉 이벤트 및 baseline 비교 |
| `plot_real2sim.py` | 관절 궤적과 예측 시간별 오차 그림 |
| `run_pipeline.py` | 전체 절차 실행 |

주요 결과는 `summary.json`, `identified_metrics.md`, `identified_metrics.json`, `identified_parameters.json`, `comparison.png`, `comparison_horizon.png`다. 원본/전처리/예측 파일의 SHA256, 관절 순서, clip ID, 모델과 설정을 함께 남긴다. 생성된 `*.model.xml`은 원본 mesh 디렉터리의 절대 경로를 참조한다.

## 데이터 처리

- 관절 순서는 SDK의 **FR, FL, RR, RL**, 각 다리는 **hip, thigh, calf**다. XML 내부 순서는 이름으로 매핑한다.
- LowState tick을 millisecond로 해석하고 중복 tick은 마지막 값을 유지한다. PC 수신 시각과의 median offset으로 명령 시간축에 매핑한다. clock drift와 수신 잔차를 진단하지만 동작에 맞춰 시간축을 변형하지 않는다. tick reset/wrap이 있는 기록은 먼저 분리해야 한다.
- 기본 상태 간격은 **2ms**, clip 길이 **0.2초**, stride **0.2초**다. 상태는 선형 보간, quaternion은 SLERP, 발 raw 값과 명령은 이전 값을 유지한다. 상태/명령의 **10ms 초과 공백**을 가로지르는 clip은 제외한다. PC 수신 지터와 실제 tick 공백은 구분한다.
- 기본 **1.5초 시간 블록**을 `train/train/train/validation/test` 순서로 반복한다. 학습/검증 창을 무작위로 섞지 않는다. 최대 지연만큼 명령 이력을 각 블록 안에 확보하고, clip은 블록 경계를 넘지 않는다. 인접 블록의 시간적 상관은 남으므로 독립 trial 검증을 대체하지 않는다.
- 접촉 임계값은 train 발 센서의 5/95 percentile에서 발별로 임시 결정한다. hysteresis와 6ms dwell을 적용한다. 이는 센서 보정이 아니다. `--contact-config`로 다음 형식의 JSON을 줄 수 있으며 여전히 검증 전 값으로 표시한다.

```json
{"on": [30, 32, 33, 29], "off": [20, 22, 23, 19], "min_duration_s": 0.006}
```

## Replay 모델과 식별

`src/assets/robots/unitree_go2/xmls/go2.xml`의 기구학·관성을 사용한다. `go2_constants.py`의 armature(0.01/0.01/0.02), torque limit(23.5/23.5/45 Nm), foot friction(0.6), solimp(0.9/0.95/0.023)을 명시적으로 적용한다. 물리 간격은 **1ms**, integrator는 `implicitfast`다. 기존 mjlab CSV의 무작위화 설정이나 정책 상태를 복원하는 실행은 아니다.

각 clip 시작에서만 측정 q/dq/quaternion/gyro를 복사한다. base 높이는 가장 낮은 발 구면이 지면에 닿도록 복원한다. base 선속도는 잠정 지지발의 no-slip 조건에서 추정한다. 둘 다 외부 측정값이 아니며, 낮은 발 우선 초기화는 다른 지지발을 지면에서 띄울 수 있다. 이후 매 스텝 측정 상태를 주입하지 않는다.

quaternion은 wxyz, gyro는 base와 정렬된 IMU의 body frame으로 해석한다. 실제 로봇의 IMU extrinsic과 장착 변경 여부는 추가 확인 대상이다. MuJoCo free joint의 선속도는 world frame, 각속도는 local frame으로 설정한다.

```text
tau = gain[type] * Kp * (q_target - q_sim)
    + damping[type] * Kd * (dq_target - dq_sim)
    + tau_feedforward
```

명령은 `t - delay_s` 이전에 기록된 마지막 값을 사용하고 토크 제한을 적용한다. 실제 `tau_est`를 구동 입력이나 정확한 torque 정답으로 쓰지 않는다. 최적화 변수는 hip/thigh/calf별 gain 3개, damping 배율 3개, 0~40ms 유효 지연으로 기본 7개다. gain 범위는 0.5~1.5, damping은 0.4~1.6이다. `identify_dynamics.py --fit-mass`는 base 질량과 관성을 같은 배율로 변경하는 변수 1개를 추가한다. 접촉 파라미터·무게중심·토크 포화 비선형 모델은 이번 초기 식별에서 고정한다.

목적함수는 q/dq/gyro/회전 오차 각각의 평균 제곱을 `0.1rad / 1rad/s / 0.3rad/s / 0.1rad` 척도로 정규화한 합과 파라미터 정규화 항이다. 회전 오차는 상대 quaternion의 rotvec다. 시작 상태는 오차에서 제외한다. 기본 모델도 후보에 포함해 train 목적함수가 개선되지 않으면 기본값을 유지한다. validation은 전후 진단에만 사용하고 최적 후보는 train으로 선택한다. 기본 seed=7, population=10, iteration=20이다. 이 반복 수는 수렴을 보장하지 않는다.

## 해석 범위

- 단일 22.55초 기록의 짧은 horizon 결과다. 독립 반복 기록과 새로운 움직임에서 재검증해야 한다.
- 유효 gain/감쇠는 미모델링된 동역학과 초기화 오차도 흡수할 수 있다. 실제 모터의 물리적 보정값이라고 해석하거나 실기에 바로 적용하면 안 된다.
- 수신 시각만 있으므로 `delay_s`는 하드웨어 지연이 아니다. 양수 지연만 허용하는 causal replay다. 실제 명령 적용이 PC 수신보다 먼저 일어났다면 그 시간차는 이 변수만으로 재현할 수 없다. 1ms 미만 지연의 수치도 정밀 측정값이 아니다.
- 접촉 이벤트는 threshold와 초기 접촉 복원에 민감하다. clip 시작 이벤트는 제외하며, 창 경계를 넘는 대응은 계산하지 않는다. 짧은 clip의 정착 과정이 extra touchdown을 만들 수 있다. matched/missed/extra를 함께 보고, 현재 timing MAE를 물리적 접촉 정합의 증거로 쓰지 않는다.
- 실제 발 값은 raw counts다. **N 단위 force RMSE는 계산하지 않는다.** 오디오나 접촉력 파형을 학습하지 않으며 동역학 검증 이후의 별도 단계다.

추가로 필요한 것은 로봇/장착물 질량과 위치, 바닥 정보, 독립 반복 보행 기록, 발별 비접촉/정적 접촉 기록이다. base 운동 검증에는 외부 odometry가, 접촉력 peak/impulse 검증에는 동기화된 고대역 힘 측정이 필요하다.

## 개별 실행 예시

아래에서 `S=src/sim_to_real_data/scripts`, `O=/tmp/go2-real2sim-run`으로 두고 실행할 수 있다.

```bash
python3 "$S/inspect_dataset.py" --output "$O/inspection.json"
python3 "$S/prepare_dataset.py" --output "$O/dataset.npz"
python3 "$S/replay_mujoco.py" "$O/dataset.npz" --output "$O/baseline.npz"
python3 "$S/evaluate_real2sim.py" "$O/dataset.npz" "$O/baseline.npz" --output "$O/baseline_metrics.json"
python3 "$S/identify_dynamics.py" "$O/dataset.npz" --output "$O/identified_parameters.json"
python3 "$S/replay_mujoco.py" "$O/dataset.npz" --parameters "$O/identified_parameters.json" --output "$O/identified.npz"
python3 "$S/evaluate_real2sim.py" "$O/dataset.npz" "$O/identified.npz" --output "$O/identified_metrics.json" --compare "$O/baseline_metrics.json"
python3 "$S/plot_real2sim.py" "$O/dataset.npz" "$O/baseline.npz" "$O/identified.npz" --output "$O/comparison.png"
```

새로운 horizon을 평가할 때는 같은 시간 블록 분할을 유지하고 양 모델의 clip·초기화·물리 간격을 동일하게 비교한다. 새로운 test 결과에 맞춰 반복 조정하면 해당 test는 더 이상 미사용 평가가 아니다.

참고: [SPI-Active 논문](https://proceedings.mlr.press/v305/sobanbabu25a.html), [MuJoCo API](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html), 로컬 `../reference/`의 두 PDF.
