# Go2 소음 예측·학습·분석

`src/db_prediction/`은 전처리된 로봇 상태에서 소음 예측 특징을 만들고, 모델을 학습·평가·추론하는 폴더입니다. 접촉·착지 분석, 기존 특징 선정 실험, 실시간 실행 코드도 관리합니다.

원본 NPZ의 시간 정렬과 CSV 생성은 [db_preprocessing](../db_preprocessing/README.md)이 담당합니다. 실행 예시는 저장소 루트 `/home/hajaemin/go2_rl` 기준입니다.

예측 대상은 마이크 PCM의 **20 ms RMS 소음(dBFS)**입니다. 물리적 음압 레벨인 dB SPL과 다르므로 마이크·입력 게인·설치 위치가 달라지면 측정 레벨도 달라집니다.

## 1. 전체 처리 흐름

```text
원본 NPZ: LowState + SportMode 상태 + PCM + 시간 정보
    │
    ▼  db_preprocessing/scripts/
원본 읽기 → 착지/음향 사건 동기화 → 500 Hz 공통 시간축 → 보행 구간 선택
    │
    ├── 녹화별 CSV, 품질 판정, split.json, 동기화 진단
    └── 접촉 이벤트 특징·시계열·원본 PCM 통계
            │
            ├── db_prediction/noise_prediction_model/analysis/
            │      접촉·착지·소음 관계 및 정렬 영향 분석
            │
            ▼  db_prediction/noise_prediction_model/scripts/
       과거 정보로 64차원 특징 생성 → 녹화 단위 CV → 학습 → 모델 고정 → test 평가
            │
            ├── datasets/: 모델 입력·정답·스키마·출처
            ├── models/: 체크포인트와 선택 설정
            └── results/: 예측·지표·그림·검증 보고서
```

실시간 코드의 흐름은 `/lowstate → 이력 버퍼 → 특징 → 모델 → ROS 토픽/CSV/웹 화면`입니다. 다만 현재 `realtime_prediction/`은 이전 joblib 모델 로더를 사용하므로 위 PyTorch 체크포인트와 연결된 상태는 아닙니다. 실제 경로 차이는 마지막 절에 정리했습니다.

## 2. 디렉터리 구조

```text
db_prediction/
├── noise_prediction_model/
│   ├── scripts/                 # 현재 Ridge·HGB·Sparse GPR 파이프라인
│   ├── analysis/                # 접촉·착지·소음 분석
│   ├── experiments/feature_selection/ # 기존 특징 선정·sklearn 실험
│   ├── datasets/                # 모델 입력 배열·메타데이터·스키마
│   ├── models/                  # .pt 및 기존 실험의 .joblib 모델
│   ├── results/                 # 평가 지표·예측·그림·분석 결과
│   ├── runs/1104/               # 별도 1104 실험의 데이터·모델·결과
│   ├── paths.py                 # 공통 결과 경로와 과거 경로 해석
│   ├── features.py              # scripts/features.py로 연결되는 심볼릭 링크
│   ├── lowstate_kinematics.py    # scripts/lowstate_kinematics.py로 연결되는 링크
│   └── requirements.txt          # 현재 학습·분석 의존성
├── preprocessing/               # 예측용 기구학 코드와 URDF
├── realtime_prediction/         # ROS 수신·스트리밍·오디오·웹 화면
├── requirements.txt             # 이전 배포용 NumPy/sklearn/joblib 의존성
└── SHA256SUMS.json              # 이전 배포 파일 무결성 기록
```

`preprocessing/`은 **예측용 기구학 보조 모듈**입니다. 원본 녹화의 동기화와 CSV 생성은 별도 `db_preprocessing/scripts/`가 담당합니다.

`noise_prediction_model/models/`, `scripts/`, `results/`는 나란한 폴더입니다. 모델 파일과 실행 코드는 `results/` 아래에 있지 않습니다.

### 전처리 결과와의 연결

입력 폴더에는 `split.json`과 `{train,test}/<trial>.csv`가 필요합니다. `scripts/features.py`는 `split.json`의 녹화별 CSV 전체를 읽습니다. 전처리의 `all_trials.csv`에 포함되지 않은 동기화 검토 녹화도 사용할 수 있으며 그 상태는 `sync_group` 등으로 기록합니다. 학습 대상의 품질 판정과 범위를 먼저 확인하세요.

## 3. 소음 학습·평가·추론

### 현재 모델의 입력과 정답

현재 파이프라인은 `noise_prediction_model/scripts/`입니다. 입력은 LowState에서 얻는 네 발의 힘, 상대 수직 속도, 힘 변화율, 속도 변화율입니다.

- 입력 차원: **4개 발 × 4종 특징 × 4개 시점 = 64차원**.
- 시점: 현재와 10/20/40 ms 과거. 변화율은 6 ms 후방 차분이므로 최소 46 ms 이력이 필요합니다.
- 특징 순서: `lag → force/velocity/force_change/velocity_change → fr/fl/rr/rl`.
- 발 수직 속도: `[R(ω × p + Jdq)]_z`. 몸체 병진 속도를 더하지 않는 LowState 기반 상대 운동입니다.
- 예측 주기: 20 ms. 정답은 입력 시점에서 **[+10, +30) ms** 구간의 RMS dBFS입니다.
- 녹화 또는 연속 구간의 경계를 가로질러 시간 창을 만들지 않습니다.

SportMode 속도와 오디오 정답은 CSV에 있어도 현재 64차원 모델 입력에는 포함되지 않습니다. 정답 오디오는 학습·평가 때 필요하고 CSV 추론 때는 생략할 수 있습니다. 힘은 원래 센서 단위를 사용하므로 시뮬레이터의 접촉력과 단위가 자동으로 맞춰지지 않습니다.

### 학습 코드별 역할

아래 파일명은 `noise_prediction_model/scripts/` 기준입니다.

| 파일 | 역할 |
|---|---|
| `features.py` | CSV → 물리 특징·과거 창·미래 정답, 연속 구간 분리, 스키마·출처 저장 |
| `lowstate_kinematics.py` | LowState 벡터 구성과 상대 발 수직 속도 계산 |
| `torch_models.py` | Ridge, HGB 트리의 텐서 변환·추론, 체크포인트 저장·로딩 |
| `sparse_gp.py` | 희소 변분 GP 학습, 전체 train 사후분포 계산, 평균·불확실성 추론 |
| `train_evaluate.py` | 녹화 단위 5-fold CV, 설정 선택, 전체 train 재학습, 모델 고정·test 평가 |
| `predict.py` | 단일 녹화 CSV와 `.pt` 모델에서 예측 CSV 생성 |
| `plots.py` | 실측/예측·잔차·녹화별 시계열·모델 비교·불확실성 그림 |
| `verify.py` | 데이터 출처, 분할, 체크포인트 해시, 저장 지표와 분석 결과 검증 |
| `test_pipeline.py` | 인과성·구간 경계·모델 수식·HGB 변환·저장/복원 등 검증 |

| 모델 | 학습과 추론 | 기본 체크포인트 |
|---|---|---|
| Ridge | PyTorch 선형계 풀이로 L2 회귀 학습, 텐서 선형 추론 | `models/01_ridge_regression/model.pt` |
| HGB | sklearn으로 학습한 트리를 PyTorch 텐서로 변환해 추론 | `models/02_hgb/model.pt` |
| Sparse GPR | PyTorch 변분 GP 학습·추론, 평균과 불확실성 출력 | `models/03_sparse_gpr/model.pt` |

설정 선택은 train 녹화 단위 CV의 녹화별 RMSE 평균을 사용합니다. 최종 모델을 고정한 뒤 test를 평가합니다. `train`은 최종 모델의 학습 적합도, `train_oof`는 해당 녹화를 학습하지 않은 fold 모델의 예측, `test`는 보류 녹화 평가입니다. OOF는 설정 선택에도 사용하므로 독립적인 최종 평가와 구분합니다.

`.pt`는 metadata와 `state_dict`를 저장한 파일이며 joblib이나 TorchScript가 아닙니다. `load_bundle()`은 모델과 스키마 등 묶음을, `load_model()`은 모델을 반환합니다. 정규화·특징 순서·단위·시간 창을 학습과 동일하게 유지해야 합니다.

### 모델·데이터·결과 구분

| 경로 | 저장 내용 |
|---|---|
| `datasets/` | `causal_forecast.npz`, 표본 메타데이터, 특징 스키마, CV 녹화 목록, 출처 해시 |
| `models/<모델>/` | `.pt`, 선택 설정, 계수·트리·커널 설명 |
| `models/frozen_manifest.json` | test 평가 전에 고정한 체크포인트 해시 |
| `results/<모델>/` | CV 지표, train/OOF/test 지표, 녹화별 지표, 압축 예측 CSV, 추론 시간, 그림 |
| `results/model_comparison.*` | 모델 간 비교 표·보고서·그림 |
| `results/verification.json`, `results/run.json` | 검증 결과와 실행 상태·설정 |
| `runs/1104/` | 별도 실험의 `datasets/`, `models/`, `results/` 등; 기본 결과와 분리 |

GP 결과에는 잠재 함수 표준편차와 관측 잡음을 포함한 표준편차·예측 구간을 구분해 저장합니다. 추론 시간 보고서는 준비된 입력에 대한 모델 호출 시간을 측정하며 전처리·통신·시뮬레이터 실행 시간은 포함하지 않습니다.

### 분석·기존 실험·실시간 코드

| 경로 | 역할과 현재 학습 코드와의 관계 |
|---|---|
| `analysis/contact_event_correlations.py` | 접촉 이벤트의 힘·속도·소음 상관 분석 |
| `analysis/landing_event_dynamics.py` | 착지 전후 힘·속도·변화율·음향 시계열 분석 |
| `analysis/audit_*.py`, `review_*.py`, `summarize_*.py` 등 | 접촉 시각·정렬 영향·검출 범위·상관·분포 점검과 보고서 생성 |
| `analysis/study_1104.py` | train 탐색 분석과 모델 고정 후 평가를 분리한 1104 실험 |
| `experiments/feature_selection/` | 기존 후보 특징 선정과 sklearn/joblib 모델 비교; 현재 64차원 모델과 입력·가중치 정책이 다름 |
| `realtime_prediction/stream_model.py` | 상태 재표본화·이력·특징·예측; 현재 이전 joblib 번들 로더 사용 |
| `realtime_prediction/predict_live.py` | ROS `/lowstate` 수신 또는 NPZ 재생, 예측 토픽·CSV·웹 연결 |
| `realtime_prediction/remote_audio.py` | SSH/ALSA PCM 수신과 실측 소음 계산 |
| `realtime_prediction/web_dashboard*.py`, `dashboard*` | HTTP 서버·브라우저 화면, 실제 모델 없이 실행하는 데모 |
| `realtime_prediction/predict_live_v2.py`, `run_live*.sh` | 화면 버전 선택 및 실행 래퍼; 이전 경로가 남아 있음 |

착지 이후 최대값이나 centered 변화율을 쓰는 회고 분석은 미래 정보가 들어갈 수 있습니다. 이를 현재 예측 모델의 과거 정보 기반 입력과 혼동하지 않아야 합니다. 기존 실험의 `train_feature_optimal.py`는 필요한 `train_op.py`가 없어 기본 실행 대상으로 사용하지 않습니다.

## 4. 실행 예시

현재 학습·분석 의존성은 다음 파일에 있습니다. ROS 실시간 실행에는 별도로 ROS 2, `rclpy`, `unitree_go`, `std_msgs`와 DDS 설정이 필요합니다.

```bash
python3 -m pip install -r src/db_prediction/noise_prediction_model/requirements.txt
```

### 학습·평가와 CSV 추론

학습에는 `split.json`과 해당 녹화별 CSV가 필요합니다. 아래처럼 `--source`를 명시하면 예측 코드의 과거 기본 입력 경로를 사용하지 않습니다. 학습은 지정한 출력의 모델·결과를 갱신합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 src/db_prediction/noise_prediction_model/scripts/train_evaluate.py \
  --source src/db_preprocessing/preprocessed/1104 \
  --output src/db_prediction/noise_prediction_model/runs/1104

python3 src/db_prediction/noise_prediction_model/scripts/verify.py \
  --source src/db_preprocessing/preprocessed/1104 \
  --output src/db_prediction/noise_prediction_model/runs/1104

python3 src/db_prediction/noise_prediction_model/scripts/predict.py \
  --input /absolute/path/to/one_trial.csv \
  --model hgb \
  --model-path src/db_prediction/noise_prediction_model/runs/1104/models/02_hgb/model.pt \
  --output /tmp/noise_predictions.csv
```

`predict.py`의 입력 CSV 하나는 녹화 하나여야 하고 500 Hz 시간축과 필요한 LowState 열을 가져야 합니다. 기본 모델은 `sparse_gpr`이며 기존 출력 파일을 교체하려면 `--overwrite`가 필요합니다.

모델 없이 웹 화면만 확인하는 명령은 다음과 같습니다. 이 데모는 실제 센서 수신이나 추론을 하지 않습니다.

```bash
python3 src/db_prediction/realtime_prediction/web_dashboard.py --demo --port 8080
```

## 5. 현재 경로 연결 시 확인할 사항

현재 폴더를 읽어 확인한 경로 차이입니다. 이 문서 작성 과정에서 아래 코드나 데이터 배치를 변경하지는 않았습니다.

| 위치 | 코드의 현재 참조와 필요한 조치 |
|---|---|
| 모델 학습 입력 | 기본값은 `src/db_prediction/preprocessed/1007`입니다. 실제 전처리 결과와 연결하려면 `--source`를 명시합니다. `study_1104.py`도 기본 입력이 `db_prediction/preprocessed/1104`입니다. |
| 실시간 실행 래퍼 | `run_live*.sh`는 이전 `Prediction/predict_live*.py`와 `contact_force_velocity/.../all_four.joblib`을 참조합니다. 현재 폴더에 맞춘 연결이 필요합니다. |
| 실시간 모델 로더 | `stream_model.py`는 이전 `contact_force_velocity`의 기구학 모듈과 joblib 번들을 요구합니다. 현재 `.pt`를 `--model`로 지정하는 것만으로는 호환되지 않습니다. |
| 과거 문서·manifest | 이전 프로젝트 루트와 파일 해시가 남아 있을 수 있습니다. 현재 경로와 코드 인자를 우선 확인하고, `SHA256SUMS.json`은 새 폴더 전체의 최신 manifest로 간주하지 않습니다. |

상세 알고리즘과 실험 설명은 [전처리 문서](../db_preprocessing/README.md), [모델 문서](noise_prediction_model/README.md), [접촉·착지 분석](noise_prediction_model/analysis/README.md), [기존 특징 선정 실험](noise_prediction_model/experiments/feature_selection/README.md)에 있습니다. 하위 문서의 이전 실행 경로는 위 표와 대조해야 합니다.
