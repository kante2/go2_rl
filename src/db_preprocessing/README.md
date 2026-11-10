# Go2 소음 데이터 전처리

`src/db_preprocessing/`은 로봇 상태와 녹음 데이터를 시간 정렬하고 학습·분석용 CSV를 생성하는 폴더입니다. LowState·SportMode·PCM이 포함된 원본 NPZ에서 녹화별 시계열, 접촉 이벤트 특징, 동기화 품질 보고서를 만듭니다.

모델 학습·추론은 [db_prediction](../db_prediction/README.md)이 담당합니다. 이 문서의 명령은 저장소 루트 `/home/hajaemin/go2_rl`에서 실행합니다.

## 1. 디렉터리 구조

```text
db_preprocessing/
├── scripts/                  # 전처리 실행·시간 정렬·접촉 추출·검증
├── models/
│   ├── go2_description.urdf  # 발 위치·Jacobian 계산용 Go2 구조
│   ├── LICENSE.unitree
│   └── README.md
├── dataset/{1007,1104}/       # 보관된 데이터셋 자료
├── preprocessed/1104/        # 현재 전처리 결과
│   ├── train/               # 녹화별 CSV·통과 집계·접촉 집계·품질 정보
│   ├── test/
│   ├── figure/              # 동기화 진단·접촉 그림·보고서
│   └── split.json           # 분할·원본 경로·해시·정책
└── README.md
```

`models/`는 학습된 소음 모델이 아니라 로봇의 기구학 계산용 자료입니다. `dataset/`에 보관된 파일과 원본 NPZ 입력은 구분하며, 원본 위치는 `--data-dir`로 지정합니다.

## 2. 처리 흐름과 코드 역할

```text
train/test 원본 NPZ
    → TrialReader로 로봇·오디오 읽기
    → 시작 착지로 지연 추정 / 종료 착지로 독립 검증
    → 500 Hz 공통 시간축과 보정된 오디오 특징
    → 보행·정상상태 구간 선택
    → 녹화별 CSV + 품질 판정 + 통과 녹화 집계
    → 접촉 이벤트 특징·시계열 + 그림·보고서
    → db_prediction에서 분석·학습
```

### 코드별 역할

아래 파일명은 `src/db_preprocessing/scripts/` 기준입니다.

| 파일 | 역할 |
|---|---|
| `preprocessing.py` | train/test NPZ 탐색, 처리 순서 제어, 개별 CSV·집계·manifest 저장 |
| `common_time.py` | `TrialReader`로 원본 읽기, 로봇·오디오 공통 시간축 및 RMS/피크 생성 |
| `time_synchronization.py` | 힘 기반 착지와 음향 onset 대응, 시작 지연 추정, 종료 구간 독립 검증 |
| `synchronization_diagnostics.py` | 녹화별 동기화 JSON, 전체 품질 요약·보고서 생성 |
| `synchronization_figures.py` | 동기화 요약·상세·전체 비교 그림 생성 |
| `refresh_figures.py` | 저장된 보정·매칭을 재사용해 동기화 그림 갱신 |
| `steady_state.py` | 보행·정상상태 구간 선택용 마스크 계산 |
| `dataset_output.py` | CSV 열 정의, 행 생성·저장 및 시계열 시각화 |
| `contact_detection.py` | 힘 임계값·히스테리시스 기반 공통 접촉 검출 |
| `contact_event_extraction.py` | 접촉 전후 힘·발 속도·소음 특징과 PCM 구간 추출, split별 집계 |
| `contact_noise_visualization.py` | 접촉 분석용 보조 함수와 요약 그림 |
| `go2_kinematics.py` | URDF 기반 발 위치·Jacobian·속도 계산 |
| `contact_features.py` | 과거 정보 기반 연속 접촉 특징; 예측 실험용 보조 모듈 |
| `verify_synchronization.py` | 원본 해시, 저장 지연, 센서/PCM 대응, 집계 대상 검증 |
| `verify_contact_events.py` | 접촉 추출·운동학·원본 PCM 통계 재계산 검증 |
| `test_*.py` | 합성 지연, 접촉 검출, 동기화 진단·시각화 등의 회귀 테스트 |

## 3. 처리 정책과 입출력

### 전처리 순서와 정책

1. 원본 `train/`, `test/` 구분을 유지하고 `no_vel*` 녹화는 제외합니다. 오디오는 NPZ에 포함된 PCM을 사용합니다.
2. 시작 구간의 힘 기반 착지와 음향 상승을 대응시켜 녹화별 지연을 추정합니다. 종료 구간은 독립 검증에 사용하며 시작 지연 추정에 섞지 않습니다.
3. 기본 500 Hz(2 ms) 시간축에 로봇 상태를 이전 표본 유지 방식으로 배치합니다. 기본 최대 공백은 20 ms입니다.
4. 보정된 오디오에서 20 ms RMS와 2 ms 피크를 계산하고 보행 구간을 선택합니다. `--selection walking|steady|full`로 선택 방식을 바꿀 수 있습니다.
5. 개별 CSV는 동기화 검토 대상도 보존합니다. **`all_trials.csv`에는 동기화를 통과하고 출력 길이가 기본 5초 이상인 녹화만 포함합니다.**

동기화 상태는 `CALIBRATION_REVIEW`, `INSUFFICIENT_END_EVENTS`, `END_RESIDUAL_REVIEW`, `CONSISTENT_CANDIDATE`로 기록됩니다. 이는 힘·음향 사건의 정렬 결과이며 센서 하드웨어 시계의 동기화를 보장하지 않습니다.

### 주요 입출력

| 자료 | 내용 |
|---|---|
| 원본 `train/*.npz`, `test/*.npz` | LowState, SportMode, PCM 및 시간 정보; `TrialReader`로 읽음 |
| `<output>/{train,test}/<trial>.csv` | 시간, 발 힘, 관절 상태, IMU, 속도, 오디오, 동기화 상태 |
| `all_trials.csv`, `quality.csv`, `metadata.json` | 통과 녹화 집계, 포함/제외 이유, 처리 설정 |
| `contact_event_features.csv`, `contact_event_samples.csv` 및 메타데이터 | 접촉 단위 특징과 시계열 분석 자료 |
| `<output>/split.json` | 녹화별 train/test 목록, 원본 경로·해시, 처리 정책 |
| `<output>/figure/` | 녹화별 동기화 JSON·그림, 접촉 그림, 전체 요약·보고서 |

예측 코드의 `scripts/features.py`는 `split.json`에 명시된 **녹화별 CSV 전체**를 읽습니다. `all_trials.csv`를 읽지 않으므로, 집계에서 제외된 동기화 검토 녹화도 모델 입력에 들어갈 수 있습니다. 그 상태는 `sync_group` 등 메타데이터에 남습니다. 전처리 집계의 통과 정책과 모델 입력 선택 정책을 별도로 확인해야 합니다.

## 4. 착지 기반 시간 동기화 상세

1. 원본 LowState 수신 시각 `ls_t`와 발별 `ls_ff`를 사용한다.
   시작 5초의 힘 10/90 분위수로 발별 저부하 기준 `L`과 고부하 기준 `H`를 구한다.
   OFF=`L+0.07(H-L)`, ON=`L+0.15(H-L)`이며 동적 범위가 10 미만인 발은
   동기화에서 제외한다. 종료 데이터나 오디오로 힘 임계값을 맞추지 않는다.
2. 히스테리시스와 6ms 확인을 거쳐 비접촉→접촉을 검출한다. 최초 임계값 통과
   시각을 사용한다. 초기 접촉 상태는 착지로 세지 않는다. 비접촉 구간이 60ms
   미만인 전환은 제거하고, 30ms 안의 여러 발 접촉은 가장 이른 시각의 한 묶음으로 취급한다.
3. 원본 PCM을 저장된 `audio_t0 + n * audio_slope`(n=1부터)로 매핑한다.
   약 1ms 단위 DC 제거 RMS와 3구간 평활값에서 음향 상승 후보를 찾는다.
   사전 정지 구간의 중앙값과 MAD로 상승/복귀 임계값을 고정한다. 5ms 이상
   조용했던 구간 뒤 상승 임계값을 2개 구간 연속 넘는 최초 시각을 onset으로
   사용하고 ±100ms에서 대응한다. 조용한 상태로 복귀하기 전의 큰 후속 피크는
   같은 음향 사건의 최초 onset을 공유하므로 후속 공진에 정렬하지 않는다.
4. 시작 5초 내 여러 착지에서 반복되는 지연 모드를 찾는다. 각 묶음은 한 표만
   갖고 ±6ms 안의 후보가 모드를 지지한다. 최대 prominence의 20% 미만인
   작은 후보는 제외한다. 20ms 이상 떨어진 다른 모드의 지지도가 최고 모드의
   90% 이상이면 모호한 매칭으로 보류한다. 동일 음향 onset의 중복 대응도 제외한다.
5. 최소 4개이면서 전체 후보 묶음의 60% 이상이 일치하고 MAD≤6ms인 경우에만
   중앙 지연을 채택한다. 양의 지연은 오디오가 늦다는 뜻이며 그만큼 앞당긴다.
   채택 실패 시 적용 지연은 0이고 실패 상태를 남긴다.
6. 마지막 검출 착지까지의 끝 5초를 같은 알고리즘으로 **독립적으로** 대응한다.
   정지 후 최대 2초를 포함한다. 시작·종료 창이 겹치는 짧은 시퀀스는 거부한다.
   종료 지연 중앙값의 보정 후 잔차와 종료 inlier 절대 잔차 P90이 모두 10ms
   이하여야 `CONSISTENT_CANDIDATE`이다. 종료 데이터를 시작 지연이나 시간축
   배율 추정에 사용하지 않는다.

상태는 `CALIBRATION_REVIEW`, `INSUFFICIENT_END_EVENTS`,
`END_RESIDUAL_REVIEW`, `CONSISTENT_CANDIDATE`이다. JSON에는 모드 모호성,
매칭 수, 첫/마지막 착지, 후보 전체와 제외 사유를 기록한다. 단일 첫·마지막 발의
소리가 약할 수 있으므로 주변 여러 착지를 판단 근거로 사용한다.

이는 힘 임계값 통과와 음향 상승 사이의 사건 정렬이다. 센서 취득 시각과 PC 수신
시각의 차이, 힘 센서 반응과 음향 반응 차이까지 분리한 하드웨어 동기화는 아니다.

## 5. 저장 결과와 그림

- `train/`, `test/`: 개별 시퀀스 CSV, `all_trials.csv`, `quality.csv`,
  `metadata.json`, 접촉 이벤트 특징/시계열 CSV 및 메타데이터.
- 개별 CSV는 검토 대상까지 보존하며 `synchronization_status`와
  `synchronization_passed`를 명시한다.
- **`all_trials.csv`에는 동기화 검증과 최소 출력 길이(기본 5초)를 통과한 시퀀스만 포함한다.**
  제외 대상과 이유는 `quality.csv`에 기록한다. 접촉 집계는 모든 개별 CSV를 포함하며
  `quality_ok`와 `synchronization_review_required`로 사용 여부를 구분한다.
- `figure/<trial>/`: 그림은 네 개만 유지한다. `synchronization.png`는 영어 판정·이동 방향·종료 오차·매칭 수와 첫/마지막 착지 확대를 담는다. `synchronization_detail.png`는 해당 착지의 발 힘·보정 전후 소리·음향 후보를 구분한다. 기존 `contact_noise.png`, `contact_force_velocity_with_noise.png`는 유지한다. 수신 시각 진단 수치는 JSON에 보존한다.
- `figure/synchronization_overview.png`: 처리한 시퀀스들의 종료 중앙 오차·90% 절대 오차 범위·판정을 한눈에 비교한다. 중간 구간 별도 검증이 있으면 그 결과도 판정에 표시한다.
- `refresh_figures.py`는 저장된 보정과 매칭을 그대로 사용하며, 보존 대상 두 PNG 및 기존 데이터 파일의 해시가 변하지 않았는지 확인한다.
- `figure/synchronization_summary.csv`, `report.md`, `run.json`: 전체 품질과 실행 설정.
- `split.json`: 입력 분리, 파일 경로/해시와 보정 정책.

공통 축 500Hz, 기본 이전 표본 유지, 최대 공백 20ms, 20ms 오디오 RMS와
2ms 피크, 기존 보행 구간 선택/CSV 특징 계산은 유지한다. 동기화는 구간 선택 전에
수행한다. 접촉 후 분석에서도 저장된 발별 임계값을 재사용하며 접촉 전후 특징의
계산식은 유지한다. 예측용 `contact_features.py`의 고정 설정은 별도이다.

## 6. 실행과 검증

NumPy, pandas, SciPy, Matplotlib가 필요합니다. 학습 코드까지 함께 사용할 때는 [모델 의존성 파일](../db_prediction/noise_prediction_model/requirements.txt)을 사용할 수 있습니다.

아래 전처리 명령은 원본 NPZ가 지정된 위치에 있고, 아래 경로 확인 절의 **URDF 경로 문제를 해결한 환경**을 전제로 합니다. 원본 경로는 실제 `train/`, `test/` NPZ 폴더를 가진 위치로 바꾸세요. 재실행은 출력 집계를 교체합니다.

```bash
python3 src/db_preprocessing/scripts/preprocessing.py \
  --data-dir /absolute/path/to/raw/1104 \
  --output-dir src/db_preprocessing/preprocessed/1104

python3 src/db_preprocessing/scripts/verify_synchronization.py \
  --output-dir src/db_preprocessing/preprocessed/1104

python3 src/db_preprocessing/scripts/verify_contact_events.py \
  --input-dir src/db_preprocessing/preprocessed/1104 \
  --raw-dir /absolute/path/to/raw/1104 \
  --figure-dir src/db_preprocessing/preprocessed/1104/figure
```

그림만 갱신하려면 저장된 보정 결과를 지정합니다.

```bash
python3 src/db_preprocessing/scripts/refresh_figures.py \
  --output-dir src/db_preprocessing/preprocessed/1104
```

다른 위치에서 전처리를 점검하려면 `--output-dir /tmp/1104-check --no-plot`을 사용할 수 있습니다. `--figure-dir`, `--hz`, `--max-gap`, `--sync-tolerance-ms`도 지정할 수 있습니다. 서로 다른 실행의 결과를 같은 출력 폴더에 섞지 않습니다.

회귀 테스트는 필요한 의존성과 아래 URDF 경로를 확인한 환경에서 실행합니다.

```bash
python3 -m unittest discover -s src/db_preprocessing/scripts -p 'test_*.py' -v
```

## 7. 현재 경로 확인 사항

| 위치 | 현재 상태와 필요한 조치 |
|---|---|
| 전처리 원본 | 기본값은 `src/db_preprocessing/data/1104`이나 현재 이 경로는 없습니다. `dataset/`이라는 이름만으로 원본 NPZ라고 가정하지 말고 실제 입력을 `--data-dir`로 지정합니다. |
| 전처리 기구학 | `scripts/go2_kinematics.py`는 `scripts/models/go2_description.urdf`를 찾지만 실제 URDF는 `db_preprocessing/models/`에 있습니다. 접촉 추출 실행 전에 코드의 모델 경로나 파일 배치를 맞춰야 합니다. |

현재 학습 코드는 `all_trials.csv`가 아닌 `split.json`의 녹화별 CSV를 읽습니다. 따라서 전처리 품질 통과 여부와 학습 포함 여부가 같다고 가정하지 않아야 합니다. 연결 방법과 모델 입력 정의는 [예측 코드 README](../db_prediction/README.md)를 참고하세요.
