# 기존 특징 선정 파이프라인의 설계·실험 기록

이 문서는 통합 전 루트 README의 설명과 결과를 보존한 기록입니다. 표본 수·점수·일부 추가 실험 안내는 과거 실행 기준입니다. 현재 폴더 구조와 실행 방법은 [통합 모델 안내](../../README.md)와 [특징 선정 실험 안내](README.md)를 따르세요.

# 로봇 보행 상태 기반 소음 예측

> GitHub 저장소에는 소스 코드·실행 안내·의존성 목록만 포함합니다. 원본 데이터, 전처리 CSV, PNG 등 그림, 학습 모델과 분석 결과는 `.gitignore`로 제외합니다. 아래 결과 링크는 로컬에서 파이프라인을 실행한 뒤 확인할 수 있습니다.

로봇의 현재·과거 상태로 짧은 미래의 보행 소음을 예측하고, 저소음 보행 설계에 참고할 상태 변수를 찾는다. 착지 이벤트에 한정하지 않고 보행 전체의 유효 데이터를 사용한다. 상관과 모델 중요도는 예측 기여이며 인과 효과 또는 강화학습 보상으로의 확정을 뜻하지 않는다.

## 빠른 탐색

- [현재 접촉력·발 상대 운동 기반 모델: 학습·평가](../../../noise_prediction_model/README.md)
- [현재 실시간 실행: 설치·ROS·오디오·웹 화면](../../../realtime_prediction/README.md)
- [폴더별 역할](../../../PROJECT_STRUCTURE.md)
- [데이터 분리와 전처리](#원본-데이터와-traintest-분리)
- [Feature 구성](#2-feature-extraction)
- [모델 학습](#3-모델-학습과-아키텍처)
- [Test 및 평가 결과](#4-test와-평가)
- [결과 파일과 재실행](#결과-경로와-재실행)
- [실시간 적용 시 주의점](#실시간-적용과-해석-범위)

현재 모델 학습·평가는 `noise_prediction_model/`, 실시간 실행은 `realtime_prediction/`에서 관리합니다. 아래의 `Noise_prediction_model/` 실행 순서와 결과는 별도의 기존 특징 선정·모델 비교 파이프라인에 대한 설명입니다.

코드를 새로 내려받은 환경에서는 먼저 Python 가상환경과 기본 의존성을 설치합니다. 데이터는 별도로 `data/1007/train`, `data/1007/test`에 준비해야 합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-noise.txt
```

TCN 실험은 추가로 `requirements-temporal.txt`를 설치합니다. 실제 로봇 기록 코드는 ROS 2와 Unitree 메시지 패키지가 설치된 환경에서 사용합니다.

현재 기준 데이터는 `data/1007/train`, `data/1007/test`의 원본 NPZ다. 이전 `full_lowstate`, `test_1_8`와 1ms RMS 실험 결과는 현재 경로에서 제거했다. 기본 실행은 아래 순서다.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python preprocessing/preprocessing.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python noise_prediction_model/experiments/feature_selection/noise_feature_extraction.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python noise_prediction_model/experiments/feature_selection/train.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python noise_prediction_model/experiments/feature_selection/test.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python noise_prediction_model/experiments/feature_selection/model_evaluation.py
```

## 설계 목적

큰 명령 선속도를 주어도 실제 몸체 선속도가 제한되는 조건에서는 명령 속도 증가만으로 소음 증가를 설명하기 어렵다. 착지 조건으로 데이터를 먼저 선택하면 발 충격 이외의 보행 상태를 놓칠 수도 있다. 따라서 실제 로봇 상태와 보행 전체 소음을 연결하고, 운동 상태만 사용하는 모델과 LowState를 추가한 모델을 비교한다.

## 원본 데이터와 train/test 분리

원본은 train 31개, test 6개 시퀀스로 이미 분리돼 있다. NPZ 안의 오디오와 상태를 사용하며 별도 WAV를 다시 읽지 않는다. 전처리 출력도 그 분리를 유지한다.

|용도|경로|시퀀스|전처리 행|
|---|---|---:|---:|
|Train 원본의 전처리|`preprocessed/1007/train/all_trials.csv`|31|372,881|
|실제 학습·feature 분석|위 파일 중 정지 대조 0.0_1 제외|30|357,966|
|Test 전처리|`preprocessed/1007/test/all_trials.csv`|6|74,611|

테스트는 `0.3_5`, `0.6_5`, `0.9_5`, `1.2_7`, `1.5_7`, `1.8_7`이다. 이번 분리에서 `1.8_6`은 train에 포함된다. Test는 feature 선정, 정규화, 파라미터 선택과 모델 종류 선정에 사용하지 않는다. 각 시점인 행과 여러 과거 시점을 합친 학습 창은 서로 다른 단위다. 현재 train의 유효 입력 창은 35,126개, test는 7,321개다.

## 1. 전처리: preprocessing.py

`preprocessing/preprocessing.py`는 원본 경로에 train/test가 있으면 두 split을 순서대로 처리한다. CSV와 metadata는 각 split에, 시계열·동기화 진단 그림은 `preprocessed/1007/figure`에 저장한다. 전체 모터 12개와 기록된 숫자 LowState를 보존한다. CSV는 시간·오디오·상태·호환용 별칭·구간 표시를 포함해 93개 열이다.

공통 시간축은 500Hz, 즉 2ms 간격이다. LowState와 Sport 모두 해당 시각까지 수신된 최신 과거값을 사용하며 미래 상태를 선형보간하지 않는다. 허용 공백은 기본 20ms다. 명령 move_start~move_stop 안의 전체 유효 보행 구간을 유지하고, 정상속도 여부 `is_steady`는 표시만 한다. `segment`와 시간 공백으로 끊긴 구간을 구분한다.

소음 목표는 **중심 20ms 창에서 평균을 제거한 RMS의 dBFS**다.

```text
RMS_AC = sqrt(mean(audio²) - mean(audio)²)
audio_dbfs_20ms = 20 * log10(RMS_AC / 32768)
```

이는 16bit full-scale 기준의 디지털 레벨이며 dB SPL은 아니다. 기존 1ms RMS CSV를 읽거나 그 값에 다시 RMS를 적용하지 않는다.

점프 후보를 이용한 공통 오디오 지연 추정은 **train 원본에서만** 수행한다. 그 값을 test에도 그대로 적용한다. 현재 train의 62개 점프 후보는 일관성 기준을 통과하지 못해 자동 보정값은 0ms다. 오디오와 상태를 공통 축에 놓았다는 사실만으로 물리적 동기화가 검증된 것은 아니다. [동기화 점검 보고서](../../../preprocessed/1007/figure/report.md)를 함께 확인한다.

`split.json`은 원본 경로·파일 SHA256·분리·보정 정책을 기록한다. 각 split의 `quality.csv`는 사용량과 전처리 상태, `metadata.json`은 시간축·RMS·보간·split 설정이다. 원본에 없는 모터 온도나 관절 가속도 등은 복원하지 않는다.

그림과 진단만 다시 만들려면 다음을 실행한다. CSV를 다시 쓰지 않는다.

```bash
.venv/bin/python preprocessing/preprocessing.py --diagnostics-only
```

발 접촉 상태와 소음 비교는 각 시퀀스 폴더의 `contact_noise.png`에 저장한다. 전체 구간과 중앙 1.5초 확대에서 20ms RMS와 FL/FR/RL/RR 접촉 추정 밴드를 보여준다. 발별 접촉력·부호 있는 수직 속도·이벤트 추출 영역은 별도 `contact_force_velocity.png`에 저장한다. 수직 속도는 양수가 상승, 음수가 하강이며 발을 들어 올리는 구간도 표시한다. Go2 공식 URDF의 관절 운동학에 LowState gyro·자세와 Sport 몸체 선속도를 반영한다. 기본 ON=10, OFF=5는 센서 값 기준이며 6ms 유지 후 확정한다. [접촉 그림·이벤트 표 안내](../../../preprocessed/1007/figure/contact_readme.md)에 판정·좌표계·시간 기준을 정리했다.

접촉 그림과 이벤트 표를 생성하려면 다음을 실행한다. 기존 시계열 CSV와 학습 모델은 갱신하지 않는다.

`contact_force_velocity_with_noise.png`에는 맨 위의 20ms RMS 소음과 아래의 발별 힘·수직 속도를 같은 시간축으로 배치한다. 전체 구간과 중앙 확대에서 이벤트 영역도 함께 표시한다.

```bash
.venv/bin/python preprocessing/contact_noise_visualization.py
```

`--trials 1.8_7`로 부분 실행하거나 `--on-threshold`, `--off-threshold`, `--confirm-ms`로 설정을 바꿀 수 있다. 최초 통과 시각과 검출 확정 시각을 따로 저장한다. `contact_states.csv`의 회고적 상태와 그때까지 확정된 causal 상태를 구분하며, 모델 입력에 회고적 상태를 그대로 넣으면 미래 정보가 섞일 수 있다.

접촉 전 20ms~후 40ms를 그림에 색칠하고, 각 split의 `contact_event_features.csv`에 접촉 하나당 한 행을 쌓는다. 접촉 직전 속도 xyz·크기·하강 속도, 접촉력과 후 40ms 피크·증가율·적분, 원본 PCM 기반 영역 소음을 저장한다. `contact_event_samples.csv`는 같은 영역의 표본별 힘·속도·소음이다. `--before-ms`, `--after-ms`로 영역을 바꿀 수 있다. Test 이벤트는 test 폴더에 따로 저장한다. 마이크는 공유하므로 동시 접촉 발과 기존 동기화 검토 상태를 함께 표시한다. 접촉 후 특징은 접촉 전 예측 입력으로 사용할 수 없다.

## 2. Feature extraction

`noise_feature_extraction.py`는 train CSV와 metadata를 읽고 20ms RMS와 과거값 유지 전처리를 확인한다. Test 목록은 전처리 metadata에서 가져온다.

LowState 후보 50개는 접촉력 4개, 가속도 3개, gyro 3개, quaternion 4개, 전체 모터 q/dq/tau 36개다. RPY는 quaternion과 중복되고 tick은 시간 식별자여서 입력에서 제외한다. 실제 몸체 선속도 x/y/z 3개는 별도 기준 모델과 결합 모델을 위해 유지한다. 시각·trial·명령 속도·오디오는 입력하지 않는다.

각 train 시퀀스와 각 lag에서 상태와 미래 소음의 Spearman 상관을 계산하고, 절댓값을 시퀀스·lag에 동일 가중치로 평균한다. 기본 상위 20개 raw LowState를 선정한다. 20개는 최적 개수로 확정한 값이 아니라 이번 기준 설정이다.

현재 선정은 acc x/z, gyro y, 네 다리 thigh/calf의 q와 dq, FL calf의 tau다. [선정 목록](../../../noise_prediction_model/results/feature_selection/features/selected_features.csv), [상관 분석 보고서](../../../noise_prediction_model/results/feature_selection/features/report.md)에 정확한 열 이름과 지표가 있다.

전체 후보의 모델 기반 feature 선정과 K 튜닝은 `noise_prediction_model/experiments/feature_selection/train_feature_optimal.py`로 실행한다. ExtraTrees 중요도와 내부 시퀀스 검증을 사용하며, 비교 방법과 실행 순서는 [실험 안내](../../../noise_prediction_model/results/feature_selection/feature_optimal/README.md)에 정리돼 있다. 이 실험은 기본 모델 결과를 덮어쓰지 않는다.

입력은 현재와 과거 20/60/100ms의 총 4개 시점이다. 결합 모델은 **(20 LowState + 3 선속도) × 4 = 92차원**, LowState 단독은 80차원, 속도 단독은 12차원이다. 입력 창은 시간 공백과 segment 경계를 넘지 않는다.

정답은 입력 기준 시각 t에서 **t+20ms 중심의 20ms RMS**다. 따라서 오디오 라벨은 대략 t+10~30ms 구간을 사용하지만, 모델 입력에는 미래 상태나 오디오가 없다. 다음 20ms 구간 `(t,t+20ms]`을 예측하도록 바꾸려면 라벨 정렬을 바꾸고 재학습해야 한다.

## 3. 모델 학습과 아키텍처

`train.py`는 저장된 feature 정의와 원본 파일 크기·수정 시각·metadata가 일치하는지 확인한다. 7개 모델을 비교한다.

|입력|모델|
|---|---|
|평균값|Dummy baseline|
|몸체 속도|Ridge, HistGradientBoosting|
|선정 LowState|Ridge, HistGradientBoosting|
|몸체 속도 + LowState|Ridge, HistGradientBoosting|

외부 5-fold GroupKFold는 시퀀스 전체를 검증에 남긴다. 각 외부 학습 분할 내부의 시퀀스 단위 20% holdout에서 파라미터를 고른다. Feature 순위도 외부 학습 및 내부 학습 분할에서 각각 다시 계산해 검증 정답 누출을 막는다. 각 시퀀스의 총 학습 가중치는 같아 긴 기록 하나가 손실을 지배하지 않는다.

Ridge는 학습 데이터만으로 StandardScaler를 적합하고 alpha 1/10/100/1000을 비교한다. Boosting은 120회, 학습률 0.08, L2 10, 최소 leaf 40개 샘플, 최대 leaf 7/15를 사용한다. 모델 종류는 외부 검증의 시퀀스별 RMSE 평균으로 선택하고 train 전체로 최종 적합한다. 모델과 최종 설정은 `noise_prediction_model/models/feature_selection`에 저장한다.

검증 시퀀스 안에서 500ms 블록을 섞는 permutation importance를 계산한다. 한 feature의 모든 lag는 함께 이동한다. 중요도는 모델의 예측 기여이고, 상관된 변수에 분산되거나 비현실적인 상태 조합의 영향을 받을 수 있다. 중요도가 크다는 이유만으로 그 값을 줄이는 RL 보상을 정당화하지 않는다.

## 4. Test와 평가

`test.py`는 저장된 모델로 test 6개 시퀀스의 예측만 생성한다. 재학습·튜닝·feature 선정은 하지 않는다. `model_evaluation.py`가 저장 예측에서 RMSE·MAE·R²와 높은 소음 구간 오차를 계산하고 그림과 보고서를 만든다. 높은 소음 기준은 train 목표의 90% 분위수다.

시퀀스별 평균과 전체 창을 합친 pooled 지표를 구분한다. 회귀 정확도를 백분율로 환산하지 않는다. 모델 종류는 test 점수를 보고 교체하지 않는다.

- [교차검증 결과](../../../noise_prediction_model/results/feature_selection/cross_validation/report.md)
- [테스트 결과](../../../noise_prediction_model/results/feature_selection/test/report.md)
- [테스트 모델 정확도](../../../noise_prediction_model/results/feature_selection/test/figures/model_accuracy.png)
- [실측–예측 비교](../../../noise_prediction_model/results/feature_selection/test/figures/actual_vs_predicted.png)
- [오차 분포](../../../noise_prediction_model/results/feature_selection/test/figures/error_distribution.png)
- [시퀀스별 RMSE](../../../noise_prediction_model/results/feature_selection/test/figures/trial_rmse_heatmap.png)

## 현재 학습·평가 결과

최종 선택 모델은 LowState와 선속도를 결합한 HistGradientBoosting이다. 모델 종류는 train의 외부 교차검증에서 선택했다.

|구간|시퀀스|창 수|평균 시퀀스 RMSE(dB)|평균 시퀀스 MAE(dB)|평균 시퀀스 R²|전체 창 RMSE(dB)|전체 창 R²|
|---|---:|---:|---:|---:|---:|---:|---:|
|교차검증|30|35,126|2.610|1.905|0.812|2.637|0.857|
|Test|6|7,321|2.526|1.866|0.832|2.564|0.867|

Test의 속도만 사용하는 Boosting 평균 RMSE는 3.261dB, LowState 단독은 2.548dB다. 결합 모델에서 LowState의 예측 기여를 관찰할 수 있지만 인과 효과의 증거는 아니다. 높은 소음 구간 오차는 시퀀스별 약 2.43~5.90dB이며 큰 피크에는 오차가 남는다.

## 결과 경로와 재실행

```text
data/1007/train, test                 원본 NPZ/WAV
preprocessed/1007/train, test         split별 CSV·metadata·quality
preprocessed/1007/figure              시계열·동기화·수신 품질 그림
preprocessed/1007/split.json          분리·원본 해시·보정 정책
noise_prediction_model/results/feature_selection/features   선정 목록·상관·선정 안정성·중요도
noise_prediction_model/models/feature_selection               현재 모델·설정·튜닝
noise_prediction_model/results/feature_selection/cross_validation  현재 교차검증
noise_prediction_model/results/feature_selection/test              현재 test
```

데이터를 다시 전처리하면 feature 추출부터 순서대로 다시 수행한다. 저장된 예측의 그림만 다시 만들 때는 `model_evaluation.py`만 실행한다. K 자동 튜닝은 `train_op.py`, TCN/과거 창 비교는 `train_temporal.py`의 선택 실행이다. 이번 기본 실행에는 포함하지 않았으며 이전 데이터의 결과 폴더를 현재 결과와 함께 유지하지 않는다.

관련 문서: [모델 실행 안내](../../../noise_prediction_model/experiments/feature_selection/README.md), [프로젝트 구조](../../../PROJECT_STRUCTURE.md).

## 실시간 적용과 해석 범위

현재 기본 모델은 CSV 기반이다. 실시간에서는 ROS 메시지를 시간 기반 버퍼에 저장하고 같은 순서·단위로 과거 상태를 구성해야 한다. 초기에는 100ms 이력이 필요하지만 매번 100ms를 기다리지는 않는다. RMS 윈도·예측 horizon·통신 지연·모델 계산 시간을 구분한다. 이번 Sport 전처리는 과거값 유지로 실시간 입력 방식과 맞췄지만, 실제 장치에서의 시각 정렬과 전체 지연은 별도 검증해야 한다.

같은 세션 기록의 재분할은 새 날짜·바닥·마이크 조건에 대한 독립 검증이 아니다. 오디오 지연 보정의 일관성도 확인되지 않았다. 잔차만으로 Robot State와 무관한 소음이라고 결론내리지 않는다.
