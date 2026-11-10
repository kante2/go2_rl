# 소음 예측 모델·분석·실험

이 폴더에서 현재 PyTorch 모델, 접촉·착지 분석, 기존 특징 선정 실험을 함께 관리합니다. 각 파이프라인의 입력 정의와 학습 방식은 분리해 유지합니다.

| 경로 | 역할 | 결과 |
|---|---|---|
| `scripts/` | 현재 64차원 Ridge·HGB·Sparse GPR 학습·평가·추론 | `models/01_…`~`03_…`, `results/01_…`~`03_…` |
| [analysis/](analysis/README.md) | 접촉·착지·소음의 상관·시계열 분석 | `results/analysis/` |
| [experiments/feature_selection/](experiments/feature_selection/README.md) | 기존 특징 선정·sklearn 모델 비교 | `models/feature_selection/`, `results/feature_selection/` |
| `../realtime_prediction/` | 저장된 PyTorch 모델로 실시간 실행 | 선택적 예측 CSV·ROS 토픽·웹 화면 |

저장된 모델·수치 결과는 통합 시 재학습 없이 이동했습니다. 기존 실행 메타데이터와 보고서에 포함된 과거 경로·코드 해시는 실행 당시 기록으로 유지합니다. 아래 설명과 실행 명령은 현재 `scripts/` 파이프라인을 기준으로 합니다.

## 1104 전체 시퀀스 실험

새 데이터 분석·모델은 [runs/1104](runs/1104/README.md)에 분리합니다. 동기화 검토 대상을 포함한 train 24개·test 5개 전체를 사용하며, 전처리의 `all_trials.csv` 대신 `split.json`에 명시된 개별 CSV를 읽습니다. 분석은 train만 사용하고, 모든 모델은 같은 64차원 입력과 녹화 단위 5-fold CV로 비교합니다. 기존 `models/`, `results/`의 1007 모델은 유지합니다.

```bash
.venv/bin/python noise_prediction_model/analysis/study_1104.py --stage analyze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python noise_prediction_model/scripts/train_evaluate.py \
  --source preprocessed/1104 --output noise_prediction_model/runs/1104
.venv/bin/python noise_prediction_model/analysis/study_1104.py --stage evaluate
.venv/bin/python noise_prediction_model/scripts/verify.py --output noise_prediction_model/runs/1104
```

이하 기존 기본 경로 설명은 1007 실행에 해당합니다. `--source`와 `--output`으로 데이터와 출력 위치를 지정할 수 있습니다.

## 현재 PyTorch 모델

모델 구성은 **Ridge, HGB, Sparse GPR**이다. 세 모델 모두 PyTorch 텐서로 추론한다. Ridge·GPR은 PyTorch로 학습하고, **HGB는 sklearn으로 학습한 트리를 PyTorch 추론 모델로 변환**한다. 모델은 각 모델 폴더의 `model.pt`에, 결과는 기존 `results/`에 저장한다.

현재까지의 LowState로 **20ms 뒤를 중심으로 한 20ms RMS 소음(dBFS)**을 예측한다. 입력 시점 기준 정답 구간은 **[+10, +30)ms**다. 데이터는 `preprocessed/1007`의 `split.json`과 녹화별 CSV만 사용한다.

- Train: 41개 녹화, 45,774개 표본. 녹화 단위 5-fold 교차검증으로 설정을 선택한 뒤 전체 train으로 재학습한다.
- Test: 10개 녹화, 10,609개 표본. 표준화·유도점 초기화·최적화·설정 선택에 사용하지 않는다.
- 세 모델의 64차원 입력·표본·CV 분할은 동일하다. 표본 가중치는 모두 같고, 설정 선택 기준은 녹화별 RMSE의 평균이다.

## 실행

프로젝트 루트 `/home/hajaemin/low_noise`에서 실행한다.

```bash
# 세 모델을 모두 새로 학습하고 기존 모델·결과를 갱신한다.
.venv/bin/python noise_prediction_model/scripts/train_evaluate.py

# HGB만 다시 학습하고 나머지 모델·결과는 해시/스키마/출처 확인 후 재사용
.venv/bin/python noise_prediction_model/scripts/train_evaluate.py --models hgb

# GPR 학습 epoch 변경
.venv/bin/python noise_prediction_model/scripts/train_evaluate.py --gp-epochs 30

.venv/bin/python -m unittest noise_prediction_model.scripts.test_pipeline -v
.venv/bin/python noise_prediction_model/scripts/verify.py

# 오디오 없는 CSV 추론: ridge / hgb / sparse_gpr
.venv/bin/python noise_prediction_model/scripts/predict.py \
  --input preprocessed/1007/test/0.3_5.csv \
  --model hgb --output /tmp/causal_hgb_predictions.csv
```

추론 기본 모델은 `sparse_gpr`다. `--model-path`로 별도 `.pt` 체크포인트를 지정할 수 있다. 출력 CSV가 이미 있으면 `--overwrite`가 필요하다. CSV 하나는 하나의 녹화여야 한다. 실시간 스트리밍 기본값은 사전에 지정한 HGB이며, test 성능으로 자동 선정한 모델이 아니다.

## 모델과 학습 방식

1. **Ridge Regression:** train 입력을 표준화하고 `torch.linalg.solve`의 float64 연립방정식으로 L2 회귀를 학습한다. alpha 1/100/1000을 비교한다. 최종 계수는 원래 입력 단위에 합쳐 float32로 저장하며 추론은 선형 텐서 연산이다.
2. **HGB:** 기존 `HistGradientBoostingRegressor`의 두 설정(120 iterations/15 leaves, 200 iterations/31 leaves)을 같은 녹화 단위 CV로 비교한다. 자동 early stopping은 끈다. 학습된 수치 임계값·리프 값·자식 인덱스를 텐서로 옮겨 `TorchHGB`로 저장한다. 모든 트리를 배치 단위로 함께 순회하며 임계값과 리프 값은 float64로 보존한다. 리프에 포함된 learning rate를 중복 적용하지 않는다. 범주형 분기는 지원하지 않으며 현재 64개 연속 입력에는 해당하지 않는다. **HGB 학습 자체가 PyTorch로 구현된 것은 아니다.**
3. **Sparse GPR:** Matérn-3/2 ARD 커널, 일정한 Gaussian 관측 잡음, 128/256개 유도점을 비교한다. 유도점은 해당 fold의 train에서 PyTorch 미니배치 k-means로 초기화하고, 커널·유도점을 미니배치 ELBO로 학습한다. 고정된 최종 커널에서 **전체 train으로 최적 Gaussian 변분 사후분포**를 계산한다. 사후분포 선형계는 float64로 풀고 추론 버퍼는 float32로 저장한다. 학습·사후분포 계산·추론에 NumPy/SciPy/sklearn 모델 연산을 사용하지 않는다.

Ridge·GP의 표준화와 dBFS 복원은 모델 내부에 포함되어 있다. HGB는 원래 단위의 입력으로 트리를 판정한다. 세 모델 모두 CPU의 float32 입력 텐서 호출을 평가하되, HGB 내부 임계값·리프 값과 출력은 float64다. sklearn은 HGB 학습·변환 동등성 검사·CV 분할·평가 지표에, SciPy는 불확실성 지표·그림에 사용한다. 저장된 HGB를 불러와 추론할 때는 sklearn 모델을 호출하지 않는다.

## 코드별 역할

| 파일 | 역할 |
|---|---|
| `scripts/features.py` | CSV에서 인과적인 64차원 입력·미래 정답 생성, 녹화/단절 분리, 데이터와 출처 기록 저장 |
| `scripts/lowstate_kinematics.py` | LowState 관절·자세·각속도로 발 상대 수직 속도 계산 |
| `scripts/torch_models.py` | Ridge 학습/추론, sklearn HGB 트리 → PyTorch 변환·추론, NumPy 어댑터, `.pt` 저장·로딩 |
| `scripts/sparse_gp.py` | PyTorch 변분 GP 학습, 전체 train 사후분포 계산, 평균·불확실성 추론 |
| `scripts/train_evaluate.py` | 공통 CV, 설정 선택, 최종 학습, 모델 고정, 지표·예측·추론 시간·보고서 생성 |
| `scripts/predict.py` | 오디오 없는 CSV를 읽고 모델별 예측 CSV 생성 |
| `scripts/plots.py` | 산점도·잔차·녹화별 시계열·학습 곡선·비교·불확실성 그림 생성 |
| `scripts/verify.py` | 원본 해시·분할·체크포인트·저장 지표·그림을 독립 검증 |
| `scripts/test_pipeline.py` | 인과성·단절·학습 격리·회귀/GP 수식·HGB 임계값 경계/결측 분기·체크포인트·배치·기울기 검증 |
| `../realtime_prediction/stream_model.py` | 새 체크포인트를 읽고 LowState 이력에서 실시간 특징·예측 생성 |
| `../realtime_prediction/run_live.sh` | PyTorch HGB를 지정해 실시간 추론 프로그램 실행 |
| `../realtime_prediction/test_stream_model.py` | 오프라인/실시간 특징·예측 일치, 이력 reset, NPZ 재생 검증 |
| `../realtime_prediction/web_dashboard.py` | 실시간 소음 그래프와 모델 연결 없이 실행하는 데모 제공 |

루트 `features.py`, `lowstate_kinematics.py`는 `scripts/` 파일을 가리키는 호환 심볼릭 링크다. `realtime_prediction/stream_model.py`도 새 `.pt` 로더를 사용하며, `run_live.sh`는 HGB 체크포인트를 지정한다.

## 저장 경로와 출력 정의

```text
noise_prediction_model/
├── datasets/                        # 공통 입력·정답·메타데이터·CV 녹화·출처 해시
├── models/
│   ├── 01_ridge_regression/          # model.pt, selection.json, 계수와 절편
│   ├── 02_hgb/                       # model.pt, selection.json, tree_structure.json
│   ├── 03_sparse_gpr/                # model.pt, selection.json, GP 파라미터
│   ├── feature_selection/            # 기존 sklearn/joblib 실험 모델
│   ├── causal_forecast/selection.json # 실시간 기본 모델 경로
│   └── frozen_manifest.json          # test 평가 전 고정한 .pt 해시
├── results/
│   ├── 01_ridge_regression/
│   ├── 02_hgb/
│   ├── 03_sparse_gpr/
│   ├── analysis/                     # 접촉·착지 분석 결과
│   ├── feature_selection/            # 특징 선정·기존 모델 평가 결과
│   ├── model_comparison.csv/json/md/png
│   ├── verification.json
│   └── run.json
└── logs/training.log
```

`model.pt`는 **metadata + state_dict** 형식이다. 직렬화된 sklearn 객체, joblib 또는 TorchScript 파일이 아니다. HGB에는 트리 텐서와 깊이 정보가 들어간다. `load_bundle()`이 스키마·특징 순서·학습 녹화·선택 설정과 PyTorch 모델을 반환한다. `load_model()`은 모델만 반환한다. 로더는 `torch.load(..., weights_only=True)`를 사용한다.

각 `results/<모델>/`의 출력은 다음과 같다.

| 출력 | 내용 |
|---|---|
| `cv_candidates.csv`, `cv_folds.csv` | 후보별/녹화 fold별 train OOF 점수와 학습 시간 |
| `selection.json` | 선택 설정, train CV 기준, 학습 표본/녹화, test 미사용 기록 |
| `metrics.csv/json` | train/train_oof/test의 RMSE·MAE·R²·bias, 녹화별 RMSE 평균 |
| `metrics_by_trial.csv` | 각 녹화의 오차 지표 |
| `predictions.csv.gz` | sample_id, 녹화·분할·시간, y_true_dbfs, y_pred_dbfs, error_db |
| `conversion_parity.json` | HGB에서 sklearn 원본과 저장된 PyTorch 모델의 전체 train/test 예측 차이; CV fold 차이는 `cv_folds.csv`에 저장 |
| `inference_runtime.json` | PyTorch CPU 단일 스레드에서 배치 1/64/1024의 중앙값·p95·p99·최대 지연, 처리량 |
| `training_history.csv` | HGB의 iteration별 train RMSE 또는 GPR의 표본당 negative ELBO; Ridge는 해당 없음 |
| `figures/` | 실측/예측 산점도, 잔차, 녹화별 시계열; HGB/GP 학습 곡선과 GP calibration |

GP 예측 CSV에는 `latent_std_db`, `observation_std_db`, `lower_95_dbfs`, `upper_95_dbfs`도 저장한다. `uncertainty_metrics.csv`는 구간 포함률·폭·Gaussian NLL을 기록한다. 함수 불확실성과 관측 잡음 포함 불확실성은 구분한다.

- `train`: 최종 모델이 학습한 표본에 대한 적합도다.
- `train_oof`: 해당 녹화를 학습하지 않은 fold 모델의 예측이다. 같은 OOF로 설정을 선택하므로 완전히 독립적인 최종 평가는 아니다.
- `test`: 설정을 선택하고 세 체크포인트를 모두 고정한 뒤 평가한 보류 녹화다.
- RMSE/MAE의 단위는 dB다. R²는 전체 표본을 합쳐 계산하며, 녹화별 RMSE 평균은 별도 지표다.

[최신 세 모델 비교 보고서](results/model_comparison.md)와 [비교 그래프](results/model_comparison.png)를 확인한다. `results/run.json`은 실행 상태·epoch·표본 수·백엔드를, `environment.json`은 라이브러리 버전을 기록한다.

추론 시간은 **장치에 준비된 입력 텐서의 모델 호출**이며, 배치 전체에 걸린 ms다. 단일 표본 NumPy 어댑터 호출은 `numpy_adapter_single`에 따로 기록한다. 특징 추출·장치 전송·MuJoCo 물리·정책 학습은 포함하지 않는다. CPU 수치를 GPU 성능이나 전체 강화학습 속도로 해석하면 안 된다.

## MuJoCo/PyTorch에서 사용

모델은 한 번만 로드하고 여러 환경의 특징을 한 배치로 전달한다.

```python
import torch
from noise_prediction_model.scripts.torch_models import load_model

device = torch.device('cpu')  # CUDA 환경에서는 'cuda'
model = load_model('noise_prediction_model/models/02_hgb/model.pt', device=device)

# features: 같은 장치의 float32 Tensor [num_envs, 64]
# 특징 순서는 datasets/feature_schema.csv를 따른다.
with torch.inference_mode():
    predicted_dbfs = model(features)  # HGB: float64 [num_envs], dBFS

# GP에서 불확실성이 필요할 때만:
# mean, latent_std, observation_std = gp_model.predict_distribution(features)
```

네 발의 힘·상대 수직 속도·각 변화율을 현재 및 10/20/40ms 과거에서 가져온다. 특징 순서는 **lag → force/velocity/force_change/velocity_change → fr/fl/rr/rl**이다. 변화율은 6ms 후방 차분이고 최소 46ms 이력이 필요하다. 학습 데이터의 2ms 격자와 20ms 예측 주기를 시뮬레이터에서도 맞춰야 한다. 각 환경의 reset 때 이력을 비우고 충분히 채운 뒤 사용한다.

힘은 보정되지 않은 센서 단위이며, 속도는 `[R(ω×p + Jdq)]_z`로 몸체 병진을 제외한다. 모델은 시뮬레이터 접촉력과 실측 센서 단위를 자동으로 맞추지 않는다. 정답은 dB SPL이 아닌 dBFS다.

## 해석의 한계

Test는 과거 연구에서 이미 확인한 데이터이므로 이번 비교만으로 완전히 새로운 주행 조건의 일반화를 주장할 수 없다. GP는 일정한 Gaussian 관측 잡음을 가정하며 이분산·다봉성 분포를 검증한 모델이 아니다. float32 분산 계산에는 반올림 오차가 있을 수 있다. 예측 구간과 작은 표준편차는 새로운 보행에서 정확도를 보장하지 않으며 불확실성 보상 벌점은 별도로 검증해야 한다.

GP 방법론: [Titsias 2009](https://proceedings.mlr.press/v5/titsias09a.html), [Hensman et al. 2013](https://arxiv.org/abs/1309.6835).
