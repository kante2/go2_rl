# 기존 특징 선정·모델 비교 실험

기존의 전체 보행 특징 선정 파이프라인입니다. 현재 64차원 PyTorch 모델은 [통합 모델 안내](../../README.md)를 따릅니다. 착지·접촉 분석 코드는 [analysis](../../analysis/README.md)에 있습니다. 과거 상세 설계와 결과는 [DESIGN.md](DESIGN.md)에 보존했습니다.

LowState 원본 50개와 접촉 파생 36개 후보에서 기본 상위 20개를 선정하고, 몸체 선속도 x/y/z와 0/20/60/100ms 시점을 사용합니다. 기본 모드는 `contact`이며 `--feature-mode raw`로 원본 특징만 사용할 수 있습니다. 유효성 마스크 의존성 때문에 실제 선정 열 수는 설정에 따라 달라질 수 있습니다. 목표는 t+20ms 중심의 20ms RMS 소음(dBFS)입니다.

평균 baseline, 선속도/선정 특징/결합 입력 각각의 Ridge·HGB 등 총 7개 모델을 비교합니다. 녹화 단위 외부 5-fold와 내부 holdout에서 특징·파라미터를 선택하며, 각 녹화의 총 학습 가중치를 같게 맞춥니다. 이 가중치와 입력 정의는 현재 PyTorch 비교와 다릅니다.

## 실행

프로젝트 루트에서 `noise_prediction_model/requirements.txt`의 의존성을 설치한 뒤 실행합니다. 입력은 `preprocessed/1007/train|test/all_trials.csv`와 metadata입니다.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python noise_prediction_model/experiments/feature_selection/noise_feature_extraction.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python noise_prediction_model/experiments/feature_selection/train.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python noise_prediction_model/experiments/feature_selection/test.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python noise_prediction_model/experiments/feature_selection/model_evaluation.py
```

| 코드 | 역할 |
|---|---|
| `noise_feature_extraction.py` | 후보 상관·입력 선정·시간 창 생성 |
| `train.py` | 교차검증·중요도·최종 학습 |
| `test.py` | 예약된 test 녹화 예측 생성 |
| `model_evaluation.py` | 저장 예측의 지표·그림·보고서 생성 |
| `predict_noise.py` | 대표 joblib 모델로 오디오 정답 없는 CSV 예측 |
| `compare_model_update.py` | `--run-dir`에 보관한 이전 모델과 새 모델 비교 |

특징 선정 결과는 `noise_prediction_model/results/feature_selection/features/`, 교차검증·test·raw/contact 비교는 `noise_prediction_model/results/feature_selection/`에 저장합니다. 모델은 `noise_prediction_model/models/feature_selection/`의 `noise_models.joblib`(전체)과 `noise_predictor.joblib`(대표)입니다. 기존 수치 결과와 체크포인트는 재학습 없이 이동했습니다.

```bash
.venv/bin/python noise_prediction_model/experiments/feature_selection/predict_noise.py \
  --input preprocessed/1007/test/0.3_5.csv --output /tmp/feature_selection_predictions.csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s noise_prediction_model/experiments/feature_selection -p 'test_*.py' -v
```

`train_feature_optimal.py`는 미완성 추가 실험의 보존 파일입니다. 필요한 `train_op.py`가 기존 저장소에도 없어 실행할 수 없습니다. 기본 학습·평가에는 사용하지 않습니다.
