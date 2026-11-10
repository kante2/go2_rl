# 접촉·착지·소음 분석

접촉력·발 속도와 소음의 관계를 분석하는 코드입니다. 현재 모델 학습은 [../README.md](../README.md), 기존 특징 선정 실험은 [../experiments/feature_selection/README.md](../experiments/feature_selection/README.md)를 참고하세요.

프로젝트 루트에서 실행합니다. `preprocessed/1007/train/contact_event_features.csv`와 녹화별 CSV·원본 NPZ가 필요합니다. 접촉 이벤트는 `preprocessing/contact_noise_visualization.py`로 생성합니다.

```bash
.venv/bin/python noise_prediction_model/analysis/contact_event_correlations.py
.venv/bin/python noise_prediction_model/analysis/landing_event_dynamics.py
.venv/bin/python noise_prediction_model/analysis/summarize_landing_spearman.py
.venv/bin/python noise_prediction_model/analysis/plot_landing_noise_distributions.py
```

| 코드 | 역할 | 출력 |
|---|---|---|
| `contact_event_correlations.py` | 접촉 이벤트의 힘·속도·소음 상관 | `../results/analysis/contact_events/` |
| `landing_event_dynamics.py` | 착지 −50~+100ms 힘·속도·변화율·PCM 소음 | `../results/analysis/landing_dynamics/` |
| `summarize_landing_spearman.py`, `plot_landing_noise_distributions.py` | 착지별 상관·소음 분포 요약 | 위 landing_dynamics 내부 |
| `audit_landing_contact_timing.py`, `review_landing_signal_alignment.py` | 접촉 시각·신호 정렬 진단 | 위 landing_dynamics 내부 |
| `audit_contact_alignment_impact.py`, `summarize_contact_alignment_impact.py`, `write_contact_alignment_report.py`, `plot_overall_contact_correlations.py` | 힘·속도 정렬의 영향 분석·보고서 | landing_dynamics/alignment_impact_review |
| `audit_landing_coverage.py` | 원본 기록·힘 하한·착지 검출 범위 점검 | landing_dynamics/coverage_audit |
| `verify_landing_update.py` | 저장 결과와 PCM·기구학·상관 재계산 대조 | landing_dynamics 검증 JSON |

후속 분석은 앞선 산출물이 필요합니다. `audit_landing_coverage.py`는 기존 데이터 갱신 실행의 `--run-dir`도 필요합니다. 통합 전 실행의 메타데이터는 당시 경로와 코드 해시를 보존하므로, 새 분석을 검증할 때는 새 경로에서 해당 분석을 먼저 생성하세요.

착지 후 최대값이나 centered 변화율은 회고 분석에 사용합니다. 미래 소음 예측 입력의 과거 정보 제약과 구분해야 합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s noise_prediction_model/analysis -p 'test_*.py' -v
```
