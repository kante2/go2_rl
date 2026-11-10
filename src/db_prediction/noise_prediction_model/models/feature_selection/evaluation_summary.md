# 현재 저장 모델의 업데이트·평가

대표 모델: `speed_lowstate_hist`. 모델 종류: `HistGradientBoostingRegressor`.
학습 40개 trial / 43,657개 창. 대표 모델 입력 92차원. 입력 lag [0, 20, 60, 100]ms, 주기 20ms, horizon 20ms.

Train/test 폴더 구분을 유지했다. 명령 3축이 모두 0인 기록은 기존 기본 정책으로 모델 창에서 제외했다. 측면 이동·제자리 회전은 포함했다. 명령값은 입력이 아니며 측정 선속도와 LowState·과거 접촉 파생값을 사용한다.

|Test 범위|Trial|이전 평균 trial RMSE|새 평균 trial RMSE|새 전체 R²|
|---|---:|---:|---:|---:|
|all_test|10|3.2129|2.5085|0.8670|
|existing_test|6|2.5106|2.5286|0.8673|
|added_test|4|4.2663|2.4783|0.8584|
|forward_translation|6|2.5106|2.5286|0.8673|
|in_place_rotation|1|4.4248|3.1074|0.8056|
|lateral_translation|3|4.2135|2.2686|0.8774|

[같은 test 창에서의 상세 전후 비교](../model_evaluation/data_update_20261015_224022/model_comparison_report.md)

[현재 test 평가](../model_evaluation/test/report.md) · [현재 시퀀스 교차검증](../model_evaluation/cross_validation/report.md)

착지 전후의 centered 변화율·착지 후 최대값은 회고 분석용이다. 이번 학습에서는 기존 과거 정보만 쓰는 접촉 특징 정의를 유지했다. 소음은 분리된 발 충격음이 아닌 전체 녹음 RMS dBFS다.

추가 기록에서 힘 신호가 기존 OFF=5 기준에 도달하지 않아 착지 검출이 드문 문제가 있다. 전체 시계열 학습에는 포함했다. 접촉 상태·경과시간 후보는 이 기준의 영향을 받지만, 현재 대표 모델의 선정된 접촉 파생 입력은 `impact_rate_max20_rr`이다. [검출 범위와 힘 하한 검토](../Feature_extraction/landing_dynamics/coverage_audit/report.md)를 참고한다.

[이번 데이터 갱신 전체 보고서](../model_evaluation/data_update_20261015_224022/report.md)
