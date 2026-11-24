# Real2Sim 분석 파이프라인

```bash
bash src/Real2Sim/scripts/real2sim/run_converter.sh
```

`pipeline.py`가 기본 `1118_RL_base`의 대응 real/sim NPZ를 읽고,
전처리 → 시간 정렬 → 배포 관측 변환 → 지표 계산 → 결과 저장 → UI 내보내기 → 정적 그림 생성을 수행한다.
산출물은 모두 `result/<dataset>/`에 저장한다. 임시 결과가 완성된 뒤 공개 폴더를 전환하며 실패하면 기존 결과를 유지한다.

옵션은 `--dataset`, `--data-dir`, `--result-dir`, `--policy-dir`, `--skip-plots`다.
`--skip-plots`는 정적 그림만 생략하며 배열·지표·보고서·UI JSON은 생성한다.

## 모듈

- `preprocessing.py`: 파일 매칭·로딩·관절 순서·실측 tick·sim 물리 시간·필터·보간·평가 마스크.
- `time_alignment.py`: 앞 구간 q/dq 공통 이동 추정과 뒤 구간 검증, 모든 sim 채널에 같은 이동 적용.
- `deployment_mapping.py`: 배포 파일 검사와 q/dq 관측 변환. 고정 정규화는 `deployment_normalization.json`.
- `metrics.py`: 관절·발 힘 지표와 과거 0.5초 이동 RMSE.
- `reporting.py`: 보고서와 CSV.
- `dataset_summary.py`: 저장된 결과의 전체 물리 오차·접촉 시각 TXT. 기본 결과 생성 시 `result/<dataset>_real_sim_analysis.txt`도 갱신한다.
- `export_ui.py`: 저장된 NPZ·지표에서 JSON만 생성. 원본 로그나 분석 코드를 사용하지 않는다.
- `plot_mapping.py`, `plot_torque.py`: 저장된 분석 결과의 정적 그림.
- `legacy/`: 이전 정책 재구성·물리 재생 실험. 기본 파이프라인에서 호출하지 않는다.

## 시간 차이 진단

```bash
MPLCONFIGDIR=/tmp/go2_delay_matplotlib /usr/bin/python3 -s src/Real2Sim/scripts/real2sim/analyze_time_delay.py
```

기본 대상은 `straight_0.7_0.5_v2 / FR_thigh_joint`다. 현재 결과의 입력 경로와 해시를 확인하고 원본 공통 시각에서 별도 진단한다.
`--dataset`, `--result-dir`, `--trial`, `--joint`, `--calibration 시작 끝`, `--validation 시작 끝`을 지원한다.
JSON/CSV·그림은 `result/<dataset>/diagnostics/<trial>/`에 저장하며 공개 replay의 시간 이동은 변경하지 않는다.

전체 구조·단위·실행·테스트는 [Real2Sim README](../../README.md)를 참고한다.

시간 정렬은 기본 ±250ms 탐색이 경계에 도달한 경우에만 ±500ms로 한 번 확장한다.
앞 구간에서 주기를 선택한 뒤, 뒤 구간은 그 주기를 유지한 고정 후보 상관과 국소 잔차를 검증한다.
단순 반복으로 앞 구간부터 주기 선택이 모호하거나 앞뒤 시간 차이가 변하면 적용하지 않는다.
