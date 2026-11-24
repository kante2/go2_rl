# Real2Sim 기록 분석과 replay

```text
data/real/<dataset> + data/sim/<dataset>
  → scripts/real2sim/pipeline.py
  → result/<dataset>/processed + metrics
  → export_ui.py가 저장된 결과를 읽음
  → result/<dataset>/ui
  → replay/serve.py → web/app.js · chart.js
```

분석 실행과 UI 실행은 분리되어 있다. 기본 데이터셋은 `1118_RL_base`다.
현재 파일명의 18쌍을 비교한다. 파일 목록은 실행 시 원본 폴더에서 다시 읽는다.
원본 데이터는 수정하지 않는다. 기본 분석에는 정책 추론·재시뮬레이션·물리 파라미터 피팅이 없다.

## 실행

저장소 루트에서 전체 분석과 정적 그림을 생성한다.

```bash
bash src/Real2Sim/scripts/real2sim/run_converter.sh
```

그림 없이 배열·지표·UI JSON만 생성하려면 `--skip-plots`를 추가한다.
분석 환경은 Python 3, NumPy, SciPy가 필요하며 그림 생성에는 Matplotlib이 필요하다.

저장된 결과를 브라우저에서 확인한다.

```bash
bash src/Real2Sim/replay/run_replay.sh
```

주소는 `http://127.0.0.1:8791`이다. 다른 포트는 `--port 8792`로 지정한다.
UI 서버에는 Python 표준 라이브러리만 필요하다. 원본 NPZ를 읽거나 분석을 다시 실행하지 않는다.
결과가 없거나 미완료이면 분석 명령을 안내하고 종료한다.
두 명령 모두 `--dataset <name>`을 지원하며, `--result-dir`로 완성 결과 폴더를 지정할 수 있다.

## 코드와 결과 구조

```text
Real2Sim/
  data/real/<dataset>/       원본 *_real.npz
  data/sim/<dataset>/        원본 *_sim.npz
  scripts/real2sim/
    pipeline.py             실행 순서·결과 저장·완성 결과 전환
    config.py               기본 경로·데이터셋·채널 이름
    preprocessing.py        로딩·형식·관절 순서·필터·보간·평가 마스크
    time_alignment.py       시간 이동 추정·검증·동일 이동 적용
    deployment_mapping.py   배포 관측 변환·배포 파일 검증
    deployment_normalization.json
    metrics.py              지표·과거 0.5초 이동 RMSE
    reporting.py            CSV·분석 보고서
    dataset_summary.py      저장된 물리 결과에서 전체 TXT 분석 보고서 생성
    export_ui.py            저장된 배열·지표를 UI JSON으로 직렬화
    plot_mapping.py         저장 결과의 매핑 그림
    plot_torque.py          저장 결과의 토크 그림
    analyze_time_delay.py   원본 로그 기반 별도 시간 차이 진단
    legacy/                 이전 정책 재구성·MuJoCo 재생 실험
    tests/
  result/<dataset>/
    manifest.json           입력 해시·분석 설정·코드 해시·완료 상태·산출물 해시
    processed/*_mapping.npz 물리·정책 관측·마스크·시간 이동 전 배열·이동 RMSE
    metrics/                시행별 메타데이터·지표, mapping_summary.json, metrics.csv
    figures/mapping/        매핑 그림
    figures/torque/         토크 그림·표·분석
    analysis.md             전체 분석 보고서
    ui/                     index.json·시행별 JSON
    diagnostics/            별도 진단 실행 시 생성
  result/<dataset>_real_sim_analysis.txt  전체 물리 오차·발 파형·착지 시각 분석
  replay/
    run_replay.sh           서버만 실행
    serve.py                web/와 result/<dataset>/ui를 읽어서 제공
    web/                    HTML·JS·CSS
    tests/                  HTTP 서버·UI 동작 검사
```

`result/<dataset>/`가 공식 결과 저장소다. UI용 JSON은 그 안의 `processed/`와 `metrics/`만 읽어 생성한다.
UI 내보내기는 필터·시간 정렬·관측 변환·오차 계산을 수행하지 않는다.
분석은 임시 폴더에서 모든 작업을 완료한 뒤 공개 결과를 전환한다. 분석·그림 생성에 실패하면 기존 결과를 유지한다.

## 비교 규약

실측 tick 중복 제거 후 시계를 PC 수신 시각에 대응시킨다. sim 필터·미분에는 `sim_time`을 쓰고
비교에는 공통 PC 시각을 사용한다. 양쪽 q/dq/tau에 15Hz 필터를 적용하고 ddq는 필터 dq에서 미분한다.
공통 시간축은 200Hz다. 녹음 시작 2초·끝 1초 및 데이터 누락 주변을 평가에서 제외한다.
`steady` 키는 이 평가 마스크이며 정상 보행을 자동 판정한 의미가 아니다.

앞 구간의 8개 thigh/calf 관절 q/dq로 ±250ms를 5ms 간격으로 탐색하고 별도의 뒤 구간에서 검증한다.
기본 탐색이 경계에 도달한 경우에만 ±500ms로 한 번 확장한다. 앞 구간에서 대안 peak 대비 상관 차이 0.02 이상으로
주기를 선택하고, 뒤 구간은 선택한 주기 주변 ±80ms에서 국소 잔차가 20ms 이내인지 확인한다.
뒤 구간의 고정 후보 상관은 0.70 이상이어야 하고, 더 좋은 전역 후보와의 차이가 0.02를 넘으면 거부한다.
뒤 구간에 한 보행 주기 차이의 동점이 남으면 앞에서 선택한 주기를 유지하며 UI에 모호함을 표시한다.
앞 구간부터 주기를 구분할 수 없거나 시간 차이가 변하는 시행은 확장으로 덮어쓰지 않는다.
검증된 시행별 이동 하나를 모든 sim 관절·토크·발 힘에 적용한다. 검증 미달은 0ms를 유지한다.
관절 상관만 낮은 경우에는 접촉 하중을 추가 근거로 사용한다. 앞/뒤 네 발 평균 상관 0.85 이상,
발별 상관 0.75 이상, 발별 추정 편차 20ms 이하와 관절 추정값과의 차이 20ms 이하를 모두 요구한다.
적용 시각에서 관절 상관도 0.50 이상이어야 한다. 관절 추정이 주기 모호·경계·드리프트로 실패한 경우에는 이 경로로 덮어쓰지 않는다.
양수는 `aligned_sim(t) = original_sim(t + lag)`이며 sim 파형을 왼쪽으로 이동한다.
파형 이동에는 정책 위상·동역학 차이가 포함되므로 통신 지연 측정값으로 해석하지 않는다.
현재 `straight_0.5_0.5_v1/v2/v3`는 확장 검증으로 각각 +315/+310/+315ms를 적용한다.

기본 결과 폴더로 전체 분석을 실행하면 `result/<dataset>_real_sim_analysis.txt`도 함께 갱신한다.
TXT는 저장된 물리 배열·지표를 읽고 원본 해시와 파일 목록을 확인한 뒤 생성하며,
시행별·12관절별 오차, 발별 파형 및 문턱 기반 착지 시각 차이를 기록한다.

01은 시간 정렬된 물리값이다. 02는 같은 데이터의 q/dq에 배포 관절 순서·기본각·ONNX 고정 정규화를 적용한 무차원 값이다.
두 탭은 같은 시각·평가 마스크를 사용한다. 물리 단위의 시간 이동 전후 RMSE는 별도로 표시한다.
ddq·tau·발 힘의 배포 스케일은 없으며 두 탭에서 값을 유지한다.
실제 발 센서는 counts, sim 힘은 N이다. z-RMSE는 표준화된 파형 오차다.
S3의 `sim_foot_force_vec[:, :, 2]`는 접촉 시 음수로 기록되므로 고정 부호 −1을 적용해 양의 수직 하중으로 비교한다.
원본 벡터와 센서 counts는 수정하지 않는다. 관절 시간 이동의 검증 통과가 발 접촉 동기화까지 보장하지는 않는다.
수집 NPZ에는 정책 해시가 없어 배포 파일 자체만 검증하며 이 한계를 provenance에 기록한다.

## 저장 결과로 UI 또는 그림만 다시 생성

```bash
python3 src/Real2Sim/scripts/real2sim/export_ui.py
MPLCONFIGDIR=/tmp/go2_mapping_matplotlib /usr/bin/python3 -s src/Real2Sim/scripts/real2sim/plot_mapping.py
MPLCONFIGDIR=/tmp/go2_tau_matplotlib /usr/bin/python3 -s src/Real2Sim/scripts/compare_tau.py
```

토크는 01/02에서 동일하므로 그림도 `figures/torque/`에 한 번만 생성한다.
`*_foot_force_before_after.png`의 왼쪽은 PC 시각만 사용한 기준, 오른쪽은 검증된 공통 시간 이동 후다.
이동을 적용하지 못한 시행은 그림 제목에 REVIEW와 미적용 상태를 표시한다.

## 검증

```bash
python3 -m pytest -q src/Real2Sim/scripts/real2sim/tests src/Real2Sim/replay/tests
python3 -m pytest -q src/Real2Sim/scripts/real2sim/legacy/tests
```

UI 로직의 mock DOM/Canvas 검사는 `replay/tests/ui_smoke.js`다. 실제 브라우저 시각 검사와 구분한다.

## 발 접촉 동기화 진단

```bash
MPLCONFIGDIR=/tmp/go2_contact_matplotlib /usr/bin/python3 -s src/Real2Sim/scripts/diagnose_contact_sync.py
```

저장된 양의 sim 하중과 실측 counts를 사용해 모든 시행·발의 파형 상관, 추가 파형 이동 추정치,
착지 이벤트 잔차를 앞/뒤 구간에서 따로 확인한다. 결과는 `result/<dataset>/diagnostics/contact_sync/`의 JSON·CSV·그림이다.
착지는 앞 구간 p10–p90 범위의 15%/7% 히스테리시스와 6ms 확인으로 정의한다.
이 임계값 기반 시점 차이는 센서 응답·하중 파형에도 영향을 받는다. 진단 도구는 replay에 별도 발별 이동을 적용하지 않는다.
