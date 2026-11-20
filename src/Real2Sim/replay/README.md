# Go2 real/sim 로그 매핑 UI

기존 `data/real/1116_RL_base/`와 `data/sim/1116_RL_base/` 안의
**`0.2_1.npz`~`0.2_5.npz`만** 읽어 시간 정렬 기준과 실제 배포 정책 입력을 비교한다.
로그 수정, 정책 추론, 재시뮬레이션, 회귀 피팅, 물리·센서 교정은 수행하지 않는다.

```bash
bash src/Real2Sim/replay/run_replay.sh
```

브라우저: **http://127.0.0.1:8791**. 종료는 Ctrl+C. 다른 포트는 `--port 8792`.
호스트 Python 3, NumPy, SciPy가 필요하다. 실행 위치와 관계없이 위 데이터셋의 5쌍을 선택한다. 실행 중인 서버는 페이지를 새로고침한다.
데이터만 생성하려면 `python3 src/Real2Sim/replay/build_data.py`를 실행한다.

## 01 / 02 정의

- **01 보정 전 · 시간 정렬**: 기존 물리 관측을 공통 전처리하고 보행 위상까지 정렬한 기준 데이터.
- **02 보정 후 · 배포 매핑**: 01과 동일한 시간축·평가 샘플에서 q/dq를 실제 배포 정책의
  정규화 입력으로 변환한 결과를 그대로 표시한다. 숫자 역변환은 하지 않는다.

```text
q_actor  = (q_sdk[order] - default - mean_q) / denominator_q
dq_actor = (dq_sdk[order] - mean_dq) / denominator_dq
```

`order/default`는 실제 `policy.json`, `mean/denominator`는 같은 `policy.onnx` 입력 Sub/Div 상수다.
두 도메인에 같은 고정 배포 상수를 적용한다. 실제/시뮬 로그에서 배율·오프셋을 피팅하지 않는다.
UI에서는 같은 관절 이름으로 비교하도록 배열 인덱스만 다시 대응한다. 정규화나 기본각을 되돌리지 않는다.

**01 q/dq는 rad, rad/s이고 02 q/dq는 무차원이다. 전후 RMSE 수치로 물리 오차 개선률을 계산하면 안 된다.**
02는 정책 입력 공간의 불일치를 보여준다. ddq는 정책 입력에 없으므로 rad/s² 물리값을 유지한다.
발 힘의 배포 변환도 없으므로 counts/N과 z-RMSE는 두 탭에서 유지된다.
`action_scale=0.25`는 정책 출력과 목표각 사이의 변환이며 관측 q/dq의 스케일이 아니다.
기록된 목표각에서 역산한 action 좌표는 별도 저장하며 역사적 raw action과 같다고 주장하지 않는다.

`deployment_mapping.py`는 배포 ONNX/checkpoint 해시와 samples.json/parity.npz 관절 관측 100프레임을 검사한다.
`deployment_normalization.json`은 실제 ONNX에서 추출한 고정 통계다. ONNX 해시가 다르면 실행을 거부한다.
추출기는 ONNX가 설치된 환경에서 파일의 상수만 읽는다. 추론이나 물리 엔진 실행은 없다.

```bash
python src/Real2Sim/replay/extract_deployment_normalization.py --policy-dir <배포폴더>
```

## 공통 처리와 UI

- 실제 tick 중복 제거, 공통 200 Hz 보간. 양쪽 q/dq에 15 Hz 저역통과 필터를 적용하고 ddq는 필터 dq 미분으로 계산.
- 시작 구간 발 접촉 이벤트로 상수 시간 이동을 추정하고 끝 구간으로 검증. 모든 sim 채널에 같은 이동 사용.
  관절별/발별 이동과 시간축 늘이기는 없다. 위상 정렬은 하드웨어 지연 측정이 아니며 미통과 시행은 잠정 정렬이다.
- 12관절 페이지, q/dq/ddq 비교, 과거 0.5초 이동 RMSE. 그래프와 카드의 단위가 탭에 따라 바뀐다.
- ▶/Space, 배속, 시간 슬라이더, 그래프 클릭/드래그와 확대 범위. 두 탭은 같은 실제 시각을 유지한다.
- 발 힘 카드: 실제 센서 counts와 sim world Fz(N)는 독립 축으로 표시한다.
  z-score는 파형 오차용이며 N 단위 힘 오차가 아니다. 01의 평균·표준편차를 두 탭에서 동일하게 사용한다.
- 오른쪽에 매핑 경로, 선택 관절의 실제 배포 배율/오프셋, 단위와 RMSE 의미를 표시한다.

## 산출물

- `web/data/`: UI JSON, 스키마 `go2-actor-mapping-v4`. 관측 공간과 단위를 탭별로 기록한다.
- `../result/data_mapping/*_mapping.npz`: 전후 관측 배열과 공통 시간/마스크,
  정책 순서의 q 상대각·정규화 q/dq, 기록 목표각의 action 좌표. `real_q` 등 기존 키는 01 물리값의 별칭이다.
- `mapping_summary.json`, `metrics.csv`: 단위를 포함한 관절별·발별 지표와 배포 상수 출처/해시.
- `analysis.md`, `figure/`: 분석과 별도 단위 축의 그림. RMSE 요약에서 물리값과 정책 입력을 같은 축으로 겹치지 않는다.

`build_data.py --data-dir ... --dataset 1116_RL_base --output-dir ... --result-dir ... --policy-dir ...`로 경로를 지정한다.
기존 `figure_after`의 명령 재생 결과는 읽지 않는다. 서버는 `web/`만 loopback으로 제공한다.

정적 그림은 시스템 NumPy/Matplotlib 환경에서 생성한다. UI 실행에는 Matplotlib이 필요 없다.

```bash
MPLCONFIGDIR=/tmp/go2_mapping_matplotlib /usr/bin/python3 -s src/Real2Sim/replay/plot_mapping.py
python3 -m pytest -q src/Real2Sim/replay/tests
```

테스트는 실제 입력 SHA, 전후 공통 시간/평가 마스크, 정확한 정규화 공식과 RMSE 배율,
역변환 부재, 유지해야 하는 ddq/발 힘을 검사한다. `tests/ui_smoke.js`는 mock DOM에서
관절/탭 이동, 단위 갱신, RMSE/발 힘 모드, 재생과 Canvas 호출을 검사한다. 실제 브라우저 시각 검증과는 구분한다.

## 토크 비교

`ls_tau`를 real 추정 모터 토크 / sim 액추에이터 토크로 비교한다. 양쪽에 동일한 15 Hz 필터,
공통 시간 정렬과 정상 보행 평가 마스크를 적용한다. UI에서 12관절의 tau 그래프·RMSE/MAE/bias,
RMSE 항목 선택의 tau(과거 0.5초)를 확인한다. 배포 파일에 tau 보정 규약이 없어 두 탭 모두 N·m을 유지한다.
`action_scale`을 토크에 적용하지 않는다.

정적 그림과 표는 `scripts/compare_tau.py`로 생성한다. 이 명령은 기존 로그로 UI 데이터를 갱신한 뒤
`result/figure_before`와 `result/figure_after`에 각 5회 전체/확대 그림, 요약 그림, CSV/JSON/분석을 추가한다.

```bash
MPLCONFIGDIR=/tmp/go2_tau_matplotlib /usr/bin/python3 -s src/Real2Sim/scripts/compare_tau.py
```

이미 갱신한 매핑 NPZ로 그림만 생성하려면 `--from-export`를 추가한다. 추론·재시뮬레이션은 없다.
