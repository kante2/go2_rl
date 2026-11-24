# 저장된 Real2Sim 결과의 replay UI

```bash
bash src/Real2Sim/replay/run_replay.sh
```

`http://127.0.0.1:8791`에서 `result/1118_RL_base/`의 완성 결과를 읽어 표시한다.
다른 데이터셋은 `--dataset <name>`, 다른 결과 폴더는 `--result-dir <path>`, 다른 포트는 `--port 8792`다.
분석을 실행하거나 원본 로그·결과 파일을 변경하지 않는다. Python 표준 라이브러리만 필요하다.

결과가 없으면 먼저 다음 명령으로 생성한다.

```bash
bash src/Real2Sim/scripts/real2sim/run_converter.sh
```

## 파일 흐름

- `serve.py`는 HTML/JS/CSS를 `web/`에서 제공한다.
- `/data/index.json`, `/data/<trial>.json`은 `result/<dataset>/ui/`에서 읽는다.
- `app.js`는 시행·관절·01/02 탭·재생 시각·배속·표시 범위를 관리한다.
- `chart.js`는 저장된 배열을 Canvas에 그린다. 지연 추정·필터·지표 계산을 다시 하지 않는다.
- 서버는 manifest의 완료 상태·데이터셋·시행 목록을 확인한다.
- 동일한 결과 폴더와 web 파일을 제공하는 서버가 이미 있으면 재사용한다.
  다른 서비스나 다른 결과를 제공하는 서버가 포트를 점유하면 다른 포트를 안내한다.

## 화면의 의미

01은 시간 정렬된 물리값, 02는 같은 데이터의 q/dq를 배포 정책 입력으로 정규화한 값이다.
01 q/dq는 rad·rad/s, 02 q/dq는 무차원이다. ddq·토크·발 힘은 두 탭에서 유지한다.
카드의 시간 이동 전후 RMSE는 같은 물리 단위·평가 샘플로 비교한다.
검증된 시행별 공통 이동을 모든 sim 채널에 적용하며, 미검증 시행은 0ms와 사유를 표시한다.
±500ms 확장으로 정렬한 시행은 앞 구간에서 선택한 보행 주기를 뒤 구간에서 유지했음을 표시한다.
주기적인 뒤 구간에는 한 주기 차이의 동점이 남을 수 있으므로 하드웨어 지연으로 해석하지 않는다.
토크 오차가 증가하는 경우도 표시한다. 실제 발 counts와 sim N은 독립 축으로 그린다.

전처리·시간 보정·산출물 구조와 추가 명령은 [Real2Sim README](../README.md)에 설명되어 있다.
