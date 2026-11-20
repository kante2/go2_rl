# 기존 real/sim 로그의 배포 관측 매핑

```bash
bash src/Real2Sim/scripts/real2sim/run_converter.sh
bash src/Real2Sim/replay/run_replay.sh
```

첫 명령은 데이터 생성, 두 번째는 데이터 생성과 UI 서버 실행이다. UI를 사용하면 두 번째만 실행한다.
01은 시간 정렬된 물리 관측, 02는 같은 관측에 실제 배포 policy.json/ONNX 규약을 적용한 q/dq 정책 입력이다.
02에서 숫자 역변환은 하지 않는다. 02 q/dq와 RMSE는 무차원이며 01의 물리 단위 오차와 직접 비교할 수 없다.
ddq/발 힘은 배포 변환 규약이 없어 물리값을 유지한다. 회귀 피팅과 재시뮬레이션은 수행하지 않는다.

결과는 `src/Real2Sim/result/data_mapping`, UI 데이터는 `src/Real2Sim/replay/web/data`에 저장한다.
옵션은 `--data-dir`, `--output-dir`(UI JSON), `--result-dir`(분석 결과), `--policy-dir`(배포 폴더)다.
상세 공식과 고정 ONNX 정규화 계수 출처는 [replay README](../../replay/README.md)를 참조한다.

## 과거 명령 재생 코드

`real2sim_converter.py`, `analyze_converter.py`, `full_record_replay.py`와 `figure_after` 산출물은 과거 별도 실험으로 보존한다.
현재 기본 실행과 UI에서 사용하지 않는다.
