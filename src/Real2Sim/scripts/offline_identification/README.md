# 후속 단계: 오프라인 동역학 식별

현재 수집 단계에서는 실행하지 않는다. 실제 LowCmd를 MuJoCo에서 재생하고
관절·자세·각속도 예측 오차를 줄이는 유효 PD 배율과 지연을 CMA-ES로 찾는 기존 코드다.
기본 입력은 이전 단일 기록 `src/Real2Sim/real/NORMAL_20261019_155919.npz`다.
25개 실제 기록의 시행별 분할은 아직 구현하지 않았다.

* `inspect_dataset.py`: 원본 로그 진단.
* `prepare_dataset.py`: 시간축 정렬과 짧은 clip 생성.
* `mujoco_replay.py`, `replay_mujoco.py`: 실제 모터 명령 기반 상태 예측.
* `identify_dynamics.py`: hip/thigh/calf별 P·D 배율과 유효 지연 식별. 선택적 몸체 질량/관성 배율.
* `evaluate_real2sim.py`, `plot_real2sim.py`: 전후 오차 평가와 시각화.

실제 선속도·몸체 높이의 독립 정답이 없어 초기화는 추정에 의존한다.
발 raw counts는 N 단위 힘이 아니며 접촉 지표도 임시 threshold 기반이다.
식별된 배율은 미모델링 오차를 흡수할 수 있으므로 실제 모터 보정값으로 해석하면 안 된다.
수집기의 새 `go2-baseline-sim-v1` 형식은 입력 대상이 아니다.

의존성은 이 폴더의 `requirements.txt`, 기존 회귀 테스트는 `tests/`에 있다.

```bash
python3 -m pytest -q src/Real2Sim/scripts/offline_identification/tests
python3 src/Real2Sim/scripts/offline_identification/identify_dynamics.py --help
```
