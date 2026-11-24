# Go2 QuietWalk 1124

[QuietPaw 논문](../../../quietpaw_.pdf)의 CNCP를 Go2 MuJoCo/mjlab 환경에서 학습한다. 평지와 험지 태스크 이름은 각각 `Unitree-Go2-QuietWalk-1124-Flat`, `Unitree-Go2-QuietWalk-1124-Rough`다. 기존 QuietPaw의 분해 critic, 관측, 보행 보상, PPO 저장소, PID 및 실행기를 공유한다.

## 실행

저장소 루트의 Go2 학습 컨테이너에서 실행한다. 논문 규모의 기본값은 환경 4,000개, 학습 10,000회다. GPU 메모리에 따라 `--env.scene.num-envs`를 조절할 수 있다.

```bash
python scripts/train.py Unitree-Go2-QuietWalk-1124-Flat

# 작은 GPU에서 학습 경로 확인
python scripts/train.py Unitree-Go2-QuietWalk-1124-Flat \
  --env.scene.num-envs 8 --agent.max-iterations 2 --agent.num-steps-per-env 4

# 학습 로그에 표시된 실행 폴더와 체크포인트 번호로 RUN/model_N.pt를 바꾼다.
python scripts/play.py Unitree-Go2-QuietWalk-1124-Flat \
  --checkpoint-file logs/rsl_rl/go2_quietwalk_1124/RUN/model_N.pt --quietness 0.8

python -m src.locomotion_rl.go2_quietwalk_1124.evaluate \
  --checkpoint-file logs/rsl_rl/go2_quietwalk_1124/RUN/model_N.pt \
  --limits '(0,0.5,1)' --velocities '(1,)' --output logs/quietwalk_1124.csv
```

험지 평가는 `--task Unitree-Go2-QuietWalk-1124-Rough`와 해당 지형 체크포인트를 사용한다. 학습 로그와 체크포인트는 `logs/rsl_rl/go2_quietwalk_1124/`에 저장된다. 환경마다 하나의 ε를 고정하므로 여러 조건을 학습하려면 병렬 환경 수도 둘 이상이어야 한다. ONNX 정책의 마지막 입력은 ε다. 기본 설정에서 `quietness=0.8`은 ε=0.2에 대응한다.

## 학습 정의

- 식 (2)~(5). ε를 [0, 1]에 균등 배정한다. PID는 에피소드 시작부터의 할인 비용 합과 ε를 비교한다. 미완료 에피소드의 비용과 할인율 거듭제곱은 다음 rollout에 이어지고, 남은 미래 비용은 cost critic으로 추정한다. 실제 종료에서는 누적을 끝내고 시간 제한에서는 reset 직전 관측으로 bootstrap한다. 재개 시 환경은 새 에피소드에서 시작한다.
- 식 (6)~(10). 상태 특징과 ε 가중치의 64차원 내적으로 reward와 cost critic을 각각 구성한다. Actor와 critic 상태 인코더의 은닉층은 512, 256, 128이다. 두 critic은 각각 제곱오차를 학습한다. 해당 rollout의 기존 λ로 advantage를 결합하고 다음 rollout의 λ를 갱신한다. 보행 위상 보상은 2,000회 중심의 sigmoid로 줄인다.
- 물리 설정. Go2는 50 Hz에서 12개 관절 위치 명령을 받는다. 논문의 실기 PD 게인 25/0.6을 시뮬레이터에 적용했다. 각 정책 스텝의 발별 최대 접촉력과 최초 접촉 순간의 직전 수직 하강 속도를 사용한다. 한 스텝에 여러 번 착지하면 최대 제곱속도를 취한다. 힘 채널과 속도 site를 같은 발 이름으로 대응시킨다. 실패는 종료와 기존 보행 보상에서 처리한다.
- 재현 범위. 물리 엔진과 보행 보상은 저장소의 mjlab 이식을 사용한다. PID 게인, 조건 MLP의 은닉층, 보행 커리큘럼의 전환 폭, 수직 충돌 속도 근사는 구현 선택이다.

## 식 (1)의 부호와 정규화

인쇄된 식의 원점값을 `S₀ = 4 × [0.5 exp(Fmax/50) + 0.5]`라 하고 원래 식의 출력을 S라 한다. S는 충격이 커지면 감소하는데, 본문과 그림 2에서는 비용이 증가한다. 논문은 정규화 구간 [0, 1]만 제시한다. 여기서는 원점값으로 정규화하고 두 해석을 선택할 수 있게 했다.

| 학습 옵션 | 스텝 비용 | 해석 |
|---|---|---|
| `--env.noise-cost.orientation increasing` (기본) | 1 − S/S₀ | 그림 2처럼 강한 충격에 큰 비용 |
| `--env.noise-cost.orientation printed` | S/S₀ | 인쇄된 식의 방향 유지. 최소화하면 강한 충격을 선호할 수 있음 |

식 (1)에 제시된 가중치 0.5/0.5와 척도 50 N, 0.5 m²/s²를 사용한다. PDF에서 수치가 제공되지 않은 Fmax의 기본값은 50 N이다. `--env.noise-cost.force-threshold`로 변경할 수 있다. 제어 주기나 (1 − γ)를 비용에 곱하지 않으므로 스텝 비용은 [0, 1]이어도 할인 합은 1을 넘을 수 있다. ε=0은 가장 엄격한 제약이다.

## 평가와 검증

평가 기본값은 표 I의 ε 여덟 개와 0.5~2.25 m/s 범위다. CSV에는 환경별 스텝 평균 비용 기준 위반량, 학습 PID와 같은 할인 비용 기준 위반량, 속도 추종 오차, 실패 종료 횟수, 같은 속도에서의 Pareto 여부를 기록한다. 각 CSV는 한 체크포인트 결과다. 논문의 3개 학습 seed 비교에는 독립적으로 훈련한 체크포인트 3개가 필요하다. 실제 소음 저감량은 실로봇 음압 측정으로 검증해야 한다.

```bash
QUIETWALK_GPU_TESTS=1 MUJOCO_GL=egl \
  python -m unittest discover -s src/locomotion_rl/go2_quietwalk_1124/tests -v
```
