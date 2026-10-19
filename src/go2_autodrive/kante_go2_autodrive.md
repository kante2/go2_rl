# go2_autodrive 실행

호스트 터미널에서 모델을 선택하면 Docker 안에서 MuJoCo, 웹 뷰어,
경로 플래너, UDP↔ROS 브리지를 함께 실행한다.

```bash
cd /home/kante/go2_rl
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive mujoco.launch.py model:=quietpaw_long
```

| model 인자 | config/model_pt 아래 폴더 | 정책 입력 |
|---|---|---|
| baseline | baseline | 47개 |
| quietpaw | quietpaw | 48개 |
| quietpaw_long | quiet_paw_long | 48개, 학습된 LongStride 주기 |

기본으로 선택한 폴더의 `model_<번호>.pt` 중 가장 큰 번호를 사용한다.
특정 체크포인트를 선택하려면 파일명을 전달한다.

```bash
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive mujoco.launch.py \
  model:=quietpaw_long checkpoint:=model_29999.pt quietness:=0.5
```

브라우저에서 `http://localhost:8080`을 연다. 다른 포트는 `viewer_port:=8081`로 지정한다.
모델을 바꾸려면 실행 터미널에서 Ctrl+C로 종료하고 다른 `model`로 다시 실행한다.
QuietPaw의 `quietness`는 0~1이며 생략하면 `config/<model>.json`의 공통 값을 사용한다.
시뮬레이션은 PT를 직접 불러오므로 ONNX 준비가 필요 없다.

다른 호스트 터미널에서 키보드를 실행한다.

```bash
cd /home/kante/go2_rl
bash scripts/run_go2_docker.sh ros2 run go2_autodrive keyboard_waypoint
```

기본 ROS_DOMAIN_ID는 42이다. 변경할 때는 두 터미널 모두 같은 값으로 맞춘다.
기존 `simulation.launch.py`로 플래너/브리지를 이미 실행했다면
`mujoco.launch.py navigation:=false`로 MuJoCo만 실행한다.

화면 없는 제한 실행:

```bash
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive mujoco.launch.py \
  model:=baseline viewer:=none navigation:=false steps:=100
```

새 환경에서 launch 파일 설치/갱신이 필요하면 컨테이너에서 한 번 실행한다.

```bash
cd /opt/go2_ros_ws
colcon build --base-paths /workspace/unitree_rl_mjlab/src/go2_autodrive \
  --packages-select go2_autodrive --symlink-install
source install/setup.bash
```

## 실로봇

실로봇도 `baseline`, `quietpaw`, `quietpaw_long` 중 선택한다.
시뮬과 동일한 `config/<model>.json`을 읽는다. `checkpoint: null`이면 모델 폴더에서 최신 학습 번호를 선택한다.
특정 PT는 `checkpoint:=model_29999.pt`처럼 파일명으로 지정한다.
기존 LongStride 전용 실행 파일도 유지되지만, 세 모델의 공통 실행은 아래 launch를 사용한다.

새 모델을 가져온 뒤 준비한다. 로봇 연결 없이 ONNX 출력과 관측 매핑을 검증하고
`artifacts/go2_real/<모델 종류>/<PT 해시>/`에 결과를 저장한다.
함께 가져온 `policy.onnx`가 있으면 PT와 비교해 사용한다.
PT에서 새로 변환하려면 `export_onnx:=true`를 추가한다. 원본 모델은 수정하지 않는다.

```bash
cd /home/kante/go2_rl
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive real.launch.py \
  model:=quietpaw_long mode:=prepare
```

| mode | 동작 |
|---|---|
| prepare | 모델 준비 및 출력/관측 매핑 검증, 로봇 연결 없음 |
| check | 준비된 모델의 hash·종류·입출력 크기 검사, 로봇 연결 없음 |
| dry | 실로봇 센서 수신과 추론, 모터 출력 없음 (기본값) |
| live | 사전 검사와 ARM 입력 후 모터 제어 |
| keyboard | 선택한 모델 JSON의 통신 포트/속도를 사용하는 키보드 |

실로봇 센서와 추론을 10초 확인한다. `interface`는 로봇 연결 유선 인터페이스에 맞춘다.

```bash
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive real.launch.py \
  model:=quietpaw_long mode:=dry interface:=enp3s0 seconds:=10
```

실제 제어는 대화형 터미널에서 실행한다. 로봇을 지지한 상태에서 `ARM`을 입력하고
`REAL POLICY READY`가 나올 때까지 기다린다. launch는 현재 터미널을 제어 프로세스에
그대로 전달한다. 입력 확인을 자동으로 건너뛰지 않는다.

```bash
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive real.launch.py \
  model:=quietpaw_long mode:=live interface:=enp3s0 quietness:=0.5
```

다른 터미널에서 실로봇용 키보드를 실행한다. 시뮬레이션의 waypoint 키보드와 다르다.
컨트롤러에 `config:=...`를 전달했다면 키보드에도 같은 파일을 전달한다.

```bash
cd /home/kante/go2_rl
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive real.launch.py model:=quietpaw_long mode:=keyboard
```

W/S 전후, A/D 좌우, J/L 회전, Space 목표 속도 0, Q 키보드 종료.
제어 터미널 Ctrl+C 또는 리모컨 L2+B로 damping 정지한다.
모델 교체는 제어기를 종료한 뒤 다른 `model`로 재실행하고 다시 ARM을 입력한다.

## 모델별 공통 JSON

| model 선택값 | 시뮬/실로봇 공통 설정 | PT 폴더 |
|---|---|---|
| baseline | config/baseline.json | config/model_pt/baseline |
| quietpaw | config/quietpaw.json | config/model_pt/quietpaw |
| quietpaw_long | config/quietpaw_long.json | config/model_pt/quiet_paw_long |

`mujoco.launch.py`와 `real.launch.py` 모두 모델 이름으로 같은 JSON을 자동 선택한다.
직접 Python runner를 실행할 때도 `--model`로 동일하게 선택한다. 생략 시 quietpaw_long이다.

모든 설정은 JSON 최상위에 있으며 `real`/`simulation` 구분이 없다.
모델 폴더, 체크포인트, quietness, 속도 제한, 포트는 두 실행 방식이 동일한 값을 읽는다.
`model_dir`는 JSON 위치 기준, JSON의 `checkpoint`는 모델 폴더 기준이다.
`checkpoint: null`이면 학습 번호가 가장 큰 PT를 선택한다.

공통 quietness는 0.2이며 baseline에서는 사용하지 않는다.
기존 값이 달랐던 속도 제한은 낮은 값으로 통일했다: 전진 0.5 m/s, 횡이동 0.2 m/s, 회전 0.35 rad/s.
공통 UDP 포트는 command_port=9901, telemetry_port=9902, keyboard_status_port=9903이다.
같은 포트를 사용하므로 시뮬과 실로봇 제어기는 하나씩 실행한다.
장애물/scan_range는 시뮬에서, 모터 보호/키보드 설정은 해당 기능에서 사용한다.

보행 주기, 관절 순서, PD gain 등 학습 관련 정보는 선택한 PT/검증된 policy.json에서 복원한다.
이 JSON은 실행 설정이며 학습에서 내보낸 정책 메타데이터를 대체하지 않는다.
명령어의 `checkpoint:=...`, `quietness:=...`는 JSON보다 우선한다.
`config:=...`로 다른 JSON을 지정할 때는 `model:=...`도 일치해야 한다.
Docker 실행 시 명시적 config 경로는 컨테이너 경로(`/workspace/unitree_rl_mjlab/...`)를 사용한다.
키보드도 제어기와 동일한 model/config로 실행한다.

기존 simulation.json, real.json, real_quietpaw_long.json과 config/models 폴더는 모델 이름의 세 JSON으로 대체했다.
전용 `go2_real_udp_runner_queitpaw_long.py`도 이제 config/quietpaw_long.json을 읽는다.


## 보행 상태 UI

MuJoCo와 키보드 실행 후 별도 터미널에서:

```bash
cd /home/kante/go2_rl
bash scripts/run_go2_docker.sh ros2 launch go2_gait_monitor monitor.launch.py
```

브라우저에서 http://localhost:8081 접속. 모델 이름은 MuJoCo에서 자동으로 전달한다.
발별 수직 접촉력(N), 착지 후 최대 50ms 충격량(N·s), 목표/실제 속도,
접지 상태와 최근 그래프, CSV 저장을 제공한다. 실제 소음(dB) 측정은 아니다.
측정 코드 추가 전에 실행한 MuJoCo는 재실행한다.
빌드와 측정 정의는 [go2_gait_monitor README](../go2_gait_monitor/README.md)를 참고한다.


## 실로봇 속도 인자

`real.launch.py`의 `speed`(W/S m/s), `lateral_speed`(A/D m/s), `yaw_rate`(J/L rad/s)를 지원한다.
제어기에서 지정한 값을 키보드가 자동으로 수신하므로 양쪽에 반복 입력할 필요가 없다.

```bash
# 컨테이너 터미널 A
ros2 launch go2_autodrive real.launch.py model:=quietpaw_long mode:=live interface:=enp3s0 speed:=0.4 lateral_speed:=0.1 yaw_rate:=0.3
# 컨테이너 터미널 B — 모델은 A와 동일
ros2 launch go2_autodrive real.launch.py model:=quietpaw_long mode:=keyboard
```

B에서 속도 인자를 명시하면 해당 축에 우선 적용한다. 생략한 축은 A의 값을 따른다.
입력값은 JSON의 `max_*` 상한 이내인 양수여야 한다. 상한과 JSON 파일은 변경하지 않는다.
속도 지정 자체로 움직이지 않으며 ARM/READY 후 방향 입력이 필요하다.
자세한 교체/재시작 절차는 `go2_autodrive/launch/real_launch.txt` 8절을 참고한다.
