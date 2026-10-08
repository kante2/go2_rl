# Go2 autonomous driving — MuJoCo / LAN 실로봇

`model_10000.pt`로 두 가지 실행 구성을 제공합니다. 시뮬레이션은 키보드 웨이포인트와 [Navigation-Physical-Experiment](https://github.com/Yuxin916/Navigation-Physical-Experiment)의 `localPlanner`·`pathFollower`를 사용합니다. 실로봇은 키보드 속도와 실제 IMU·관절 피드백으로 제어하며 MuJoCo에 측정 자세를 표시합니다.

## 실로봇 센서로 학습 보행 + MuJoCo에 실제 자세 표시

실로봇 실행 경로는 `실제 LowState(IMU·관절) → 학습 모델 → 관절 매핑 → 실제 Go2`의 폐루프입니다. 기존 `simulation.launch.py`, `run_go2_simulation.sh`, `keyboard_waypoint`의 **시뮬레이션 전용 구성은 그대로 사용**합니다.

실로봇에서는 `model_10000.pt`에서 관측 정규화까지 포함해 변환한 동일 정책 `artifacts/go2_real/policy.onnx`를 호스트 CPU에서 50Hz로 실행합니다. 모델 입력 47개는 실제 몸체 각속도·중력 방향, 키보드 속도 명령, 보행 위상, 실제 관절 위치·속도, 이전 action입니다. 출력은 기존 학습 기본 자세·action scale·관절 순서·게인으로 변환해 전송합니다. MuJoCo의 상태와 가상 접촉 정보는 이 경로의 정책 입력에 사용하지 않습니다.

MuJoCo는 별도 프로세스에서 **실제 관절 각도와 IMU 자세**만 표시합니다. 물리 시뮬레이션을 진행하거나 제어 명령을 생성하지 않습니다. 위치추정이 없으므로 몸체 xyz는 표시용 `(0, 0, 0.32)`m에 고정합니다. 화면의 위치·높이·바닥 접촉은 실제 측정 결과가 아닙니다. 수신이 끊기면 `STALE`로 표시하고 마지막 자세를 고정합니다. **뷰어 종료는 모터 정지가 아닙니다.** 모터 제어는 터미널 2에서 종료합니다.

이 경로는 위치추정 없이 사용하는 직접 키보드 속도 제어이며 플래너를 실행하지 않습니다. W/S는 몸체 기준 전진/후진, A/D는 좌/우 이동, J/L은 좌/우 회전입니다. 실제 위치와 장애물을 사용하는 자율주행은 독립 위치추정을 추가한 뒤 연결해야 합니다.

기존 실행을 종료한 후 다음 3개를 실행합니다. 처음 설치할 때는 아래 시뮬레이션 빌드와 `bash scripts/prepare_go2_real_robot.sh`로 메시지·모델 변환·호스트 실행 환경을 준비합니다. 현재 PC는 준비되어 있습니다.

**터미널 1 — MuJoCo 실제 자세 뷰어**

```bash
bash /home/kante/go2_rl/scripts/run_go2_real_viewer.sh
```

브라우저에서 <http://localhost:8080>을 직접 엽니다. 처음에는 실제 상태를 기다립니다. 시뮬레이션 전용 뷰어와 동시에 같은 8080 포트를 사용할 수 없습니다.

**터미널 2 — 실제 센서 기반 제어기, 우선 모터 송신 없이 확인**

```bash
bash /home/kante/go2_rl/scripts/run_go2_real_closed_loop.sh
```

`phase=[DRY READY]`에서는 실제 센서로 추론하고 화면에 상태를 전송하지만 모터 명령을 보내지 않습니다. 실로봇 제어를 시험할 때는 터미널 2를 종료한 뒤 **동일 명령 끝에 `--execute`를 붙입니다.** 서 있는 로봇을 지지한 상태에서 `ARM`을 입력하고 **`REAL POLICY READY:`**를 기다립니다. 이후 주기 로그에도 `phase=[REAL POLICY READY]`가 반복됩니다. 처음에는 지지를 유지하고 실제 관절·자세 응답을 검증해야 합니다. 폐루프 연결 검증만으로 학습 정책의 실보행 안정성이 검증되는 것은 아닙니다.

**터미널 3 — 실제 제어용 직접 키보드**

```bash
bash /home/kante/go2_rl/scripts/run_go2_real_keyboard.sh
```

- W/↑, S/↓: 전후 0.20 m/s. A/←, D/→: 좌우 0.12 m/s. J/L: 회전 ±0.25 rad/s. 방향 입력은 Space까지 유지합니다.
- Space: 속도 0을 요청하고 제한된 감속으로 서 있기 정책으로 전환합니다. Q와 키보드 Ctrl+C도 정지를 요청한 뒤 키보드를 종료합니다. 키보드 종료 후에는 타임아웃 보호도 적용됩니다.
- 키보드·상태 연결이 끊기면 이동을 해제하며, 재연결만으로 이전 이동을 재개하지 않습니다. 새 방향 입력이 필요합니다.
- 터미널 2 Ctrl+C / 리모컨 L2+B / 상태 지연·기울기·관절 오차 등: 0.5초 damping 후 모터 송신 종료. 몸체가 내려갈 수 있으므로 지지해야 합니다. 기본 Unitree 모드는 자동 복구하지 않습니다. 복구 방법은 아래 기본 모드 복구 절을 참고합니다.

로봇 DDS는 domain 0 / `enp3s0`이고, 로컬 UDP는 속도 9901, 뷰어 상태 9902, 키보드 상태 9903입니다. `config/real_closed_loop.json`에서 속도와 포트를 설정합니다. 뷰어·키보드는 로봇 DDS로 직접 명령을 보내지 않습니다. 기존 시뮬레이션 포트 9871/9872 및 웨이포인트 노드와 분리되어 있습니다.

키보드의 전후 속도는 실행 옵션으로 지정할 수 있습니다. 기본은 0.20 m/s이고, 전후 상한은 `config/real_closed_loop.json`의 `max_forward_speed`입니다. 예를 들어 `bash scripts/run_go2_real_keyboard.sh --speed 0.5`로 W/S에 ±0.50 m/s를 요청합니다. 이 값은 목표 속도이며 실제 측정 속도와 다를 수 있습니다. JSON의 상한·감속·전환 설정을 변경하면 터미널 2도 재시작해야 합니다. 로봇을 지지하고 Space로 이동을 해제한 뒤 터미널 2를 종료·재실행하며, 다시 ARM과 REAL POLICY READY 절차를 거칩니다. 이후 상한 이내의 키보드 속도 변경은 터미널 3 재실행으로 적용합니다. 높은 상한을 설정해도 해당 속도에서의 실제 보행 안정성이 보장되지는 않습니다.

**정지·후진 전환:** 전후 가속·감속을 각각 0.5 m/s²로 제한합니다. 정상 주기 기준 0.5에서 정지 명령까지 약 1초, 1.5에서는 약 3초이며, 전진 1.5에서 후진 -1.5로 전환하면 0을 거쳐 약 6초에 걸쳐 바뀝니다. Space를 누른 뒤에도 감속 구간이 있으므로 정지 거리가 필요합니다. 이는 정책에 입력하는 목표 속도의 변화 시간이며 실제 로봇의 제동 시간·거리를 보장하지 않습니다. `requested_cmd`는 키보드 요청, `cmd`는 정책에 적용 중인 속도입니다. 뷰어에도 두 값을 따로 표시합니다.

관절 목표 변화는 평상시 초당 12 rad(50Hz 정책 한 주기당 0.24 rad)로 제한합니다. 학습 관측은 속도 명령 크기가 0.1 미만이면 보행 위상을 즉시 0으로 바꿉니다. 이 관측·모델은 유지하며, 보행에서 정지로 바뀌는 순간부터 0.4초 동안 관절 목표 변화만 초당 3 rad(한 주기당 0.06 rad)로 줄입니다. `phase_transition_joint_rate`, `phase_transition_seconds`로 지정합니다. 실제 로그의 0.10→0.08 전환 직후 허벅지 목표가 급변한 문제를 완화하기 위한 처리입니다. 다음 관측의 이전 action에는 제한 후 실제 사용한 action을 넣습니다.

원래 정책 출력의 관절 범위 검사는 제한 전에 수행합니다. 실제 관절과 전송 목표의 **0.65rad 오차 한계**, 기울기·상태 지연·리모컨 비상정지 보호는 유지합니다. 키보드 연결 중단은 정책 속도를 즉시 0으로 만들고 위상 전환의 관절 목표 제한은 적용됩니다. 비상정지·고장은 정상 감속이나 전환 처리를 기다리지 않고 damping으로 종료합니다.

오차 한계를 넘으면 `joint`, `sdk_index`, 실제/목표 각도, 요청/적용 속도가 오류에 표시됩니다. 종료 damping 후 `log/go2_real_closed_loop_fault.json`에 마지막 약 2초의 관측·목표·정지 전환 잔여 시간 및 고장 순간 관절 오차를 기록합니다. 모델·기계·센서 원인을 구분할 다음 진단 자료이며, 보호 한계를 높여 재시도하는 용도가 아닙니다.

`scripts/check_go2_control_transitions.py`는 Docker 안에서 실제 모델과 MuJoCo로 각 속도의 정지/후진을 보행 위상 6개씩 검사합니다. 기본은 0.5·0.78·1.5 목표이며 `--speeds`, `--modes`, `--forward-seconds`로 범위를 지정합니다. 기본 시험은 같은 초기 상태와 물리 파라미터를 사용하고, `--randomized`는 play 설정의 무작위 물리 파라미터를 유지합니다. 검사 주기는 정책의 50Hz이며 실로봇 모터 루프의 500Hz 전체를 검증하지 않습니다. 첫 보호 조건이 발생한 시험은 실패로 기록하고 이후 결과를 집계하지 않습니다.

이번 정지 전환 수정으로 0.5·0.78 목표의 정지/후진 24조건은 기존 0.65rad 한계 아래였습니다. `--speeds 1.5 --modes space --forward-seconds 1.56`의 가속 도중 정지 6조건도 통과했습니다. 저장된 실제 센서의 마지막 세 프레임을 재생한 계산에서는 목표 필터 수정으로 마지막 오차가 약 0.76→0.55rad로 줄었습니다. 센서 재생은 수정된 모터 명령에 따른 실제 로봇 응답을 포함하지 않습니다. 1.5까지 계속 가속하는 명목 조건은 수정 전후 모두 약 1.43 명령에서 원래 모델 출력이 관절 범위를 벗어나 정지 이전에 실패했습니다. 기본 전체 시험이 실패하는 이유이며 고속 보행의 별도 미해결 사항입니다. 수정 후 실제 보행은 검증하지 않았습니다.

실제 모터 없이 검증하려면 `scripts/check_go2_real_policy.py`로 정책/관측 일치를 확인하고, `scripts/check_go2_real_transport.py --closed-loop`로 loopback domain 94의 가상 Go2에서 준비·제어 전환·직접 키보드·CRC·damping을 확인합니다. `--closed-loop --fault keyboard_timeout`, `--closed-loop --fault stale_state`, `--closed-loop --fault estop`으로 각 중단 동작을 검사합니다.

2026-10-07 검증: 단위 테스트 46개 통과. 100개 프레임에서 원본 정책/ONNX 최대 오차 `8.35e-7`, 실제 관측 구성 최대 오차 `1.67e-6`을 확인했습니다. 실제 LowState로 약 50Hz 추론, 실제 키보드 W/Space, 뷰어 HTTP 200과 측정 자세 표시를 **모터 송신 없이** 확인했습니다. MuJoCo 관절 이름별 각도·속도와 몸체 quaternion 일치, 물리 적분 없이 xyz 고정도 검사했습니다. 가상 Go2에서는 정상 종료·키보드 타임아웃·LowState 중단·L2+B·Ctrl+C 처리를 통과했습니다. 기존 시뮬레이션 전용 구성에서도 W 한 번으로 약 3.15m 연속 전진하고 Space로 정지했습니다. 이 변경으로 실제 보행 시험을 수행하지는 않았습니다.

## 실로봇 환경 준비와 기본 모드 복구

처음 설치할 때는 시뮬레이션 빌드 후 프로젝트 루트에서 `bash scripts/prepare_go2_real_robot.sh`를 실행합니다. 공식 Unitree 메시지, 호스트 `.venv_go2_real`, ONNX 변환·매핑 파일을 준비합니다. `model_10000.pt`는 `src/go2_autonomous_driving/model_pt/`에 배치합니다. 체크포인트를 바꾸면 다시 준비하여 PT·ONNX·매핑의 해시를 일치시킵니다. 파일 내용이 같고 위치만 바뀐 경우에는 ONNX를 다시 변환할 필요가 없습니다.

호스트 LAN은 `enp3s0 = 192.168.123.99/24`, Go2는 `192.168.123.161`입니다. 다른 인터페이스는 실행 스크립트에 `GO2_NETWORK_INTERFACE=인터페이스명`으로 지정합니다. 현재 폐루프는 실제 `/lowstate`만 사용하며, 플래너 연결에는 별도 위치추정 구현이 필요합니다.

damping으로 종료되면 로봇을 지지하고 학습 제어기를 종료한 뒤 다음 명령으로 기본 모드를 복구합니다. `NORMAL` 입력 시 Unitree MotionSwitcher의 `SelectMode("normal")`을 호출하며 자세가 바뀔 수 있습니다. 필요하면 복구 후 앱/리모컨으로 일어서기를 수행합니다.

```bash
bash /home/kante/go2_rl/scripts/recover_go2_normal.sh --restore
```

`--restore` 없이 실행하면 현재 모드만 읽습니다. 학습 제어기의 모터 송신이 활성화되어 있으면 복구를 거부합니다.

가상 로봇 통신 검증은 다음 명령을 사용합니다. DDS domain 94와 loopback `lo`를 강제하여 실제 로봇에 연결하지 않습니다. `--fault estop`, `--fault stale_state`, `--fault keyboard_timeout`, `--fault joint_tracking`, `--fault sigint`로 각 중단 조건을 검사할 수 있습니다.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="src/go2_autonomous_driving:${PYTHONPATH:-}" \
  .venv_go2_real/bin/python scripts/check_go2_real_transport.py
```

## MuJoCo에서 실행

시뮬레이션의 플래너·브리지·MuJoCo·학습 모델·키보드를 모두 Docker 안에서 실행합니다. 호스트에는 Docker와 NVIDIA 드라이버가 필요하며 웹 화면은 호스트 브라우저에서 봅니다. Docker 준비·학습·실로봇별 구성은 [Docker 안내](../../docker/README.md)에 있습니다.

최초 빌드는 호스트 프로젝트 루트에서 `bash scripts/setup_go2_docker.sh`로 수행합니다. 호스트 프로젝트는 컨테이너의 `/workspace/unitree_rl_mjlab`에 연결됩니다. ROS는 이미지 안의 `/opt/go2_ros_ws/install`을 사용하므로 호스트의 `install/setup.bash`를 컨테이너에서 source하지 않습니다.

호스트 터미널을 3개 열고 각 터미널에서 컨테이너에 접속합니다. ROS 환경은 자동으로 설정됩니다.

```bash
sh /home/kante/go2_rl/scripts/enter_go2_docker.sh
```

**터미널 1 — 컨테이너에서 플래너와 ROS 브리지**

```bash
ros2 launch go2_autonomous_driving simulation.launch.py
```

**터미널 2 — 컨테이너에서 MuJoCo와 학습 모델**

```bash
bash scripts/run_go2_simulation.sh --viewer viser
```

호스트 브라우저에서 <http://localhost:8080>을 엽니다. 체크포인트는 호스트의 `/home/kante/go2_rl/src/go2_autonomous_driving/model_pt/model_10000.pt`입니다. 초기화가 끝나고 `Loaded ... iter=10000`이 출력될 때까지 기다립니다.

기본 모델 경로는 `config/simulation.json`의 `checkpoint`로 지정하며 상대 경로는 해당 JSON 파일의 디렉터리를 기준으로 해석합니다. 기본값 `../model_pt/model_10000.pt`는 패키지의 `model_pt/`를 가리킵니다. `--checkpoint /경로/다른모델.pt`를 주면 설정값 대신 해당 파일을 사용합니다.

**터미널 3 — 컨테이너에서 키보드 웨이포인트**

```bash
ros2 run go2_autonomous_driving keyboard_waypoint --ros-args \
  --params-file src/go2_autonomous_driving/config/navigation.yaml
```

키보드 터미널에 포커스를 두고 입력합니다.

| 키 | 동작 |
| --- | --- |
| `W` / ↑ | 입력 당시 전방 방향으로 계속 이동 |
| `S` / ↓ | 입력 당시 후방 방향으로 계속 이동 |
| `A` / ← | 입력 당시 왼쪽 방향으로 계속 이동 |
| `D` / → | 입력 당시 오른쪽 방향으로 계속 이동 |
| Space | 속도 명령을 0으로 설정 |
| `Q` / Ctrl+C | 정지 명령 후 키보드 노드 종료 |

키를 한 번 누르면 입력 당시의 방향을 월드 좌표로 기억하고, 현재 위치에서 그 방향으로 1m 앞의 목표를 0.2초마다 갱신합니다. 키에서 손을 떼어도 계속 이동하며 Space를 누르면 목표 갱신과 이동 명령을 중단합니다. 다시 방향 키를 누르면 그때의 로봇 방향을 기준으로 진행 방향을 바꿉니다. 플래너가 장애물을 피해 몸체를 회전하더라도 처음 선택한 이동 방향은 유지합니다. 후방·측방 명령은 플래너가 회전과 이동으로 추종합니다.

새 방향 입력에는 새 경로를 기다리고, 같은 방향의 목표 갱신에는 현재의 유효한 경로를 사용하면서 재계획합니다. 따라서 목표 갱신마다 속도가 0으로 끊기는 것을 피합니다. 로컬 플래너의 `/way_point` 입력은 launch에서 `/go2/planner_way_point`로 연결하며, 브리지가 새 목표와 연속 갱신을 검사한 뒤 전달합니다.

기존처럼 한 번에 1m만 이동하고 목표에서 0.2m 이내에 도달하면 정지하려면 `continuous:=false`로 실행합니다.

```bash
ros2 run go2_autonomous_driving keyboard_waypoint --ros-args -p continuous:=false
```

속도 0에서도 학습 정책은 자세 유지를 위해 계속 관절 action을 출력하므로 작은 몸체 움직임은 있을 수 있습니다.

앞으로 내다보는 목표 거리 변경 예시:

```bash
ros2 run go2_autonomous_driving keyboard_waypoint --ros-args -p waypoint_distance:=2.0
```

## 파일과 설정

```text
go2_rl/
├── src/go2_autonomous_driving/      # ROS 2 서브 패키지
│   ├── go2_autonomous_driving/      # Python 패키지
│   │   ├── sim/                    # MuJoCo·플래너 브리지·웨이포인트 키보드
│   │   ├── real/                   # 실로봇 제어·매핑·상태 뷰어·직접 키보드
│   │   ├── core.py                 # 양쪽에서 사용하는 공통 자료형·명령 검사
│   │   └── __init__.py
│   ├── config/                     # 설정
│   ├── model_pt/                   # 학습 체크포인트 model_10000.pt
│   ├── launch/                     # 플래너·브리지 실행
│   ├── resource/                   # ROS 패키지 등록
│   ├── test/                       # 제어 로직 회귀 테스트
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   ├── .gitignore
│   └── README.md
├── scripts/                        # 프로젝트 환경용 실행·빌드·검증 도구
├── third_party/go2_navigation/      # 외부 의존 소스 (자동 준비, Git 제외)
├── build/                          # 빌드 결과 (Git 제외)
├── install/                        # ROS 환경 (Git 제외)
└── log/                            # 빌드·검증 로그 (Git 제외)
```

바깥쪽 `go2_autonomous_driving`은 ROS 패키지 디렉터리이고, 안쪽의 같은 이름 디렉터리는 Python 모듈입니다. ROS 2 Python 패키지의 표준 구조입니다. 다른 ROS 작업 공간에서도 이 패키지를 `src/` 아래에 두고 빌드할 수 있으며, `local_planner` 의존성이 먼저 준비되어 있어야 합니다. MuJoCo 실행에는 상위 `go2_rl`의 Go2 학습 환경과 모델이 필요합니다.

Python 코드는 실행 환경에 따라 `sim/`과 `real/`로 나뉩니다. 실로봇 상태를 MuJoCo로 표시하는 뷰어도 실제 센서 수신용이므로 `real/`에 있습니다. 루트 `scripts/run_go2_*.sh`와 ROS 실행 이름은 그대로 사용합니다. `python -m`을 직접 사용하는 경우에는 `go2_autonomous_driving.sim.simulation`, `go2_autonomous_driving.real.real_closed_loop`처럼 새 경로를 지정합니다. 기존 설치 환경을 업데이트할 때는 변경된 `setup.py` 진입점을 반영하도록 해당 ROS 작업 공간에서 `go2_autonomous_driving` 패키지를 다시 빌드합니다.

| 파일 | 역할 |
| --- | --- |
| `go2_autonomous_driving/sim/keyboard_node.py` | 방향 유지, 새 목표 및 연속 목표 갱신 발행 |
| `go2_autonomous_driving/real/real_closed_loop.py` | 실제 IMU·관절로 학습 정책 실행, 직접 키보드 속도 수신 |
| `go2_autonomous_driving/real/real_state_viewer.py` | 실제 관절·IMU를 MuJoCo 모델에 반영하는 읽기 전용 화면 |
| `go2_autonomous_driving/real/keyboard_velocity.py` | 위치추정 없이 사용하는 실로봇용 키보드 속도 제어 |
| `go2_autonomous_driving/real/real_robot.py` | 폐루프에서 사용하는 공통 DDS·모터 송신·정책 스레드·종료 처리 |
| `go2_autonomous_driving/real/real_control.py` | 관측·관절 순서 매핑과 Unitree CRC |
| `go2_autonomous_driving/real/control_transitions.py` | 속도 감속 및 정지 전환 시 관절 목표 변화 제한 |
| `go2_autonomous_driving/real/direct_control.py` | 명령 세션·타임아웃·재연결 및 상태 패킷 검증 |
| `config/real_closed_loop.json` | 실제 피드백 제어와 뷰어·키보드의 속도·포트·중단 조건 |
| `launch/simulation.launch.py` | 외부 저장소의 실제 C++ `localPlanner`, `pathFollower` 실행 |
| `go2_autonomous_driving/sim/ros_bridge.py` | MuJoCo 상태를 ROS로 발행, 플래너 속도를 제한·검사 후 UDP 전달 |
| `go2_autonomous_driving/sim/simulation.py` | 기존 Go2 Flat 환경·정책 로드, 외부 속도를 `vel_command_b`에 입력 |
| `go2_autonomous_driving/core.py` | 좌표 변환, 속도 제한, 오래된 명령 거부, 목표 도달 판정 |
| `config/simulation.json` | 체크포인트, UDP 포트, 속도 상한, 물리 장애물 배치 |
| `config/navigation.yaml` | 브리지 속도 상한·타임아웃, 연속 이동 모드·목표 거리·갱신 주기 |

프로젝트 루트의 `scripts/build_go2_navigation.py`는 플래너 의존성과 패키지를 빌드하고, `scripts/run_go2_simulation.sh`는 Docker에서 모델과 웹 뷰어를 실행하며, `scripts/check_go2_navigation.py`는 전체 연결을 검증합니다. `build_go2_navigation.py --workspace-root /경로`로 빌드 작업 공간을 따로 지정할 수도 있습니다.

기본 속도 상한은 전후 0.50m/s, 좌우 0.20m/s, yaw 0.50rad/s입니다. 상한을 바꾸면 시뮬레이션 JSON, ROS YAML, launch의 플래너 설정을 함께 맞춥니다. 설정을 변경한 뒤에는 플래너와 시뮬레이션을 다시 실행해야 반영됩니다. UDP는 동일 컴퓨터의 `127.0.0.1:9871/9872`를 사용합니다. 동시에 두 세트를 실행할 때는 각각 포트를 변경해야 합니다.

기존 보행 정책은 명령 크기 0.1 미만에서 위상 관측을 0으로 만듭니다. 최대 속도 0.50m/s에 맞춰 `slowDwnDisThre=0.7`로 설정해 단일 목표 모드에서는 목표 근처에서 감속하고, 명령이 0.1 아래로 내려가기 전에 0.2m 도달 판정으로 정지하도록 합니다. 연속 모드에서는 목표를 1m 앞으로 계속 갱신하므로 이 도달 조건 때문에 1m마다 정지하지 않습니다.

초기 상태는 속도 0입니다. 목표·키보드 heartbeat·현재 위치·점군·유효한 경로·플래너 속도가 있어야 이동 명령을 전달합니다. 키보드/센서가 0.5초 동안 갱신되지 않거나 플래너/UDP 명령이 0.3초 동안 갱신되지 않으면 속도를 0으로 설정합니다. 키보드가 위치 갱신 중단을 감지하면 진행 방향도 해제하므로 복구 후 방향 키를 다시 눌러야 합니다. 시뮬레이션 리셋이나 플래너 재시작 뒤에도 방향 키를 새로 눌러야 합니다.

## 검증 및 범위

정리 후 패키지 빌드와 단위 테스트 44개를 통과했습니다. 학습 정책·관절 매핑, 명령 타임아웃, 감속 및 정지 전환 보호를 검사합니다.

호스트 실로봇 환경의 단위 테스트:

```bash
cd /home/kante/go2_rl
source /opt/ros/jazzy/setup.bash
source install/setup.bash
PYTHONPATH="src/go2_autonomous_driving:${PYTHONPATH:-}" \
  .venv_go2_real/bin/python -m pytest -q -p no:cacheprovider src/go2_autonomous_driving/test
```

전체 연결 확인은 컨테이너에서 터미널 1·2를 실행한 상태로 **키보드 노드 대신** 아래 스크립트를 실행합니다. 스크립트가 전방 목표 하나를 자동 생성하고, 실제 경로·속도·전진 이동·정지 명령을 검사합니다.

```bash
python3 scripts/check_go2_navigation.py
```

이 스크립트는 기본적으로 단일 목표 모드를 검사합니다. 목표 도달 상태까지 검사하려면 `--duration 18 --require-reached`를 추가합니다. 연속 이동은 `--continuous --duration 8`로 검사하며, 한 번의 W 입력으로 첫 1m 목표를 넘어 이동하는지, 목표 갱신 중 속도가 끊기지 않는지, Space로 정지하는지 확인합니다.



화면 없이 실행하려면 `bash scripts/run_go2_simulation.sh --viewer none`을 사용합니다. `--steps 100`은 100 policy step 후 종료합니다. `--fast`는 headless 계산 확인용이며 실시간 플래너 검증에서는 사용하지 않습니다.

현재 위치는 MuJoCo의 정답 위치입니다. 장애물 점군은 JSON에 있는 박스 표면에서 만든 가상 점군이며, 실제 LiDAR의 가림·노이즈·SLAM은 구현하지 않았습니다. 박스는 MuJoCo 물리 환경에도 같은 위치에 생성합니다. 로컬 플래너와 경로 추종기를 연결한 단계이며, 저장소의 SLAM 및 전역 FAR planner 전체를 실행하는 구성은 아닙니다.

체크포인트는 기존 Go2 Flat의 47차원 actor 관측과 12차원 action, 저장된 관측 정규화를 사용합니다. 체크포인트 자체에는 학습 당시 모든 설정이 없으므로, 학습 설정을 변경했다면 해당 관측·action 설정도 맞춰야 합니다. 시뮬레이션 실행기는 실제 로봇 SDK/DDS에 연결하지 않습니다. 실로봇은 위의 별도 실행 절차를 사용합니다.

외부 플래너 확인 버전: `Navigation-Physical-Experiment`, `jazzy`, `9d95a6555bbfbd27b24e9f989c0e67a8268f1500`.
