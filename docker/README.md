# Go2 RL: 컨테이너에서 학습·시뮬레이션 실행

시뮬레이션의 플래너·MuJoCo·학습 모델·키보드는 모두 같은 `unitree-rl-mjlab` 컨테이너에서 실행합니다. 호스트에서는 Docker와 NVIDIA 드라이버를 제공하고 브라우저로 화면을 봅니다. 시뮬레이션 실행에 호스트 ROS 설치는 필요하지 않습니다.

실로봇 폐루프 제어기·키보드·화면·기본 모드 복구도 같은 컨테이너에서 실행합니다. 호스트에 Conda, Python 가상환경, ROS를 설치하거나 활성화할 필요가 없습니다.

## 최초 준비: 호스트에서 한 번

```bash
cd /home/kante/go2_rl
bash scripts/setup_go2_docker.sh
```

Dockerfile은 공식 Ubuntu 24.04 / ROS 2 Jazzy 이미지에 플래너를 빌드하고, 고정된 Python 3.11 학습 환경을 포함합니다. `go2-rl:local` 이미지를 생성하고 컨테이너를 시작한 뒤 ROS·GPU·MuJoCo 환경을 검사합니다. 이전 ROS 미포함 이미지에서 업데이트할 때는 `--no-build`를 생략합니다. 실행 중인 학습·시뮬레이션을 끝낸 뒤 환경을 재생성하세요.

이 PC에서는 새 이미지를 빌드·검증한 뒤 제공하므로, 이후에는 다음 명령으로 시작할 수 있습니다.

```bash
bash scripts/setup_go2_docker.sh --no-build
```

필수 조건: Linux x86_64, Docker Engine, Compose 2.30 이상, 호스트 `nvidia-smi` 정상 및 NVIDIA Container Toolkit. 체크포인트는 `src/go2_autonomous_driving/model_pt/model_10000.pt`에 둡니다. 모델은 프로젝트 공유 마운트로 제공하며 이미지에 복사하지 않습니다. CUDA 라이브러리는 이미지에 포함됩니다. [Docker 설치](https://docs.docker.com/engine/install/), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## 컨테이너 접속

호스트에서 터미널을 3개 열고 **각 터미널에서** 다음 명령을 실행합니다.

```bash
sh /home/kante/go2_rl/scripts/enter_go2_docker.sh
```

세 터미널은 같은 컨테이너에 접속합니다. 접속 위치는 `/workspace/unitree_rl_mjlab`이며 ROS 환경과 프로젝트 패키지는 자동으로 설정됩니다. `source install/setup.bash`를 따로 실행하지 않습니다. 이 경로의 `install/`은 호스트 빌드이므로 컨테이너에서 사용하지 않습니다.

컨테이너 셸을 종료하려면 실행 중인 명령을 Ctrl+C로 끝낸 뒤 `exit`를 입력합니다. 다른 터미널과 컨테이너는 계속 실행됩니다.

## 시뮬레이션: 접속한 터미널 안에서 실행

터미널 1 — ROS 플래너·브리지:

```bash
ros2 launch go2_autonomous_driving simulation.launch.py
```

터미널 2 — MuJoCo + 기존 학습 모델:

```bash
bash scripts/run_go2_simulation.sh --viewer viser
```

호스트 브라우저에서 <http://localhost:8080>을 엽니다. 모델 로딩이 끝나면 키보드를 실행합니다.

터미널 3 — 키보드:

```bash
ros2 run go2_autonomous_driving keyboard_waypoint \
  --ros-args --params-file src/go2_autonomous_driving/config/navigation.yaml
```

W/↑는 연속 전진, S/↓는 후방 방향 이동, A/D는 좌우 방향 이동, Space는 정지, Q는 키보드 종료입니다. 세 프로세스의 `ROS_DOMAIN_ID` 기본값은 42입니다. 이 구성은 실로봇에 모터 명령을 보내지 않습니다.

## 학습: 접속한 컨테이너 안에서 실행

```bash
bash scripts/run_go2_training.sh Unitree-Go2-Flat
```

짧은 동작 확인:

```bash
bash scripts/run_go2_training.sh Unitree-Go2-Flat \
  --env.scene.num-envs 16 --agent.max-iterations 2 \
  --agent.num-steps-per-env 8 --agent.save-interval 1 \
  --agent.logger tensorboard --agent.run-name docker_smoke
```

학습·시뮬레이션 스크립트는 호스트에서 실행하면 컨테이너로 전달하고, 컨테이너 안에서 실행하면 그대로 프로그램을 시작합니다. 컨테이너 안에서 Docker를 다시 실행하거나 Docker 소켓을 공유하지 않습니다.

## 파일과 Python 환경

| 항목 | 경로/역할 |
| --- | --- |
| 호스트 프로젝트 | `/home/kante/go2_rl` |
| 컨테이너 프로젝트 | `/workspace/unitree_rl_mjlab` (호스트 폴더 공유) |
| ROS 설치 | `/opt/ros/jazzy` |
| 컨테이너의 ROS 빌드 | `/opt/go2_ros_ws/install` (호스트 `install/`과 분리) |
| ROS Python | `python3` → 시스템 Python 3.12 |
| 실로봇 Python | `go2-real-python` → 이미지 내부 Python 3.12 + ROS/Unitree/ONNX |
| 학습 Python | `python` 또는 `go2-python` → Python 3.11 |
| 학습 로그·모델 | 호스트 `logs/rsl_rl/go2_velocity/`에 저장 |

학습용 Python 래퍼는 ROS Python 확장 모듈 경로를 제거한 뒤 실행합니다. PyTorch/mjlab의 139개 Python 패키지 버전은 `requirements.lock`으로 유지합니다. 플래너는 외부 저장소의 고정 커밋 `9d95a6555bbfbd27b24e9f989c0e67a8268f1500`에서 가져옵니다. OS/ROS apt 패키지는 빌드 당시 저장소에서 설치합니다. ROS 바이너리는 빌드에 사용된 Python과 일치해야 합니다: [ROS 공식 Python 안내](https://github.com/ros2/ros2_documentation/blob/jazzy/source/How-To-Guides/Using-Python-Packages.rst).

프로젝트 소스·설정 수정은 공유 마운트로 반영됩니다. Python 의존성, C++ 플래너, ROS 패키지 메타데이터/실행 항목을 바꾸면 이미지를 다시 빌드합니다. 이미지에는 체크포인트·호스트 가상환경·호스트 ROS 빌드 결과를 복사하지 않습니다.

## 실로봇: Docker에서 실행

최초 설치 또는 이전 이미지에서 전환할 때 `bash scripts/setup_go2_docker.sh`로 이미지를 빌드합니다. Unitree 메시지·CycloneDDS·ONNX Runtime은 이미지에 포함됩니다. `bash scripts/prepare_go2_real_robot.sh`로 모델을 변환하고 정책 일치 검사를 수행합니다. 기존 ONNX·매핑 파일도 프로젝트 공유 폴더로 계속 사용합니다.

아래 명령은 호스트에서 실행하면 Docker로 전달됩니다. 컨테이너 안에서는 프로젝트 루트에서 `bash scripts/...` 경로로 실행합니다. 네트워크는 host 모드를 사용하며 실로봇 DDS는 domain 0, 기본 인터페이스는 `enp3s0`입니다. `GO2_NETWORK_INTERFACE=인터페이스명`으로 변경할 수 있습니다. 시뮬레이션의 기본 domain 42는 유지됩니다.

터미널 1 — 실제 자세 화면 (스크립트가 Docker에서 실행):

```bash
bash /home/kante/go2_rl/scripts/run_go2_real_viewer.sh
```

터미널 2 — 실제 피드백 제어 (Docker):

```bash
bash /home/kante/go2_rl/scripts/run_go2_real_closed_loop.sh --execute
```

터미널 3 — 직접 키보드 (Docker):

```bash
bash /home/kante/go2_rl/scripts/run_go2_real_keyboard.sh --speed 0.5
```

`ARM` 입력 후 `REAL POLICY READY`가 표시되면 방향 키를 사용합니다. Space는 감속 정지, 터미널 2의 Ctrl+C는 damping 후 제어 종료입니다. 뷰어 종료는 모터를 멈추지 않습니다. 실제 자세 화면의 몸체 위치는 고정이며, 실제 이동 거리를 표시하지 않습니다. 자세 전환·비상 정지·복구 절차는 [패키지 README](../src/go2_autonomous_driving/README.md)에 있습니다.

## 점검과 CPU 뷰어

컨테이너에서 환경 확인:

```bash
python scripts/check_go2_docker.py --require-cuda
```

GPU 없이 실제 자세 뷰어만 사용하는 경우에는 호스트에서 CPU 설정으로 시작합니다.

```bash
bash scripts/setup_go2_docker.sh --cpu --no-build
```

GPU 모드로 돌아오려면 `bash scripts/setup_go2_docker.sh --no-build`를 실행합니다. GPU 없는 CPU 설정으로 학습·물리 시뮬레이션을 검증하지는 않습니다.

- 호스트 `nvidia-smi`가 실패하면 먼저 호스트 GPU 상태를 복구합니다.
- 포트 8080을 다른 뷰어가 사용 중이면 이전 뷰어를 끝내거나 실제 상태 뷰어에 `--port 8081`을 사용합니다.
- host network로 UDP 9871/9872(시뮬레이션), 9901/9902/9903(실로봇 화면·키보드)을 공유합니다. `ports:` 설정은 사용하지 않습니다.
- 호스트 ROS의 `install/setup.bash`와 호스트 가상환경은 사용하지 않습니다. 기존 터미널에 `.venv_go2_real` 표시가 남아 있으면 `deactivate`로 나올 수 있습니다.

## 검증 기록 (2026-10-08)

- ROS 2 Jazzy의 플래너·키보드 패키지를 이미지 안에서 빌드했습니다. 접속 스크립트로 들어간 대화형 셸에서 ROS 패키지 조회, Python 분리, 실제 키보드 CLI 실행 및 Q 종료를 확인했습니다.
- 플래너·MuJoCo·키보드 검증 코드를 모두 같은 컨테이너 안에서 실행했습니다. 기존 `model_10000.pt`로 1,500 policy step, 연속 W 입력으로 약 4.92m 전진, Space 정지, HTTP 200을 확인했습니다.
- ROS 환경을 불러온 상태에서도 GPU 학습 16개 환경 / PPO 2 iteration과 모델 저장이 통과했습니다. 학습용 Python 패키지 139개의 고정 버전이 일치합니다.
- 별도 CPU 컨테이너에서 ROS·MuJoCo 환경 검사, 실제 자세 뷰어 HTTP 200, 호스트의 가상 측정 자세 138프레임 수신을 확인했습니다.
- Dockerfile 빌드 검사와 셸 구문 검사를 통과했습니다. 다운로드 캐시는 `PIP_NO_CACHE_DIR`를 해제해 사용하며, 캐시 재사용을 별도 다운로드 검사로 확인했습니다.

로그는 호스트 `log/docker_ros/`에 있습니다. 실제 로봇 모터를 구동하는 시험은 수행하지 않았습니다. 교체 전 이미지는 `/home/kante/SSD/go2_rl_docker_backup/20261008_before_ros/`에 보관했습니다.

### 실로봇 Docker 통합 검증 (2026-10-08)

- 이 PC는 기존 학습·ROS 이미지를 기반으로 위 Dockerfile의 실로봇 의존성 단계를 추가 빌드했습니다. 전체 이미지를 처음부터 재빌드하지는 않았습니다. 전환 전 이미지는 `go2-rl:before-real-migration` 태그로 보관합니다.
- 네트워크를 차단한 검증 컨테이너에서 실로봇 Python 3.12의 ROS·Unitree 메시지 타입·CycloneDDS·ONNX Runtime을 확인했습니다. 단위 테스트 44개가 모두 통과했습니다.
- 기존 100개 프레임의 ONNX/PT 최대 오차는 `8.35e-7`, 실제 관측 매핑 최대 오차는 `1.67e-6`으로 허용치 `1e-4` 이내입니다.
- 가상 DDS domain 94 / loopback에서 정상 종료, SIGINT, 비상 정지, 센서 중단, 키보드 타임아웃, 관절 추종 오류 6가지 검사를 통과했습니다. 실제 로봇 모터는 구동하지 않았습니다.
- 호스트와 컨테이너 안의 실행 경로, 인터페이스 전달, 키보드 대화형 Q 종료, 시뮬레이션 domain 42 유지, GPU·EGL 검사를 통과했습니다. 로그는 `log/docker_real/`에 있습니다.
