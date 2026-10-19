# Go2 보행 모니터

MuJoCo의 발별 지면 접촉력, 착지 충격량과 실제/목표 속도를 보는 로컬 웹 UI.
`baseline`, `quietpaw`, `quietpaw_long`을 자동 식별한다. 모델 선택은 기존 MuJoCo launch에서 한다.
실로봇 센서/마이크를 읽는 기능은 포함하지 않는다.

## 빌드

저장소 루트에서 실행한다. 최초 설치와 패키지 구성 변경 시 필요하다.

```bash
cd /home/kante/go2_rl
bash scripts/run_go2_docker.sh bash -c 'cd /opt/go2_ros_ws && colcon build --base-paths /workspace/unitree_rl_mjlab/src/go2_autodrive /workspace/unitree_rl_mjlab/src/go2_gait_monitor --packages-select go2_autodrive go2_gait_monitor --symlink-install'
```

## 실행

각 명령은 별도 터미널에서 저장소 루트를 작업 폴더로 실행한다.

```bash
# 1. MuJoCo (model 값으로 보행 모델 선택)
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive mujoco.launch.py model:=quietpaw_long

# 2. 기존 키보드 웨이포인트 제어
bash scripts/run_go2_docker.sh ros2 run go2_autodrive keyboard_waypoint

# 3. 보행 UI
bash scripts/run_go2_docker.sh ros2 launch go2_gait_monitor monitor.launch.py
```

브라우저: **http://localhost:8081**. MuJoCo 뷰어는 기존 http://localhost:8080.
콘솔 실행도 가능하다: `ros2 run go2_gait_monitor dashboard`.
이미 컨테이너 셸 안이면 앞의 `bash scripts/run_go2_docker.sh`를 생략해도 된다.
측정 코드가 추가되기 전에 실행한 시뮬레이터는 한 번 재실행해야 한다.
UI는 시뮬보다 먼저/나중에 시작할 수 있으며 UI 종료는 제어에 영향을 주지 않는다.

## 표시 항목과 정의

| 항목 | 계산/단위 |
|---|---|
| 종방향/횡방향 속도 | 몸체 좌표계 실제 vx/vy 및 적용 목표 / m/s |
| 각속도 | 몸체 좌표계 실제/목표 wz / rad/s |
| 현재 접촉력 | 각 발과 `terrain` 사이 합력의 월드 수직 성분 절댓값 `abs(Fz)` / N |
| 힘 그래프 | 매 물리 스텝에서 읽은 힘의 송신 구간 최대값 / N |
| 착지 충격량 | 착지 이후 최대 50ms 동안 `sum(abs(Fz) * physics_dt)` / N·s |
| 착지 피크 | 위 적분 구간에서 측정한 최대 수직 힘 / N |
| 접지 | `abs(Fz) > 1 N` |
| 착지 횟수 | 20ms 이상 공중 상태 뒤 접촉한 이벤트의 적분 완료 횟수 |
| 실시간 배율 | 수신 구간의 시뮬 경과 시간 / 실제 경과 시간 |

발 순서는 FR(오른앞), FL(왼앞), RR(오른뒤), RL(왼뒤).
발이 50ms보다 일찍 떨어지면 그때까지의 충격량으로 완료한다.
아직 완료된 착지가 없으면 `—`로 표시하며, 마지막 착지값 옆에 완료 후 경과 시간을 표시한다.
초기 지면 위에 떠 있던 로봇이 떨어져 닿는 이벤트도 이 조건을 만족하면 착지로 집계한다.
서 있는 동안의 하중을 매 프레임 새 착지로 세지 않는다.

충격량은 평지 월드 수직축 기준이며 중력에 의한 지지력도 포함한다.
센서는 terrain 접촉만 측정하므로 박스 장애물과의 접촉은 포함하지 않는다.
이 물리량은 **실제 소음(dB)이 아니며**, 기존 학습의 noise_cost와도 별도 지표이다.
모델 간 비교는 바닥·명령 속도·quietness·측정 구간을 맞추고 시작 직후 낙하 구간을 제외한다.

## 기록과 연결 상태

- 물리 샘플링은 환경의 `physics_dt`마다 수행한다. 현재 세 모델은 0.005초, 200Hz.
- UI UDP 송신은 최대 20Hz. 송신 간 모든 물리 샘플을 적분하므로 짧은 힘 피크도 반영한다.
- 브라우저는 약 10Hz로 갱신하고 최근 10/30/60초 그래프를 선택할 수 있다.
- 1초 이상 수신이 없으면 `수신 중단`과 마지막 측정값을 표시한다.
- 화면 일시정지는 표시만 멈춘다. 측정과 CSV 버퍼 기록은 계속한다.
- CSV 저장은 현재 시뮬 세션의 최근 12,000개 수신 샘플(최대 송신률 기준 약 10분)을 내려받는다.
  각 행은 송신 샘플이며 원시 200Hz 전체 기록은 아니다. 착지값은 다음 착지까지 반복되므로
  이벤트별 분석 시 `landing_count` 또는 `landing_time_s`가 바뀌는 행을 사용한다.
- 모델 교체, 시뮬 재실행, 에피소드 리셋은 별도 세션으로 처리해 그래프·CSV 버퍼를 초기화한다.
  모델을 바꾸기 전에 CSV를 저장한다.

## 통신과 코드

제어/내비게이션 UDP와 별개인 **9910** 포트로 측정만 보낸다.
웹 서버/UDP 수신기는 로컬 호스트에 바인딩한다. Docker는 기존 host network를 사용한다.
웹 화면은 외부 CDN·인터넷·추가 웹 프레임워크 없이 작동한다.

```bash
# 포트 변경 시 송신기와 수신기를 함께 변경
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive mujoco.launch.py model:=quietpaw monitor_port:=19910
bash scripts/run_go2_docker.sh ros2 launch go2_gait_monitor monitor.launch.py udp_port:=19910 http_port:=8082

# 측정 오버헤드 없이 기존 시뮬만 실행
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive mujoco.launch.py model:=quietpaw monitor_port:=0
```

- `go2_gait_monitor/mujoco.py`: 명시적 발별 접촉 센서와 물리 샘플 수집, 독립 UDP 송신.
- `go2_gait_monitor/metrics.py`: 착지 판정과 충격량 적분. Torch/ROS 의존성 없음.
- `go2_gait_monitor/server.py`: 패킷 검증, 세션/기록 관리, HTTP/CSV.
- `go2_gait_monitor/web/`: 화면, 그래프, 연결 상태.

한 번에 하나의 시뮬레이터를 연결한다. 중복 패킷은 버리고 누락 수를 표시한다.
추가 접촉 센서와 CPU로의 샘플 복사가 계산 비용을 더할 수 있으며 실시간 배율로 확인할 수 있다.
