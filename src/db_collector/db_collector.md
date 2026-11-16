# RL 보행 중 로봇 state 수집

마이크·소음·SSH 수신 없이 Go2의 ROS 2 상태와 모터 명령을 NPZ로 저장한다.
기존 자동 주행 기능은 유지한다. RL 제어기의 실행과 ARM은 사람이 수행한다.

## 실행

터미널 1에서 RL을 실행하고, 로봇을 지지한 상태에서 ARM을 입력한다.
`REAL POLICY READY`와 정지 자세를 확인한다.

```bash
cd ~/go2_rl
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive real.launch.py \
  model:=baseline mode:=live interface:=enx00e04c362347
```

터미널 2에서 자동 주행·수집을 실행한다. 별도 키보드는 종료한다.

```bash
cd ~/go2_rl
bash src/db_collector/data_collector_RL.sh \
  --interface enx00e04c362347 \
  --model baseline --vx 0.2 \
  --pre 5 --duration 30 --post 5 \
  --out state_vx02_trial01
```

진행 순서:

```text
LowState·LowCmd·RL live 상태 확인
 → 5초 정지 기록
 → 30초 전진 명령
 → 속도 0 요청·RL 적용 명령 0 확인
 → 5초 추가 기록
 → NPZ 저장
```

가감속은 기존 RL 제어기 설정을 따른다. 실제 이동 속도와 물리적 정지는 별도 측정이 필요하다.
준비·감속 구간도 저장되므로 전체 기록은 pre + duration + post보다 길 수 있다.

## 인자

| 인자 | 의미 |
|---|---|
| `--interface` | Go2 연결 LAN 이름, 셸 기본 enx00e04c362347 |
| `--domain` | DDS domain, 기본 0 |
| `--model` | baseline / quietpaw / quietpaw_long, 기본 baseline |
| `--vx`, `--vy`, `--vyaw` | 전후·좌우 m/s, 회전 rad/s, 기본 모두 0 |
| `--pre`, `--post` | 주행 전·속도 0 응답 후 기록, 기본 각 5초 |
| `--duration` | 주행 시간. 현재 셸 기본 15초, 직접 Python 실행 기본 20초 |
| `--out` | 정확한 출력 prefix, 확장자 제외 |
| `--name` | 파일명 + 날짜·시간 자동 추가, 셸 기본 baseline_walk |
| `--control-config` | RL 실행기와 동일한 사용자 JSON (UDP 포트·속도 상한) |
| `--policy-metadata` | policy.json 복사 및 실제 RL 정책 해시 대조 |
| `--passive` | 자동 주행 없이 기록만 수행 |
| `--allow-missing-lowcmd` | passive/dry에서 LowCmd 없는 기록 허용 |

이전 `--ssh-login`, `--host`, `--card`, `--device`, `--audio-pipe`는 제거했다.
수집기는 IP를 자동 변경하지 않는다. LAN에 192.168.123.x 대역 주소가 설정되어 있어야 한다.

정상 종료 후 RL 제어기는 유지된다. 자동 모드에서 q/Q·Ctrl+C·SIGTERM 또는
센서/RL 상태 오류가 발생하면 속도 0을 요청하고 받은 상태까지 저장한다.
갑작스러운 프로세스 중단 시 기존 제어기의 command timeout이 적용된다.
제어 터미널 Ctrl+C 또는 리모컨 L2+B는 damping 종료이며 몸체가 내려갈 수 있다.

기록 전용 모드는 키보드와 함께 사용할 수 있으며 종료 시 정지 명령을 보내지 않는다.

```bash
bash src/db_collector/data_collector_RL.sh --passive --duration 30 --out manual_state01
```

## 저장 위치와 형식

호스트: `/home/hajaemin/go2_rl/src/db_collector/data/state_vx02_trial01.npz`

Docker: `/workspace/unitree_rl_mjlab/src/db_collector/data/state_vx02_trial01.npz`

상대 `--out`은 data 폴더 기준, 절대 경로는 지정 위치를 사용한다.
같은 이름의 NPZ는 새 저장 완료 후 교체한다. 압축 중 실패하면 기존 NPZ를 유지한다. WAV는 생성하지 않는다.
형식 버전은 meta의 format_version=2다.

| 키 | 내용 |
|---|---|
| `ls_t`, `ls_tick` | PC 수신 시각, 로봇 tick |
| `ls_q`, `ls_dq`, `ls_tau` | 실제 관절각·속도·추정 모터 토크 (N,12) |
| `ls_acc`, `ls_gyro`, `ls_rpy`, `ls_quaternion`, `ls_ff` | IMU·자세·발 힘 |
| `lc_t`, `lc_q`, `lc_dq`, `lc_tau`, `lc_kp`, `lc_kd` | 모터 목표·feedforward 토크·PD gain |
| `sp_*` | 선택적인 SportModeState. RL 중 유효성 별도 확인 필요 |
| `rl_t`, `rl_source_t`, `rl_sequence` | RL 상태 수신 시각·원본 시각·순번 |
| `rl_requested`, `rl_command` | RL이 보고한 요청·가감속 적용 속도 (N,3) |
| `rl_ready`, `rl_execute`, `rl_phase`, `rl_session`, `rl_checkpoint` | RL 상태·세션·정책 해시 |
| `tx_t`, `tx_sequence`, `tx_command`, `tx_intent` | 수집기가 전송한 속도 명령 |
| `ev_t`, `ev_label`, `ev_api` | 기록·주행·정지 요청·속도 0 확인 이벤트 |
| `meta` | JSON 문자열, 설정·종료 사유·수신 통계 |

각 스트림의 수신 주기가 달라 행 수는 일치하지 않는다. 시각은 PC monotonic 수신 기준이다.
관절 순서는 SDK 순서(FR, FL, RR, RL; 각 hip, thigh, calf)다.
`lc_tau`는 PD까지 합친 총 토크가 아니다. actor 원시 action은 기록하지 않는다.

읽기:

```python
import json
import numpy as np
with np.load('src/db_collector/data/state_vx02_trial01.npz') as data:
    print(data['ls_q'].shape)
    print(json.loads(data['meta'].item()))
```

저장 실패 시 표시된 임시 폴더를 복구할 수 있다. 마지막 미완성 raw 레코드는 제외한다.
복구 시 원래 실험 이벤트 대신 기록 시작·종료 경계를 재구성한다.

```bash
bash src/db_collector/data_collector_RL.sh --recover <임시폴더> --out recovered_state01
```
