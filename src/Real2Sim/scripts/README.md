# Baseline 시뮬레이션 데이터 수집

동일 baseline 체크포인트와 속도 명령으로 실제 Go2와 비교할 데이터를 수집한다.
기존 Go2 Docker 컨테이너의 mjlab `Unitree-Go2-Flat`을 사용하며 ROS/UDP/로봇 연결은 없다.

## 실행

저장소 루트에서:

```bash
bash src/Real2Sim/scripts/run_collect_sim_baseline.sh
```

기본 reference는 `src/db_collector/data/0.2_1.npz`, 출력은
`src/Real2Sim/simulator/baseline/0.2_1.npz`다. 창 없이 가능한 속도로 실행한다.
기존 출력은 덮어쓰지 않는다. 재수집할 때만 `--overwrite`를 지정한다.

* 기본 PT: `src/go2_autodrive/config/model_pt/baseline/model_34997.pt`.
  실제 기록의 SHA256와 일치해야 실행한다. 다른 PT는 `--checkpoint`로 지정한다.
* 입력: reference의 **가감속 적용 후 `rl_command`**를 수신 시각 기준 zero-order hold로 재생한다.
  최초 수신 전에는 0이며 가감속을 다시 적용하지 않는다.
* 기록 길이: 실제 `record_start`부터 `record_stop`까지. 20 ms 제어 주기 단위로 올림한다.
* 제어: 정규화된 PT actor, 실기 `PolicyMapping`과 `JointTargetRamp`를 사용한다.
  이전 action 관측에는 목표각 변화 제한 후 action을 넣는다.
* 환경: 명목 평지 모델, 기본 자세·0 속도·yaw=0. 마찰/무게중심/encoder bias 무작위화와
  외란을 제거한다. 신발·바닥별 실험을 추가하지 않는다.
* 환경 종료/예외/중단 시 이미 수집한 부분은 `meta.completed=false` 및 실패 사유와 함께
  저장하고 프로세스도 실패로 종료한다. 초기화 전에 실패하면 저장하지 않는다.

다른 시행과 속도:

```bash
bash src/Real2Sim/scripts/run_collect_sim_baseline.sh \
  --reference src/db_collector/data/0.2_2.npz
bash src/Real2Sim/scripts/run_collect_sim_baseline.sh \
  --reference src/db_collector/data/0.4_1.npz --vx 0.4
```

reference 없이 정지·전진·감속·정지 명령을 생성할 수도 있다.
`--pre/--duration/--post/--trial`은 이 모드에서 사용한다.

```bash
bash src/Real2Sim/scripts/run_collect_sim_baseline.sh \
  --no-reference --vx 0.2 --trial 1 --pre 5 --duration 15 --post 5
```

CLI 경로는 컨테이너 안에서 해석된다. 저장소 상대 경로 사용을 권장한다.
Docker 안에서는 `python src/Real2Sim/scripts/collect_sim_baseline.py`로 직접 실행한다.

## NPZ 형식

관절 순서는 SDK **FR, FL, RR, RL; 각 hip, thigh, calf**다.
상태는 물리 200 Hz, 정책·명령은 50 Hz다. 실제 500 Hz로 인위적으로 업샘플링하지 않는다.
모든 시간은 기록 시작 기준 시뮬레이션 초다. 상태는 t=0과 이후 적분 후 상태를 포함하며,
명령 시각은 해당 명령이 적용되기 시작한 시각이다.

| 키 | 의미 |
|---|---|
| `ls_t`, `ls_tick` | 시뮬레이션 시간 [s], [ms] |
| `ls_q`, `ls_dq`, `ls_tau` | 관절각 [rad], 속도 [rad/s], actuator 토크 [Nm] |
| `ls_quaternion`, `ls_rpy` | 몸체 자세; quaternion wxyz |
| `ls_gyro`, `ls_acc` | 모델 IMU 각속도·가속도 센서값 |
| `sim_base_pos_world`, `sim_base_vel_body`, `sim_base_vel_world` | 몸체 위치·선속도 정답 |
| `sim_foot_force_world_N`, `sim_foot_fz_N` | 로봇에 작용하는 발별 월드 GRF [N]; 위쪽 Fz 양수 |
| `rl_t`, `rl_command` | 정책에 입력한 속도 명령과 시각 |
| `policy_observation` | 정규화 전 47차원 정책 관측 |
| `policy_action_raw`, `policy_action_applied` | actor 출력, 목표각 변화 제한 후 action; 정책 관절 순서 |
| `lc_t`, `lc_q`, `lc_dq`, `lc_tau`, `lc_kp`, `lc_kd` | SDK 순서 모터 목표와 PD 설정; feedforward 토크는 0 |
| `ev_t`, `ev_label` | reference 또는 생성한 상대 이벤트 시간 |
| `meta` | 체크포인트/reference 해시, 완료 여부, 설정, 단위·시간·좌표계와 한계 |

실제 발 raw counts와 N 단위 힘을 혼동하지 않도록 **`ls_ff` 키는 만들지 않는다.**
실제 `ls_tau`는 추정 모터 토크이므로 시뮬레이션 토크와 동일한 정답은 아니다.

## 해석 범위

동일 정책·명령에서 보행을 비교하는 수집이다. 실제 초기 상태·보행 위상이나
watchdog/fault 상태 머신은 복원하지 않는다. 실제 기록에는 독립적인 몸체 위치·선속도가
없으므로 **0.2 m/s 명령을 실제 속도로 취급하지 않는다.** 명령의 수신 시각과 실제 적용 시각
차이도 남는다. 초기 상태와 물리 파라미터를 고정했으므로 반복 결과는 사실상 같을 수 있다.
seed만 바꿔 실제 시행 변동성을 재현했다고 해석하면 안 된다.

다음 단계는 실제·시뮬레이션 관절 진폭·주기·자세·접촉 패턴 비교다.

## 코드 정리와 검증

현재 실행 파일은 `collect_sim_baseline.py`, `run_collect_sim_baseline.sh`다.
후속 보정에 필요한 기존 재생·식별·평가 코드와 테스트는
[offline_identification/](offline_identification/README.md)로 분리했다.
이전 단일 기록용 `run_pipeline.py`는 제거했다. 원본 데이터와 `outputs/`는 유지했다.

수집기 단위 검증(호스트 NumPy/pytest):

```bash
python3 -m pytest -q src/Real2Sim/scripts/tests
```
