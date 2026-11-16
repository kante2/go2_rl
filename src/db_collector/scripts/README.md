# Go2 재연결 시 인터페이스·IP 복구 및 state 수집

모든 명령은 **호스트 PC 터미널**에서 실행한다. 네트워크 설정을 변경하기 전에
RL 제어기와 수집기를 종료하고 로봇을 지지한다.

## 1. 사용하는 주소

| 항목 | 현재 PC/로봇 기준 |
|---|---|
| Go2 연결 유선 인터페이스 | `enx00e04c362347` |
| PC 유선 IPv4 | `192.168.123.99/24` |
| Go2 확인용 IPv4 | `192.168.123.161` |
| USB LAN 어댑터 MAC | `00:E0:4C:36:23:47` |

`interface`는 **PC의 LAN 장치 이름**이다. PC 주소나 로봇 주소가 아니다.
PC나 USB LAN 어댑터를 바꾸면 인터페이스 이름을 다시 확인해야 한다.
같은 MAC 기반 USB 어댑터의 이름은 보통 유지되지만 분리 중에는 목록에서 사라진다.

```bash
ip -br link
ip -br addr
```

`enx00e04c362347` 자체가 없으면 USB LAN 어댑터 인식부터 확인한다.
IP 명령으로 없는 인터페이스를 생성할 수는 없다.

```bash
lsusb
sudo dmesg -w
# USB LAN 어댑터를 다시 연결해 인식 로그 확인 후 Ctrl+C
```

## 2. 로봇 재부팅·USB 재연결 후 빠른 복구

이 PC에서는 이전에 활성 `contest` 프로필에 `192.168.1.100/24`가 저장되어 있었다.
`ip addr add/del/replace`는 실행 중 설정만 바꾸므로 링크 재연결·PC 재부팅 시
NetworkManager가 저장한 주소를 다시 적용할 수 있다. 로봇 부팅 직후 센서 발행이
아직 준비되지 않은 경우에도 `LowState missing/stale`이 발생할 수 있다.

먼저 현재 인터페이스 이름을 확인한 뒤 다음을 실행한다.

```bash
cd /home/hajaemin/go2_rl

GO2_IFACE=enx00e04c362347
GO2_PC_ADDR=192.168.123.99/24
GO2_ROBOT_IP=192.168.123.161

ip -br addr show "$GO2_IFACE"
sudo ip link set dev "$GO2_IFACE" up

# 이 Go2 전용 LAN에 예전 주소가 남아 있으면 해당 주소만 제거
if ip -o -4 addr show dev "$GO2_IFACE" | rg -q 'inet 192\.168\.1\.100/24'; then
  sudo ip addr del 192.168.1.100/24 dev "$GO2_IFACE"
fi

# PC에 Go2 통신용 주소 적용
sudo ip addr replace "$GO2_PC_ADDR" dev "$GO2_IFACE"

ip -br addr show "$GO2_IFACE"
ping -I "$GO2_IFACE" -c 4 "$GO2_ROBOT_IP"
```

정상 예시:

```text
enx00e04c362347  UP  192.168.123.99/24 ...
64 bytes from 192.168.123.161: ...
```

`192.168.123.99`는 PC 주소다. 로봇에 동일한 주소를 설정하지 않는다.
이 절차는 임시 복구이므로 재연결 후 주소가 돌아오면 다시 실행한다.
다른 `192.168.x.x` 주소가 함께 있다면 해당 LAN의 기존 용도를 확인한다.
다른 용도의 주소를 일괄 삭제하는 `ip addr flush`는 사용하지 않는다.

`interface:=...` 또는 `--interface ...`는 DDS가 쓸 LAN을 선택하는 인자이며,
PC의 IPv4 주소를 설정해 주지는 않는다.

## 3. 모터 출력 없이 연결 검증

Ping이 성공한 뒤 로봇 부팅이 완료되었는지 확인하고 실행한다.

```bash
cd /home/hajaemin/go2_rl
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive real.launch.py \
  model:=baseline mode:=dry interface:=enx00e04c362347 seconds:=10
```

`DRY READY`, 증가하는 states/inferences, 최근 state_age를 확인한다.
Ping은 되는데 `LowState missing/stale`이면 IP만으로 원인을 단정하지 않는다.
DDS 센서 수신을 별도로 확인한다.

```bash
GO2_NETWORK_INTERFACE=enx00e04c362347 \
  bash scripts/run_go2_real_docker.sh -m ros2cli topic hz /lowstate
# 확인 후 Ctrl+C
```

센서가 오지 않으면 로봇 부팅 상태, DDS domain 0, 인터페이스 지정, 케이블을 확인한다.
`ROS_LOCALHOST_ONLY is deprecated` 경고만으로 통신 실패를 판단하지 않는다.

## 4. 터미널 두 개로 자동 보행·state 수집

### 터미널 1: RL 제어기

```bash
cd /home/hajaemin/go2_rl
bash scripts/run_go2_docker.sh ros2 launch go2_autodrive real.launch.py \
  model:=baseline mode:=live interface:=enx00e04c362347
```

로봇을 지지한 상태에서 `ARM`을 입력하고 `REAL POLICY READY`와 정지 자세를 확인한다.

### 터미널 2: 자동 주행·기록

```bash
cd /home/hajaemin/go2_rl
bash src/db_collector/data_collector_RL.sh \
  --interface enx00e04c362347 \
  --model baseline --vx 0.2 \
  --pre 5 --duration 30 --post 5 \
  --out state_0.2_1
```

```text
state·RL 준비 확인 → 5초 정지 기록 → 30초 전진 요청
→ 감속·속도 0 응답 확인 → 5초 추가 기록 → NPZ 저장
```

- **별도 키보드는 실행하지 않는다.** 자동 수집기와 UDP 9903 상태 포트가 충돌한다.
- `--duration` 생략 시 현재 셸 기본 주행 시간은 **15초**다.
- 속도 상한은 선택 모델의 JSON과 실제 RL 제어기 설정을 따른다.
- q/Q 또는 Ctrl+C로 자동 수집을 종료하면 속도 0을 요청하고 저장한다.
- 정상 완료 후에도 RL 제어기는 유지된다. 속도 0 응답은 물리적 정지 측정과 다르다.
- 현재 수집기는 **로봇 state·모터 명령만** 기록한다. 마이크·SSH 인자는 필요 없다.

출력:

```text
/home/hajaemin/go2_rl/src/db_collector/data/state_0.2_1.npz
```

같은 `--out`이면 새 NPZ 저장 완료 후 기존 파일을 교체한다.
형식·필드 설명은 [db_collector.md](../db_collector.md)를 참고한다.

## 5. RL 종료 후 Unitree 기본 제어로 복구

수집기와 RL 제어기를 먼저 종료하고 몸체를 지지한다.

```bash
cd /home/hajaemin/go2_rl
GO2_NETWORK_INTERFACE=enx00e04c362347 \
  bash scripts/recover_go2_normal.sh --restore
```

확인 질문에 `NORMAL`을 입력한다. Docker 실행 권한 때문에 sudo를 사용하는 환경은 다음과 같다.

```bash
sudo env GO2_NETWORK_INTERFACE=enx00e04c362347 \
  bash scripts/recover_go2_normal.sh --restore
```

현재 복구 셸은 `--interface` 대신 `GO2_NETWORK_INTERFACE` 환경변수로 인터페이스를 지정한다.
지정하지 않으면 기본 `enp3s0`을 사용해 이 PC에서 인터페이스 오류가 날 수 있다.

## 6. 선택: 고정 IP 영구 저장

매번 임시 복구하지 않으려면 NetworkManager에 Go2 전용 프로필을 저장한다.
기존 `contest` 프로필은 유지하고 해당 USB 어댑터에서는 `go2`를 우선 선택하게 한다.
로봇 제어·수집을 종료한 상태에서만 적용한다.

```bash
GO2_IFACE=enx00e04c362347
GO2_MAC=00:E0:4C:36:23:47

# 이미 go2 프로필이 있으면 중복 생성하지 않음
if ! nmcli -t -f NAME connection show | rg -qx 'go2'; then
  sudo nmcli connection add type ethernet con-name go2 ifname "$GO2_IFACE"
fi

sudo nmcli connection modify go2 \
  connection.interface-name "$GO2_IFACE" \
  802-3-ethernet.mac-address "$GO2_MAC" \
  ipv4.method manual \
  ipv4.addresses 192.168.123.99/24 \
  ipv4.gateway "" \
  ipv4.never-default yes \
  ipv6.method disabled \
  connection.autoconnect yes \
  connection.autoconnect-priority 100

sudo nmcli connection up go2
nmcli -f GENERAL.CONNECTION,IP4.ADDRESS device show "$GO2_IFACE"
```

USB 재연결 후 연결 이름이 `go2`, 주소가 `192.168.123.99/24`인지 확인한다.
이 설정은 **현재 PC에만 저장**된다. 다른 PC에서는 인터페이스와 고정 IP를 다시 설정한다.
Go2 전용 LAN에는 인터넷 게이트웨이를 넣지 않아 인터넷 기본 경로는 Wi-Fi를 유지한다.
