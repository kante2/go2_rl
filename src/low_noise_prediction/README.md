# LowState-only robot noise prediction

이 폴더 전체를 복사해서 사용하세요. 원래 `low_noise` 저장소나 학습 데이터는 필요하지 않습니다. 내부 폴더 구조는 유지해야 합니다.

## 준비

- Python 3.10 권장. 모델은 scikit-learn 1.7.2로 저장되었습니다.
- ROS 2와 해당 ROS 배포판으로 빌드한 `unitree_go` 메시지 패키지가 필요합니다. `rclpy`, `std_msgs`는 ROS 환경에서 제공됩니다.
- PC에서 로봇의 `/lowstate`를 수신할 수 있도록 DDS 인터페이스·ROS_DOMAIN_ID를 설정해야 합니다.
- 실제 measured noise를 보려면 로봇에 USB 마이크와 `arecord`가 있어야 하고, 비밀번호 없는 SSH 접속이 필요합니다.

ROS 2와 **본인 PC에서 빌드된 Unitree workspace**를 먼저 source하세요. 아래 경로는 설치 위치에 맞게 바꾸세요.

```bash
source /opt/ros/humble/setup.bash
source /path/to/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

복사한 `low_noise_prediction` 폴더에서 Python 환경을 준비합니다.

```bash
cd /path/to/low_noise_prediction
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`python3`는 source한 ROS와 호환되는 Python을 사용해야 합니다. 예를 들어 Ubuntu 22.04의 ROS Humble은 Python 3.10 환경을 사용합니다.

## 실시간 실행

```bash
bash Prediction/run_live.sh
```

브라우저에서 **http://localhost:8080/** 을 엽니다.

기본 설정:

- Robot state: `/lowstate`만 구독.
- Model: `contact_force_velocity/models/causal_forecast/all_four.joblib`.
- SSH: `unitree@192.168.123.18`.
- ALSA: `plughw:CARD=Microphone,DEV=0`.
- HTML port: `8080`.

주소나 장치가 다르면 실행 옵션으로 변경할 수 있습니다.

```bash
bash Prediction/run_live.sh \
  --audio-host unitree@192.168.123.18 \
  --device 'plughw:CARD=Microphone,DEV=0' \
  --web-port 8080
```

실행 스크립트는 자신의 위치로부터 경로를 계산하므로 어느 디렉터리에서 호출해도 동작합니다. 추가 옵션은 Python 프로그램에 전달합니다. 종료는 터미널에서 `Ctrl+C`입니다.

## 화면에서 확인할 값

- **Measured noise:** 로봇에서 수신한 48 kHz·모노·PCM16의 DC 제거 20 ms RMS dBFS.
- **Predicted noise:** LowState 접촉력·관절·IMU를 이용한 모델 A의 예측 dBFS.
- 과거 46 ms 이력 확보 후 약 50 Hz 추론, 화면은 약 5 Hz 갱신.
- 예측은 t+20 ms 중심의 `[t+10 ms,t+30 ms)` 오디오 RMS를 대상으로 합니다.
- 실측 시각은 SSH 수신 시각과 nominal 오디오 샘플레이트를 이용한 추정치입니다. 두 곡선은 실시간 경향 확인용이며 물리적으로 검증된 시간 정렬이나 RMSE 계산을 제공하지 않습니다.
- 단위는 SPL이 아닌 dBFS입니다. 학습 녹음과 마이크·입력 게인·위치가 달라지면 레벨 차이가 발생합니다.

마이크는 이 프로그램이 직접 사용하므로 다른 `arecord`/녹음 프로그램과 중복 점유하지 마세요. 마이크가 없어도 state 기반 추론은 가능하지만 measured 곡선은 표시되지 않습니다.

## 로봇 없이 HTML 미리보기

```bash
python3 Prediction/web_dashboard.py --demo --port 8080
```

이 모드는 화면에 **SIMULATED DATA**를 표시하며 실제 측정·모델 추론은 하지 않습니다.

## 포함 파일

`Prediction/`은 실행·오디오 수신·HTTP 서버·영어 HTML 화면, `contact_force_velocity/`는 모델과 LowState 기구학 연결 코드, `preprocessing/`은 기구학 계산과 URDF입니다. Unitree URDF 라이선스는 `preprocessing/models/LICENSE.unitree`에 포함했습니다. `SHA256SUMS.json`은 배포 파일 무결성 확인용입니다.
