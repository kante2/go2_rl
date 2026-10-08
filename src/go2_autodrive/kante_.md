# go2_autodrive PACKAGE -----------------------------

# 모든 명령어는 container에서 실행
sh /home/kante/go2_rl/scripts/enter_go2_docker.sh
# 1. simulator
cd /workspace/unitree_rl_mjlab
source /opt/go2_ros_ws/install/setup.bash

ROS_DOMAIN_ID=42 ros2 launch go2_autodrive simulation.launch.py
# 2. mujuco + PT모델 + 웹 ui

cd /workspace/unitree_rl_mjlab/src/go2_autodrive

go2-python -m go2_autodrive.simulation.go2_mujuco_udp_runner \
  --config config/simulation.json \
  --viewer viser \
  --viewer-port 8080

# 3. keyboard

source /opt/go2_ros_ws/install/setup.bash
ROS_DOMAIN_ID=42 ros2 run go2_autodrive keyboard_waypoint