#!/usr/bin/env bash
# Comparison dashboard (v2). Sets the robot DDS settings that run_live.sh leaves to the caller.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$PROJECT_ROOT"

# Inside the go2-rl container: ROS Jazzy and the image's unitree_go messages.
if [[ -f /opt/go2_unitree_ws/install/local_setup.bash && ":${AMENT_PREFIX_PATH:-}:" != *"/opt/go2_unitree_ws/"* ]]; then
  source /opt/ros/jazzy/setup.bash
  source /opt/go2_unitree_ws/install/local_setup.bash
fi

# The container defaults to domain 42; the robot publishes /lowstate on domain 0.
go2_iface="${GO2_NETWORK_INTERFACE:-enp3s0}"
export ROS_DOMAIN_ID="${GO2_ROS_DOMAIN_ID:-0}" ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$go2_iface\"/></Interfaces></General></Domain></CycloneDDS>"

# The container installs requirements.txt into /opt/go2_real; elsewhere use the active python3.
go2_python=python3
[[ -x /opt/go2_real/bin/python ]] && go2_python=/opt/go2_real/bin/python

exec "$go2_python" Prediction/predict_live_v2.py \
  --model contact_force_velocity/models/causal_forecast/all_four.joblib \
  --audio-host unitree@192.168.123.18 \
  --card Microphone \
  --web-port 8080 \
  "$@"
