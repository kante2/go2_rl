#!/usr/bin/env bash
# Used by both the container entrypoint and docker exec; neither starts nodes.
set -e
source /opt/ros/jazzy/setup.bash
source /opt/go2_ros_ws/install/setup.bash
source /opt/go2_unitree_ws/install/local_setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export PYTHONPATH="/workspace/unitree_rl_mjlab/src/go2_autonomous_driving:${PYTHONPATH:-}"
if [[ $# -eq 0 ]]; then set -- bash; fi
exec "$@"
