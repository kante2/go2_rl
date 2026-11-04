#!/usr/bin/env bash
# Terminal 1: ROS planner + bridge inside the Go2 container.
# --speed V [m/s] (default 0.5), --lateral-speed V [m/s] (default 0.2) and
# --yaw-rate W [rad/s] (default 0.5) cap the commands
# sent to MuJoCo; use the same values for terminals 2 and 3.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
go2_launch_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --speed) go2_launch_args+=("speed:=$2"); shift 2 ;;
    --speed=*) go2_launch_args+=("speed:=${1#*=}"); shift ;;
    --lateral-speed) go2_launch_args+=("lateral_speed:=$2"); shift 2 ;;
    --lateral-speed=*) go2_launch_args+=("lateral_speed:=${1#*=}"); shift ;;
    --yaw-rate) go2_launch_args+=("yaw_rate:=$2"); shift 2 ;;
    --yaw-rate=*) go2_launch_args+=("yaw_rate:=${1#*=}"); shift ;;
    *) go2_launch_args+=("$1"); shift ;;
  esac
done
exec bash "$go2_project/scripts/run_go2_docker.sh" \
  ros2 launch go2_autonomous_driving simulation.launch.py "${go2_launch_args[@]}"
