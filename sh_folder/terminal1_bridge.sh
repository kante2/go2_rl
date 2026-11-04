#!/usr/bin/env bash
# Terminal 1: ROS planner + bridge inside the Go2 container.
# --speed V sets the max forward speed [m/s] (default 0.5); use the same V for terminal 2.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
go2_launch_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --speed) go2_launch_args+=("speed:=$2"); shift 2 ;;
    --speed=*) go2_launch_args+=("speed:=${1#*=}"); shift ;;
    *) go2_launch_args+=("$1"); shift ;;
  esac
done
exec bash "$go2_project/scripts/run_go2_docker.sh" \
  ros2 launch go2_autonomous_driving simulation.launch.py "${go2_launch_args[@]}"
