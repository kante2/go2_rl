#!/usr/bin/env bash
# Terminal 3: direct velocity keyboard. Start after the model in terminal 2 has loaded.
# W forward, X backward, A/D lateral, S stop, Q/E rotate CCW/CW; each holds until the next key. Ctrl+C quits.
# --speed V [m/s] (W/X, default 0.5), --lateral-speed V [m/s] (A/D, default 0.2) and
# --yaw-rate W [rad/s] (Q/E, default 0.5); terminals 1 and 2 cap these, so use the same values there.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
go2_params=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --speed) go2_params+=(-p "forward_speed:=$2"); shift 2 ;;
    --speed=*) go2_params+=(-p "forward_speed:=${1#*=}"); shift ;;
    --lateral-speed) go2_params+=(-p "lateral_speed:=$2"); shift 2 ;;
    --lateral-speed=*) go2_params+=(-p "lateral_speed:=${1#*=}"); shift ;;
    --yaw-rate) go2_params+=(-p "yaw_rate:=$2"); shift 2 ;;
    --yaw-rate=*) go2_params+=(-p "yaw_rate:=${1#*=}"); shift ;;
    *) echo "Unknown option: $1 (use --speed, --lateral-speed, --yaw-rate)" >&2; exit 2 ;;
  esac
done
go2_ros_args=()
if [[ ${#go2_params[@]} -gt 0 ]]; then go2_ros_args=(--ros-args "${go2_params[@]}"); fi
# Run the module directly: the image's ROS entry-point list predates this node.
exec bash "$go2_project/scripts/run_go2_docker.sh" \
  python3 -m go2_autonomous_driving.sim.keyboard_velocity "${go2_ros_args[@]}"
