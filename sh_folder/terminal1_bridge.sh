#!/usr/bin/env bash
# Terminal 1: ROS planner + bridge inside the Go2 container.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$go2_project/scripts/run_go2_docker.sh" \
  ros2 launch go2_autonomous_driving simulation.launch.py "$@"
