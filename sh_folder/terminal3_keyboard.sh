#!/usr/bin/env bash
# Terminal 3: keyboard control. Start after the model in terminal 2 has loaded.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$go2_project/scripts/run_go2_docker.sh" \
  ros2 run go2_autonomous_driving keyboard_waypoint \
  --ros-args --params-file src/go2_autonomous_driving/config/navigation.yaml
