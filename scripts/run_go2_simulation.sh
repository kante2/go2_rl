#!/usr/bin/env bash
# Workspace Docker launcher for the go2_autonomous_driving ROS subpackage.
set -euo pipefail

go2_container="${GO2_MJLAB_CONTAINER:-unitree-rl-mjlab}"
docker start "$go2_container" >/dev/null
docker exec -it \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/unitree_rl_mjlab/src/go2_autonomous_driving:/workspace/unitree_rl_mjlab \
  -w /workspace/unitree_rl_mjlab \
  "$go2_container" python -m go2_autonomous_driving.simulation \
  --checkpoint /workspace/unitree_rl_mjlab/model_10000.pt "$@"
