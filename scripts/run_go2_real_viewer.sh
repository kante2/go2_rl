#!/usr/bin/env bash
# Independent receive-only MuJoCo viewer. Uses CPU kinematics, no dynamics/actor.
set -euo pipefail
go2_container="${GO2_MJLAB_CONTAINER:-unitree-rl-mjlab}"
docker start "$go2_container" >/dev/null
docker exec -it \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/unitree_rl_mjlab/src/go2_autonomous_driving:/workspace/unitree_rl_mjlab \
  -w /workspace/unitree_rl_mjlab \
  "$go2_container" python -u -m go2_autonomous_driving.real_state_viewer \
  --model src/assets/robots/unitree_go2/xmls/go2.xml \
  --metadata artifacts/go2_real/policy.json \
  --config src/go2_autonomous_driving/config/real_closed_loop.json "$@"
