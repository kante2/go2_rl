#!/usr/bin/env bash
# Independent receive-only MuJoCo viewer. Uses CPU kinematics, no dynamics/actor.
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$go2_scripts/run_go2_docker.sh" python -u -m go2_autonomous_driving.real.real_state_viewer \
  --model src/assets/robots/unitree_go2/xmls/go2.xml \
  --metadata artifacts/go2_real/policy.json \
  --config src/go2_autonomous_driving/config/real_closed_loop.json "$@"
