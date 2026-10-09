#!/usr/bin/env bash
# Actual LowState closes the learned controller; default is motor-free dry-run.
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$go2_scripts/run_go2_real_docker.sh" -m go2_autonomous_driving.real.real_closed_loop \
  --checkpoint src/go2_autonomous_driving/model_pt/model_10000.pt \
  --policy-dir artifacts/go2_real \
  --fault-log log/go2_real_closed_loop_fault.json \
  --config src/go2_autonomous_driving/config/real_closed_loop.json "$@"
