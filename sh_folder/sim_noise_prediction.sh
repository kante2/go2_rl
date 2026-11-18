#!/usr/bin/env bash
# Simulation noise prediction (model A) -> http://localhost:8081.
# Needs terminal 2 started with --state-port 9881. Extra args: --state-port, --web-port.
# Cannot run together with sim_record.sh (same UDP port).
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$go2_project/scripts/run_go2_docker.sh" \
  bash -c 'cd src/db_prediction/realtime_prediction && exec /opt/go2_real/bin/python predict_sim.py "$@"' _ "$@"
