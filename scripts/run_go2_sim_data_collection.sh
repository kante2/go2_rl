#!/usr/bin/env bash
# MuJoCo walking-data collection with keyboard velocity; CSV -> src/sim_to_real_data/simulator.
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Trial file names use the host's local time, not the container's UTC.
go2_tz="$(cat /etc/timezone 2>/dev/null || echo UTC)"
exec bash "$go2_scripts/run_go2_docker.sh" env TZ="$go2_tz" \
  python -m go2_autonomous_driving.sim.data_collection "$@"
