#!/usr/bin/env bash
# Record simulation states (real-robot NPZ keys) -> <dir>/<name>_<time>_sim.npz; q or Ctrl+C saves.
# Needs terminal 2 started with --state-port 9881. Args: --name NAME, --dir DIR (project-relative), --port.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# File names use the host's local time, not the container's UTC.
go2_tz="$(cat /etc/timezone 2>/dev/null || echo UTC)"
exec bash "$go2_project/scripts/run_go2_docker.sh" env TZ="$go2_tz" \
  /opt/go2_real/bin/python src/db_collector/record_sim.py "$@"
