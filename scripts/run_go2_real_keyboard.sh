#!/usr/bin/env bash
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$go2_scripts/run_go2_docker.sh" go2-real-python -m go2_autonomous_driving.real.keyboard_velocity \
  --config src/go2_autonomous_driving/config/real_closed_loop.json "$@"
