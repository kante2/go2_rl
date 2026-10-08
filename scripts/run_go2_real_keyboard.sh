#!/usr/bin/env bash
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$go2_project/src/go2_autonomous_driving:${PYTHONPATH:-}"
exec "$go2_project/.venv_go2_real/bin/python" -m go2_autonomous_driving.real.keyboard_velocity \
  --config "$go2_project/src/go2_autonomous_driving/config/real_closed_loop.json" "$@"
