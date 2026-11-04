#!/usr/bin/env bash
# Terminal 2: MuJoCo + trained policy; open http://localhost:8080 on the host.
# Extra args pass through, e.g. --checkpoint src/go2_autonomous_driving/model_pt/model_34997.pt
# --speed V [m/s] (default 0.5), --lateral-speed V [m/s] (default 0.2) and
# --yaw-rate W [rad/s] (default 0.5) cap the commands;
# use the same values for terminals 1 and 3.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$go2_project/scripts/run_go2_simulation.sh" --viewer viser "$@"
