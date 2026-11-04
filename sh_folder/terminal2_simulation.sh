#!/usr/bin/env bash
# Terminal 2: MuJoCo + trained policy; open http://localhost:8080 on the host.
# Extra args pass through, e.g. --checkpoint src/go2_autonomous_driving/model_pt/model_34997.pt
# --speed V sets the max forward speed [m/s] (default 0.5); use the same V for terminal 1.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$go2_project/scripts/run_go2_simulation.sh" --viewer viser "$@"
