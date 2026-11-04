#!/usr/bin/env bash
# Terminal 2: MuJoCo + trained policy; open http://localhost:8080 on the host.
# Extra args pass through, e.g. --checkpoint logs/rsl_rl/go2_velocity/resume_model_10000/model_10000.pt
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$go2_project/scripts/run_go2_simulation.sh" --viewer viser "$@"
