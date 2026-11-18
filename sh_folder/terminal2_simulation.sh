#!/usr/bin/env bash
# Terminal 2: MuJoCo + trained policy; open http://localhost:8080 on the host.
# --model N (or a .pt path) picks the walking model, e.g. --model 34997 (default: config simulation.json).
# --speed V [m/s] (default 0.5), --lateral-speed V [m/s] (default 0.2) and
# --yaw-rate W [rad/s] (default 0.5) cap the commands;
# use the same values for terminals 1 and 3.
# --native also opens a MuJoCo window that follows the robot, alongside the browser viewer.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$go2_project/scripts/go2_model_args.sh"
go2_parse_model_args "$go2_project" "" "$@"
if [[ -n "$go2_checkpoint" ]]; then
  echo "Walking model: $go2_checkpoint"
  go2_args=(--checkpoint "$go2_checkpoint" "${go2_args[@]}")
fi
exec bash "$go2_project/scripts/run_go2_simulation.sh" --viewer viser "${go2_args[@]}"
