#!/usr/bin/env bash
# Actual LowState closes the learned controller; default is motor-free dry-run.
# --model N (or a .pt path) picks the walking model; its converted policy must exist in
# artifacts/go2_real/baseline/<sha256> (create it with prepare_go2_real_robot.sh --model N).
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
go2_project="$(cd -- "$go2_scripts/.." && pwd)"
source "$go2_scripts/go2_model_args.sh"
go2_parse_model_args "$go2_project" 34997 "$@"
go2_policy="$(go2_policy_dir "$go2_project" "$go2_checkpoint")"
if [[ ! -f "$go2_project/$go2_policy/policy.json" ]]; then
  echo "No converted real-robot policy for $go2_checkpoint." >&2
  echo "Run first: bash scripts/prepare_go2_real_robot.sh --model $go2_checkpoint" >&2
  exit 2
fi
echo "Walking model: $go2_checkpoint  (policy: $go2_policy)"
exec bash "$go2_scripts/run_go2_real_docker.sh" -m go2_autonomous_driving.real.real_closed_loop \
  --checkpoint "$go2_checkpoint" \
  --policy-dir "$go2_policy" \
  --fault-log log/go2_real_closed_loop_fault.json \
  --config src/go2_autonomous_driving/config/real_closed_loop.json "${go2_args[@]}"
