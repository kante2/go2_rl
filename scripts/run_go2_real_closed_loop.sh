#!/usr/bin/env bash
# Actual LowState closes the learned controller; default is motor-free dry-run.
# --model N (or a .pt path) picks the walking model; its converted policy must exist in
# artifacts/go2_real/baseline/<sha256> (create it with prepare_go2_real_robot.sh --model N).
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
go2_project="$(cd -- "$go2_scripts/.." && pwd)"
source "$go2_scripts/go2_model_args.sh"
go2_parse_model_args "$go2_project" 34997 "$@"

# Noise-conditioned 48-input models use real_quietpaw.py with their own converted policy
# (prepare_go2_real_quietpaw.sh --checkpoint PT); same transport, guards and keyboard:
#   QuietPaw (e.g. --model 9999): artifacts/go2_real_quietpaw,          --quietness 0..1 (default 0.5)
#   QPace    (e.g. --model 1000): artifacts/go2_real_qpace/<sha256>,    --target-db dB (default reference_db)
# Options that do not apply to the chosen model are dropped with a note, so one command fits every model.
go2_sha="$(sha256sum "$go2_project/$go2_checkpoint" | cut -d' ' -f1)"
go2_drop_options() {  # go2_drop_options <option>... : remove "--opt V" / "--opt=V" from go2_args
  local kept=() i drop
  for ((i = 0; i < ${#go2_args[@]}; i++)); do
    drop=""
    for opt in "$@"; do
      if [[ "${go2_args[i]}" == "$opt" ]]; then drop=2; elif [[ "${go2_args[i]}" == "$opt="* ]]; then drop=1; fi
    done
    if [[ -n "$drop" ]]; then
      echo "Note: ${go2_args[i]%%=*} ignored for this model" >&2
      [[ "$drop" == 2 ]] && ((i++)) || true
    else
      kept+=("${go2_args[i]}")
    fi
  done
  go2_args=("${kept[@]}")
}
go2_family="" go2_condition_policy=""
if [[ -f "$go2_project/artifacts/go2_real_quietpaw/policy.json" ]] \
    && grep -q "\"checkpoint_sha256\": \"$go2_sha\"" "$go2_project/artifacts/go2_real_quietpaw/policy.json"; then
  go2_family=QuietPaw go2_condition_policy="artifacts/go2_real_quietpaw"
  go2_drop_options --target-db
elif [[ -f "$go2_project/artifacts/go2_real_qpace/$go2_sha/policy.json" ]]; then
  go2_family=QPace go2_condition_policy="artifacts/go2_real_qpace/$go2_sha"
  go2_drop_options --quietness
fi
if [[ -n "$go2_family" ]]; then
  echo "Walking model: $go2_checkpoint  ($go2_family policy: $go2_condition_policy)"
  exec bash "$go2_scripts/run_go2_real_docker.sh" -m go2_autonomous_driving.real.real_quietpaw \
    --checkpoint "$go2_checkpoint" \
    --policy-dir "$go2_condition_policy" \
    --fault-log log/go2_real_quietpaw_fault.json \
    --config src/go2_autonomous_driving/config/real_closed_loop.json "${go2_args[@]}"
fi
# Baseline (47-input) models have no noise condition.
go2_drop_options --quietness --target-db
go2_policy="artifacts/go2_real/baseline/$go2_sha"
if [[ ! -f "$go2_project/$go2_policy/policy.json" ]]; then
  echo "No converted real-robot policy for $go2_checkpoint." >&2
  echo "Run first: bash scripts/prepare_go2_real_robot.sh --model $go2_checkpoint" >&2
  echo "(QuietPaw/QPace 48-input model: bash scripts/prepare_go2_real_quietpaw.sh --checkpoint $go2_checkpoint)" >&2
  exit 2
fi
echo "Walking model: $go2_checkpoint  (policy: $go2_policy)"
exec bash "$go2_scripts/run_go2_real_docker.sh" -m go2_autonomous_driving.real.real_closed_loop \
  --checkpoint "$go2_checkpoint" \
  --policy-dir "$go2_policy" \
  --fault-log log/go2_real_closed_loop_fault.json \
  --config src/go2_autonomous_driving/config/real_closed_loop.json "${go2_args[@]}"
