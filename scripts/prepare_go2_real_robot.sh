#!/usr/bin/env bash
# ROS/Unitree/ONNX dependencies are built into the Docker image.
# --model N (or a .pt path): export that Flat 47-input model to artifacts/go2_real/baseline/<sha256>
# and check ONNX/PyTorch parity. Each model keeps its own folder; nothing is overwritten.
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
go2_project="$(cd -- "$go2_scripts/.." && pwd)"
source "$go2_scripts/go2_model_args.sh"
go2_parse_model_args "$go2_project" 34997 "$@"
go2_policy="$(go2_policy_dir "$go2_project" "$go2_checkpoint")"
echo "Walking model: $go2_checkpoint -> $go2_policy"
bash "$go2_scripts/run_go2_docker.sh" python scripts/export_go2_real_policy.py \
  --checkpoint "$go2_checkpoint" --output "$go2_policy" "${go2_args[@]}"
bash "$go2_scripts/run_go2_docker.sh" go2-real-python scripts/check_go2_real_policy.py --policy-dir "$go2_policy"
echo "Prepared. Run: bash scripts/run_go2_real_closed_loop.sh --model $go2_checkpoint (add --execute for motors)"
