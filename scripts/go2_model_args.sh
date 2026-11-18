#!/usr/bin/env bash
# Sourced helper: pick the walking model from --model / --checkpoint in "$@".
#   --model 34997             -> src/go2_autonomous_driving/model_pt/model_34997.pt
#   --model model_34997.pt    -> same folder
#   --model path/to/x.pt      -> that file (project-relative or absolute inside the project)
#   --checkpoint path/to/x.pt -> unchanged behaviour
# Sets go2_checkpoint (project-relative path; empty if none given) and go2_args (the rest).
go2_parse_model_args() {
  local project="$1" default="$2"; shift 2
  local model=""
  go2_checkpoint=""
  go2_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model|--checkpoint) model="$2"; shift 2 ;;
      --model=*|--checkpoint=*) model="${1#*=}"; shift ;;
      *) go2_args+=("$1"); shift ;;
    esac
  done
  model="${model:-$default}"
  [[ -z "$model" ]] && return 0
  if [[ "$model" =~ ^[0-9]+$ ]]; then
    model="src/go2_autonomous_driving/model_pt/model_${model}.pt"
  elif [[ "$model" != */* ]]; then
    model="src/go2_autonomous_driving/model_pt/${model%.pt}.pt"
  fi
  # Docker sees the project at another path, so keep it project-relative.
  local resolved
  resolved="$(cd -- "$project" && realpath -m --relative-to="$project" -- "$model")"
  if [[ "$resolved" == ../* || ! -f "$project/$resolved" ]]; then
    echo "Model not found inside the project: $model" >&2
    echo "Available: $(cd -- "$project/src/go2_autonomous_driving/model_pt" && ls *.pt | tr '\n' ' ')" >&2
    return 2
  fi
  go2_checkpoint="$resolved"
}

# Real-robot policy folder for a checkpoint: artifacts/go2_real/baseline/<sha256>.
go2_policy_dir() {
  local project="$1" checkpoint="$2"
  echo "artifacts/go2_real/baseline/$(sha256sum "$project/$checkpoint" | cut -d' ' -f1)"
}
