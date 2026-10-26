#!/usr/bin/env bash
# Run after sourcing ROS 2 and the built Unitree workspace.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$PROJECT_ROOT"

exec python3 Prediction/predict_live.py \
  --model contact_force_velocity/models/causal_forecast/all_four.joblib \
  --audio-host unitree@192.168.123.18 \
  --card Microphone \
  --web-port 8080 \
  "$@"
