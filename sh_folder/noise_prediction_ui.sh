#!/usr/bin/env bash
# Noise prediction dashboard (v2) in the Go2 container: /lowstate -> predicted dBFS,
# robot UMIK-1 (SSH) -> measured dBFS. Open http://localhost:8081 on the host.
# Port 8081 avoids the real-robot pose viewer on 8080. Extra args override, e.g. --web-port 8082.
# Do not run with terminal4_noise_record.sh: both open the same robot microphone.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$go2_project/scripts/run_go2_docker.sh" \
  bash src/db_prediction/Prediction/run_live_v2.sh --card UMIK1 --web-port 8081 "$@"
