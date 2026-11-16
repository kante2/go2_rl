#!/usr/bin/env bash
# RL 자동 주행 + 로봇 state/모터 명령 기록. RL 제어기 실행/ARM은 별도 터미널.
# 예: bash src/db_collector/data_collector_RL.sh --interface enx00e04c362347 --vx 0.2 --duration 30 --out 0.2_3
set -euo pipefail

collector_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$collector_dir/../.." && pwd)"
cd -- "$project_dir"

exec bash "$project_dir/scripts/run_go2_docker.sh" \
  go2-real-python src/db_collector/scripts/RL_data_collector.py \
  --interface "${GO2_NETWORK_INTERFACE:-enx00e04c362347}" \
  --name baseline_walk \
  --duration 15 \
  "$@"
