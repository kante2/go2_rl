#!/usr/bin/env bash
set -euo pipefail
go2_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$go2_repo_root"
exec bash scripts/run_go2_docker.sh python src/Real2Sim/scripts/collect_sim_baseline.py "$@"
