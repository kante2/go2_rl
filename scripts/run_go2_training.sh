#!/usr/bin/env bash
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then set -- Unitree-Go2-Flat; fi
exec bash "$go2_scripts/run_go2_docker.sh" python scripts/train.py "$@"
