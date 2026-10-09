#!/usr/bin/env bash
# Explicit operator recovery; no low-level motor or StandUp commands.
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$go2_scripts/run_go2_real_docker.sh" scripts/recover_go2_normal.py "$@"
