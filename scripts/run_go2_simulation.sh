#!/usr/bin/env bash
# MuJoCo physics and the trained policy run together in Docker.
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$go2_scripts/run_go2_docker.sh" python -m go2_autonomous_driving.sim.simulation "$@"
