#!/usr/bin/env bash
# ROS/Unitree/ONNX dependencies are built into the Docker image.
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$go2_scripts/run_go2_docker.sh" python scripts/export_go2_real_policy.py "$@"
bash "$go2_scripts/run_go2_docker.sh" go2-real-python scripts/check_go2_real_policy.py
echo 'Prepared in Docker. Default run_go2_real_closed_loop.sh only reads sensors and runs inference.'
