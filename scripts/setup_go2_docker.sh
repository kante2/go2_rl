#!/usr/bin/env bash
# Build the environment and start an idle container; never enable robot motors.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
go2_cpu=false
go2_build=true
go2_gui=false
for go2_arg in "$@"; do
  case "$go2_arg" in
    --cpu) go2_cpu=true ;;
    --no-build) go2_build=false ;;
    --gui) go2_gui=true ;;
    -h|--help)
      echo 'Usage: setup_go2_docker.sh [--cpu] [--no-build] [--gui]'
      echo 'Default: NVIDIA training/simulation. --cpu: real-state viewer only.'
      echo '--gui: share the local X11/XWayland display for native MuJoCo windows.'
      exit 0 ;;
    *) echo "Unknown argument: $go2_arg" >&2; exit 2 ;;
  esac
done
docker info >/dev/null
docker compose version
go2_compose=(docker compose -f "$go2_project/docker/docker-compose.yml")
if "$go2_cpu"; then
  go2_compose+=(-f "$go2_project/docker/docker-compose.cpu.yml")
else
  if ! nvidia-smi >/dev/null 2>&1; then
    echo 'Host NVIDIA GPU is unavailable. Restore nvidia-smi before GPU simulation/training.' >&2
    echo 'For the receive-only real-state viewer: bash scripts/setup_go2_docker.sh --cpu' >&2
    exit 1
  fi
fi
if "$go2_gui"; then
  source "$go2_project/scripts/prepare_go2_x11.sh"
  go2_prepare_x11
  go2_compose+=(-f "$go2_project/docker/docker-compose.gui.yml")
fi
"${go2_compose[@]}" config --quiet
if "$go2_build"; then
  "${go2_compose[@]}" build
fi
"${go2_compose[@]}" up -d --no-build
go2_check=()
if ! "$go2_cpu"; then go2_check+=(--require-cuda); fi
if "$go2_gui"; then go2_check+=(--require-display); fi
GO2_MJLAB_CONTAINER=unitree-rl-mjlab bash "$go2_project/scripts/run_go2_docker.sh" \
  python scripts/check_go2_docker.py "${go2_check[@]}"
