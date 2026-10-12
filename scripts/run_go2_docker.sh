#!/usr/bin/env bash
# Host: docker exec. Inside the Go2 image: run directly, without nested Docker.
set -euo pipefail
if [[ $# -eq 0 ]]; then set -- bash; fi
if [[ -f /opt/go2/container ]]; then
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
  exec /usr/local/bin/go2-env "$@"
fi
go2_container="${GO2_MJLAB_CONTAINER:-unitree-rl-mjlab}"
if ! docker inspect --type container "$go2_container" >/dev/null 2>&1; then
  echo "Container $go2_container is missing. Run bash scripts/setup_go2_docker.sh first." >&2
  exit 1
fi
docker start "$go2_container" >/dev/null
if ! docker exec "$go2_container" test -x /usr/local/bin/go2-env; then
  echo 'This container predates the ROS image. Run bash scripts/setup_go2_docker.sh to rebuild.' >&2
  exit 1
fi
go2_tty=(-i)
if [[ -t 0 && -t 1 ]]; then go2_tty+=(-t); fi
go2_display_env=()
go2_auth_mount="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/tmp/go2-xauth"}}{{.Source}}{{end}}{{end}}' "$go2_container")"
if [[ -n "$go2_auth_mount" && -n ${DISPLAY:-} ]]; then
  source "$(dirname -- "${BASH_SOURCE[0]}")/prepare_go2_x11.sh"
  GO2_XAUTH_DIR="$go2_auth_mount"
  go2_prepare_x11
  go2_display_env=(-e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/go2-xauth/Xauthority)
fi
exec docker exec "${go2_tty[@]}" "${go2_display_env[@]}" -w /workspace/unitree_rl_mjlab \
  "$go2_container" /usr/local/bin/go2-env "$@"
