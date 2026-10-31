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
elif [[ -n ${DISPLAY:-} ]]; then
  # Container without the X11 mount: host networking still reaches Xorg's abstract
  # socket, so copy only this display's cookie instead of recreating the container.
  source "$(dirname -- "${BASH_SOURCE[0]}")/prepare_go2_x11.sh"
  go2_xauth_tmp="$(mktemp -d)"
  if (GO2_XAUTH_DIR="$go2_xauth_tmp" go2_prepare_x11) 2>/dev/null \
      && docker exec -i "$go2_container" sh -c \
        'umask 077; mkdir -p /tmp/go2-xauth-exec && cat > /tmp/go2-xauth-exec/Xauthority' \
        < "$go2_xauth_tmp/Xauthority"; then
    go2_display_env=(-e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/go2-xauth-exec/Xauthority)
  fi
  rm -rf -- "$go2_xauth_tmp"
fi
# Hybrid laptop: the desktop runs on the iGPU; render on NVIDIA via PRIME offload
# (the container's "graphics" capability already provides the NVIDIA GLX library).
if [[ ${#go2_display_env[@]} -gt 0 ]] && xrandr --listproviders 2>/dev/null | grep -q 'name:NVIDIA-G'; then
  go2_display_env+=(-e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia)
fi
exec docker exec "${go2_tty[@]}" "${go2_display_env[@]}" -w /workspace/unitree_rl_mjlab \
  "$go2_container" /usr/local/bin/go2-env "$@"
