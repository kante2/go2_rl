#!/usr/bin/env bash
# Sourced by setup/run helpers. Export only this display's cookie, never xhost +.
go2_prepare_x11() {
  local go2_display_number go2_entries go2_auth_tmp
  if [[ ! ${DISPLAY:-} =~ ^(:|unix:)([0-9]+)(\.[0-9]+)?$ ]]; then
    echo 'GUI mode needs a local desktop DISPLAY (for example :0 or :1).' >&2
    return 1
  fi
  go2_display_number="${BASH_REMATCH[2]}"
  if [[ ! -S /tmp/.X11-unix/X${go2_display_number} ]]; then
    echo "X11 socket /tmp/.X11-unix/X${go2_display_number} is missing." >&2
    return 1
  fi
  if ! command -v xauth >/dev/null 2>&1; then
    echo 'GUI mode needs xauth on the host (Ubuntu: sudo apt install xauth).' >&2
    return 1
  fi
  go2_entries="$(xauth nlist "$DISPLAY")" || return 1
  if [[ -z "$go2_entries" ]]; then
    echo 'No X11 cookie found. Run from the desktop session with its XAUTHORITY.' >&2
    return 1
  fi
  export GO2_XAUTH_DIR="${GO2_XAUTH_DIR:-${XDG_RUNTIME_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}}/go2-rl-x11}"
  (
    umask 077
    mkdir -p "$GO2_XAUTH_DIR"
    go2_auth_tmp="$(mktemp "$GO2_XAUTH_DIR/.Xauthority.XXXXXX")" || exit 1
    # FamilyWild permits the container hostname; the secret remains display-specific.
    if ! printf '%s\n' "$go2_entries" | sed 's/^..../ffff/' | xauth -f "$go2_auth_tmp" nmerge -; then
      rm -f "$go2_auth_tmp"
      exit 1
    fi
    # Mount the directory so refreshes also reach an already-running container.
    mv -f "$go2_auth_tmp" "$GO2_XAUTH_DIR/Xauthority"
  )
}
