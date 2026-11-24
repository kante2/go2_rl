#!/usr/bin/env bash
set -euo pipefail
go2_replay_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$go2_replay_dir/serve.py" "$@"
