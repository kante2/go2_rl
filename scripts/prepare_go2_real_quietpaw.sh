#!/usr/bin/env bash
# Export in Docker, verify on the host, then publish only validated artifacts.
set -euo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$go2_project"
go2_checkpoint="src/go2_autonomous_driving/model_pt/model_9999.pt"
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != --checkpoint ]]; then
    echo 'Usage: prepare_go2_real_quietpaw.sh [--checkpoint PATH]' >&2
    exit 2
  fi
  go2_checkpoint="$2"
fi
# Convert a host workspace path to the relative path shared with Docker.
go2_checkpoint="$(realpath --relative-to="$go2_project" -- "$go2_checkpoint")"
if [[ "$go2_checkpoint" == ../* || ! -f "$go2_checkpoint" ]]; then
  echo 'Checkpoint must be an existing file inside the go2_rl workspace.' >&2
  exit 2
fi
# The existing artifacts/ may be owned by the Docker user (root).
go2_stage="$(mktemp -d .go2_quietpaw_export_XXXXXX)"
trap 'rm -rf -- "$go2_stage"' EXIT
bash "$go2_project/scripts/run_go2_docker.sh" python scripts/export_go2_quietpaw_policy.py \
  --checkpoint "$go2_checkpoint" --output "$go2_stage"
PYTHONPATH="$go2_project/src/go2_autonomous_driving:${PYTHONPATH:-}" \
  "$go2_project/.venv_go2_real/bin/python" scripts/check_go2_real_policy.py --policy-dir "$go2_stage"
bash "$go2_project/scripts/run_go2_docker.sh" python - "$go2_stage" "$(id -u)" "$(id -g)" <<'PY'
import os
from pathlib import Path
import sys
stage = Path(sys.argv[1])
output = Path('artifacts/go2_real_quietpaw')
output.mkdir(parents=True, exist_ok=True)
owner = (int(sys.argv[2]), int(sys.argv[3]))
os.chown(output, *owner)
# Metadata last: a partial replacement cannot pass the runner's hash check.
for name in ('policy.onnx', 'parity.npz', 'samples.json', 'policy.json'):
    os.replace(stage / name, output / name)
    os.chown(output / name, *owner)
PY
echo 'Prepared artifacts/go2_real_quietpaw. Baseline artifacts/go2_real was not changed.'
