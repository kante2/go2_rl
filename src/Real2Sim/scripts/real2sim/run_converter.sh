#!/usr/bin/env bash
# Default converter now maps existing logs only; never launches a simulator.
set -euo pipefail
go2_converter_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec python3 "$go2_converter_root/src/Real2Sim/scripts/real2sim/pipeline.py" "$@"
