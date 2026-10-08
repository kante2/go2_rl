#!/bin/sh
# Run with sh or bash on the host; all terminals join the same container.
set -eu
go2_scripts="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec bash "$go2_scripts/run_go2_docker.sh" bash -i
