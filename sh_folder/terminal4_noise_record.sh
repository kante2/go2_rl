#!/usr/bin/env bash
# Terminal 4 (host): real-robot ROS state + PC microphone noise recording.
# Asks for a name, saves src/db_collector/data/<name>_<YYYYmmdd_HHMMSS>.npz/.wav; q stops and saves.
# Extra args pass through, e.g. --host unitree@192.168.123.18 for the robot microphone.
set -eo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HOME/unitree_ros2/setup.sh"
export ROS_DOMAIN_ID=0  # real-robot DDS domain
cd "$go2_project"
exec python3 src/db_collector/sync_record_rl.py "$@"
