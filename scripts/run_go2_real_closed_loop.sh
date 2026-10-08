#!/usr/bin/env bash
# Actual LowState closes the learned controller; default is motor-free dry-run.
set -eo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
go2_iface="${GO2_NETWORK_INTERFACE:-enp3s0}"
source /opt/ros/jazzy/setup.bash
source "$go2_project/install/setup.bash"
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$go2_iface\"/></Interfaces></General></Domain></CycloneDDS>"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$go2_project/src/go2_autonomous_driving:${PYTHONPATH:-}"
exec "$go2_project/.venv_go2_real/bin/python" -m go2_autonomous_driving.real.real_closed_loop \
  --checkpoint "$go2_project/src/go2_autonomous_driving/model_pt/model_10000.pt" \
  --policy-dir "$go2_project/artifacts/go2_real" \
  --fault-log "$go2_project/log/go2_real_closed_loop_fault.json" \
  --config "$go2_project/src/go2_autonomous_driving/config/real_closed_loop.json" "$@"
