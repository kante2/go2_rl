#!/usr/bin/env bash
# Explicit operator recovery; no low-level motor or StandUp commands.
set -eo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
go2_iface="${GO2_NETWORK_INTERFACE:-enp3s0}"
source /opt/ros/jazzy/setup.bash
source "$go2_project/install/setup.bash"
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$go2_iface\"/></Interfaces></General></Domain></CycloneDDS>"
exec /usr/bin/python3 "$go2_project/scripts/recover_go2_normal.py" "$@"
