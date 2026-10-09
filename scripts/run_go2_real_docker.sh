#!/usr/bin/env bash
# Enter Docker first; configure robot DDS only for the requested process.
set -euo pipefail
go2_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f /opt/go2/container ]]; then
  exec bash "$go2_scripts/run_go2_docker.sh" env \
    GO2_NETWORK_INTERFACE="${GO2_NETWORK_INTERFACE:-enp3s0}" \
    bash scripts/run_go2_real_docker.sh "$@"
fi
cd "$go2_scripts/.."
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
go2_iface="${GO2_NETWORK_INTERFACE:-enp3s0}"
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$go2_iface\"/></Interfaces></General></Domain></CycloneDDS>"
exec /usr/local/bin/go2-env go2-real-python "$@"
