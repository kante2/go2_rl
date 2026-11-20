#!/usr/bin/env bash
# R4 녹화 세션 (호스트): 실로봇 ROS 상태 + 로봇 UMIK-1 마이크 + 시뮬(UDP 9881)을 Enter 한 번으로 함께 녹화.
# 실로봇 제어(R2)보다 먼저 켜서 계속 유지한다. Enter = 녹화 시작, q = 종료.
# 예: bash sh_folder/record_session.sh --base NORMAL_0.5_0.5 --dir src/sim_to_real_data/real_sim --duration 30
set -eo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HOME/unitree_ros2/setup.sh"
export ROS_DOMAIN_ID=0  # real-robot DDS domain
cd "$go2_project"
exec python3 src/db_collector/record_session.py \
  --host unitree@192.168.123.18 --card UMIK1 "$@"
