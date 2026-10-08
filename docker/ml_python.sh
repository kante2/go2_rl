#!/bin/sh
# ROS binary extensions target Python 3.12; keep them out of ML Python 3.11.
export PYTHONPATH=/workspace/unitree_rl_mjlab/src/go2_autonomous_driving:/workspace/unitree_rl_mjlab
exec /usr/local/bin/python3.11 "$@"
