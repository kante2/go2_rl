#!/usr/bin/env bash
# Dependencies/artifacts belong to the workspace, never inside the ROS subpackage.
set -eo pipefail
go2_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$go2_project"
go2_pip_python="$(command -v python3)"
go2_ros_source="$go2_project/third_party/go2_navigation/unitree_ros2"
if [[ ! -d "$go2_ros_source/.git" ]]; then
  git clone https://github.com/unitreerobotics/unitree_ros2.git "$go2_ros_source"
  git -C "$go2_ros_source" checkout 668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
fi
source /opt/ros/jazzy/setup.bash
colcon --log-base log build --base-paths \
  "$go2_ros_source/cyclonedds_ws/src/unitree/unitree_go" \
  "$go2_ros_source/cyclonedds_ws/src/unitree/unitree_api" \
  src/go2_autonomous_driving \
  --packages-select unitree_go unitree_api go2_autonomous_driving \
  --build-base build --install-base install --symlink-install --executor sequential \
  --cmake-args -DBUILD_TESTING=OFF
if [[ ! -x .venv_go2_real/bin/python ]]; then
  /usr/bin/python3 -m venv --without-pip --system-site-packages .venv_go2_real
fi
if ! .venv_go2_real/bin/python -m pip --version >/dev/null 2>&1; then
  go2_site_packages="$(.venv_go2_real/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
  "$go2_pip_python" -m pip install --target "$go2_site_packages" pip
fi
.venv_go2_real/bin/python -m pip install 'onnxruntime==1.23.2'
bash "$go2_project/scripts/run_go2_docker.sh" python scripts/export_go2_real_policy.py "$@"
PYTHONPATH="$go2_project/src/go2_autonomous_driving:${PYTHONPATH:-}" \
  .venv_go2_real/bin/python scripts/check_go2_real_policy.py
echo 'Prepared. Default run_go2_real_closed_loop.sh only reads sensors and runs inference.'
