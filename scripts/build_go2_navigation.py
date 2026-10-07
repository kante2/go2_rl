#!/usr/bin/env python3
"""Build the Go2 ROS subpackage and planner dependencies at workspace level."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


def call(arguments, env=None):
    subprocess.run([str(arg) for arg in arguments], check=True, env=env)


def clone(url, branch, target):
    if not (target / ".git").is_dir():
        call(["git", "clone", "--depth", "1", "--branch", branch, url, target])


def main():
    project = Path(__file__).resolve().parents[1]
    package = project / "src/go2_autonomous_driving"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-root", type=Path,
                        default=project / "Navigation-Physical-Experiment")
    parser.add_argument("--workspace-root", type=Path, default=project,
                        help="Store dependencies and build/install/log outside the package")
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    if workspace == package or package in workspace.parents:
        raise SystemExit("The workspace root must be outside src/go2_autonomous_driving")
    upstream = args.planner_root.resolve()
    original = upstream / "src/base_autonomy/local_planner"
    serial = upstream / "src/utilities/serial"
    if not (original / "src/localPlanner.cpp").is_file():
        raise SystemExit("Clone Yuxin916/Navigation-Physical-Experiment (jazzy branch) first, or use --planner-root")
    sources = workspace / "third_party/go2_navigation/src"
    sources.mkdir(parents=True, exist_ok=True)
    clone("https://github.com/ros-perception/pcl_msgs.git", "ros2", sources / "pcl_msgs")
    clone("https://github.com/ros-perception/perception_pcl.git", "jazzy", sources / "perception_pcl")
    # Keep the user's upstream checkout untouched. pcl_ros is not used by either
    # executable; pcl_conversions already exports the PCL includes and libraries.
    copy = sources / "local_planner"
    shutil.copytree(original, copy, dirs_exist_ok=True)
    cmake = copy / "CMakeLists.txt"
    cmake.write_text(cmake.read_text().replace("find_package(pcl_ros REQUIRED)\n", "")
                     .replace(" pcl_ros ", " "))
    manifest = copy / "package.xml"
    tree = ET.parse(manifest)
    for dependency in list(tree.getroot()):
        if dependency.text == "pcl_ros":
            tree.getroot().remove(dependency)
    tree.write(manifest, encoding="unicode")
    # Jazzy's Rate.sleep can throw after SIGINT invalidates its context. Keep
    # shutdown handling in our build copy; leave the upstream algorithms intact.
    for name in ("localPlanner.cpp", "pathFollower.cpp"):
        source = copy / "src" / name
        code = source.read_text().replace("while (status) {", "while (rclcpp::ok()) {")
        code = code.replace("  bool status = rclcpp::ok();\n", "")
        code = code.replace("    status = rclcpp::ok();\n", "")
        code = code.replace("    rate.sleep();", """    try {
      rate.sleep();
    } catch (const rclcpp::exceptions::RCLError&) {
      if (rclcpp::ok()) throw;
      break;
    }""")
        # Check the current context if shutdown arrives before spin_some starts.
        code = code.replace("    rclcpp::spin_some(nh);", """    try {
      rclcpp::spin_some(nh);
    } catch (const rclcpp::exceptions::RCLError&) {
      if (rclcpp::ok()) throw;
      break;
    }""")
        # The upstream node pointer is global: release it while the ROS context
        # still exists, before global static destruction runs after main exits.
        code = code.replace("  return 0;\n}", "  nh.reset();\n  return 0;\n}")
        source.write_text(code)
    call([
        "colcon", "--log-base", workspace / "log", "build",
        "--base-paths", sources / "pcl_msgs", sources / "perception_pcl/pcl_conversions",
        copy, serial, package,
        "--packages-select", "pcl_msgs", "pcl_conversions", "serial", "local_planner",
        "go2_autonomous_driving", "--executor", "sequential",
        "--build-base", workspace / "build", "--install-base", workspace / "install",
        "--symlink-install", "--cmake-args", "-DBUILD_TESTING=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ], env=dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL="2", PYTHONDONTWRITEBYTECODE="1"))
    print(f"Build finished. Source {workspace / 'install/setup.bash'}")


if __name__ == "__main__":
    main()
