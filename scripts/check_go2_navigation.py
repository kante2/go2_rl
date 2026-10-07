#!/usr/bin/env python3
"""Workspace integration check: press W once, verify Go2 motion and stopping."""

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped, TwistStamped
from rclpy.parameter import Parameter
from nav_msgs.msg import Path
from std_msgs.msg import String

from go2_autonomous_driving.keyboard_node import KeyboardWaypoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--continuous", action="store_true",
                        help="Check that one key press keeps moving beyond the initial waypoint")
    parser.add_argument("--require-reached", action="store_true",
                        help="Also require the bridge to report waypoint reached")
    args = parser.parse_args()
    if args.continuous and args.require_reached:
        parser.error("--continuous and --require-reached cannot be combined")
    rclpy.init()
    node = KeyboardWaypoint(parameter_overrides=[Parameter("continuous", value=args.continuous)])
    statuses, commands, paths, goals = [], [], [], []
    node.create_subscription(String, "/go2/navigation_status", lambda msg: statuses.append(msg.data), 10)
    node.create_subscription(TwistStamped, "/go2/locomotion_command",
                             lambda msg: commands.append((time.monotonic(), msg.twist.linear.x,
                                                          msg.twist.linear.y, msg.twist.angular.z)), 10)
    node.create_subscription(Path, "/path", lambda msg: paths.append(len(msg.poses)), 5)
    node.create_subscription(PointStamped, "/go2/planner_way_point", lambda msg: goals.append(msg), 10)

    def spin_for(duration):
        end = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)

    try:
        deadline = time.monotonic() + 45
        while node.pose is None and time.monotonic() < deadline:
            spin_for(0.1)
        if node.pose is None:
            raise RuntimeError("No MuJoCo odometry: start simulation and ROS launch first")
        spin_for(0.5)
        if not commands or any(abs(v) > 1e-6 for v in commands[-1][1:]):
            raise RuntimeError("Expected zero navigation command before the first waypoint")
        start = node.pose
        started_at = time.monotonic()
        node.handle_key("w")
        spin_for(args.duration)
        finish = node.pose
        progress = ((finish.x - start.x) * math.cos(start.yaw)
                    + (finish.y - start.y) * math.sin(start.yaw))
        peak = max((abs(v) for command in commands for v in command[1:]), default=0)
        cruising = [command for command in commands if command[0] >= started_at + 2]
        moving_fraction = (sum(math.hypot(command[1], command[2]) > 0.01 for command in cruising)
                           / len(cruising)) if cruising else 0
        node.handle_key(" ")
        spin_for(0.5)
        stopped = bool(commands) and all(abs(v) < 1e-6 for v in commands[-1][1:])
        result = {"statuses": statuses, "path_messages": len(paths),
                  "goal_messages": len(goals), "moving_fraction_after_2s": round(moving_fraction, 3),
                  "peak_command": round(peak, 4), "forward_progress_m": round(progress, 4),
                  "stopped": stopped, "start": [start.x, start.y], "finish": [finish.x, finish.y]}
        print(json.dumps(result, indent=2), flush=True)
        if not (any(count > 1 for count in paths) and "following waypoint" in statuses
                and peak > 0.01 and progress > 0.15 and stopped):
            raise RuntimeError("Simulation navigation smoke check failed; inspect the metrics above")
        if args.require_reached and "waypoint reached" not in statuses:
            raise RuntimeError("Waypoint was not reached within the requested duration")
        if args.continuous and (len(goals) < 3 or progress < 1.2 or moving_fraction < 0.9):
            raise RuntimeError("Continuous motion did not persist beyond the first waypoint")
    finally:
        node.stop()
        spin_for(0.1)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
