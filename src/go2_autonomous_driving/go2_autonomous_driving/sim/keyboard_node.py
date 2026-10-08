"""Latch a travel direction and refresh waypoints until the user stops."""

import math
import os
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Int8

from ..core import Pose, WaypointIntent, decode_keys, yaw_from_wxyz


class KeyboardWaypoint(Node):
    def __init__(self, *, parameter_overrides=None):
        super().__init__("keyboard_waypoint", parameter_overrides=parameter_overrides)
        self.distance = float(self.declare_parameter("waypoint_distance", 1.0).value)
        self.timeout = float(self.declare_parameter("pose_timeout", 0.5).value)
        self.continuous = bool(self.declare_parameter("continuous", True).value)
        update_period = float(self.declare_parameter("waypoint_update_period", 0.2).value)
        if any(not math.isfinite(v) or v <= 0 for v in (self.timeout, update_period)):
            raise ValueError("pose_timeout and waypoint_update_period must be positive and finite")
        self.intent = WaypointIntent(self.distance)
        self.intent_frame = None
        self.pose, self.pose_time, self.frame = None, -math.inf, "map"
        self.enabled = False
        self.goal_pub = self.create_publisher(PointStamped, "/way_point", 10)
        self.update_pub = self.create_publisher(PointStamped, "/go2/waypoint_update", 10)
        self.enable_pub = self.create_publisher(Bool, "/go2/navigation_enabled", 10)
        self.stop_pub = self.create_publisher(Int8, "/stop", 10)
        self.create_subscription(Odometry, "/state_estimation", self.on_pose, 10)
        self.create_timer(0.1, self.heartbeat)
        self.create_timer(update_period, self.refresh_waypoint)

    def on_pose(self, msg):
        q, p = msg.pose.pose.orientation, msg.pose.pose.position
        try:
            pose = Pose(p.x, p.y, p.z, yaw_from_wxyz((q.w, q.x, q.y, q.z)))
            if not all(math.isfinite(v) for v in (p.x, p.y, p.z)) or not msg.header.frame_id:
                return
        except ValueError:
            return
        self.pose, self.pose_time, self.frame = pose, time.monotonic(), msg.header.frame_id

    def heartbeat(self):
        if self.enabled and not self.pose_is_current():
            self.stop()
            self.get_logger().warning("Localization lost: press a direction key again after recovery")
            return
        self.enable_pub.publish(Bool(data=self.enabled))

    def stop(self):
        self.enabled = False
        self.intent.stop()
        self.heartbeat()
        self.stop_pub.publish(Int8(data=2))

    def pose_is_current(self):
        return (self.pose is not None and time.monotonic() - self.pose_time <= self.timeout
                and (self.intent_frame is None or self.frame == self.intent_frame))

    def publish_goal(self, publisher):
        x, y = self.intent.target(self.pose)
        goal = PointStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.frame
        goal.point.x, goal.point.y, goal.point.z = x, y, self.pose.z
        publisher.publish(goal)
        return x, y

    def refresh_waypoint(self):
        if self.continuous and self.enabled:
            if not self.pose_is_current():
                self.stop()
                return
            self.publish_goal(self.update_pub)

    def handle_key(self, key):
        if key in (" ", "q", "\x03"):
            self.stop()
            return key not in ("q", "\x03")
        directions = {"w": (1, 0), "s": (-1, 0), "a": (0, 1), "d": (0, -1)}
        if key not in directions:
            return True
        if self.pose is None or time.monotonic() - self.pose_time > self.timeout:
            self.get_logger().warning("Current localization is required before setting a waypoint")
            return True
        self.intent.start(self.pose, *directions[key])
        self.intent_frame = self.frame
        x, y = self.publish_goal(self.goal_pub)
        self.stop_pub.publish(Int8(data=0))
        self.enabled = True
        self.heartbeat()
        mode = "Continuous travel" if self.continuous else "Single waypoint"
        self.get_logger().info(f"{mode}: {self.frame} ({x:.2f}, {y:.2f}); Space to stop")
        return True


def main(args=None):
    if not sys.stdin.isatty():
        raise SystemExit("Run keyboard_waypoint in an interactive terminal")
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = KeyboardWaypoint()
    original = termios.tcgetattr(sys.stdin)
    mode = "continuous until Space" if node.continuous else "single waypoint"
    print(f"Mode: {mode}. W/Up: forward, S/Down: backward, A/D/Left/Right: lateral; "
          "Space: stop, Q: quit", flush=True)
    buffer = ""
    running = True
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.02)
            if select.select([sys.stdin], [], [], 0)[0]:
                buffer += os.read(sys.stdin.fileno(), 32).decode(errors="ignore")
                keys, buffer = decode_keys(buffer)
                for key in keys:
                    if not node.handle_key(key):
                        running = False
                        break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
        node.stop()
        rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.try_shutdown()
