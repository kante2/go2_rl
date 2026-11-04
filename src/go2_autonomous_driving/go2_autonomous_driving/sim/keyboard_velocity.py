"""Direct body-velocity keyboard for the simulation; bypasses the waypoint planner.

W forward, X backward, A/D lateral left/right, S stop, Q/E rotate counter-clockwise/clockwise in place.
A command stays active until another key; Ctrl+C quits. The bridge forwards
/go2/keyboard_velocity to MuJoCo while it is fresh.
"""

import os
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Header

from ..core import Velocity


def decode_velocity_keys(buffer):
    """Arrows map to W/S/A/D (Down = stop) without leaking their trailing letters."""
    keys = []
    while buffer:
        if buffer[0] == "\x1b":
            if len(buffer) < 3:
                break
            if buffer[1] == "[":
                key = {"A": "w", "B": "s", "C": "d", "D": "a"}.get(buffer[2])
                if key:
                    keys.append(key)
                buffer = buffer[3:]
            else:
                buffer = buffer[2:]
        else:
            key, buffer = buffer[0].lower(), buffer[1:]
            if key in ("w", "a", "s", "d", "x", "q", "e", " ", "\x03"):
                keys.append(key)
    return keys, buffer


class KeyboardVelocity(Node):
    def __init__(self):
        super().__init__("keyboard_velocity")
        self.forward = float(self.declare_parameter("forward_speed", 0.5).value)
        self.lateral = float(self.declare_parameter("lateral_speed", 0.2).value)
        self.yaw = float(self.declare_parameter("yaw_rate", 0.5).value)
        self.command = Velocity()
        self.publisher = self.create_publisher(TwistStamped, "/go2/keyboard_velocity", 10)
        # Republish the latched command; the bridge drops it once it goes stale.
        self.create_timer(0.05, self.publish)

    def handle_key(self, key):
        directions = {"w": Velocity(self.forward, 0, 0),
                      "x": Velocity(-self.forward, 0, 0),
                      "a": Velocity(0, self.lateral, 0),
                      "d": Velocity(0, -self.lateral, 0),
                      "s": Velocity(),
                      " ": Velocity(),
                      "q": Velocity(0, 0, self.yaw),    # counter-clockwise
                      "e": Velocity(0, 0, -self.yaw)}   # clockwise
        if key in directions:
            self.command = directions[key]
            self.publish()
            c = self.command
            print(f"command=[{c.vx:.2f}, {c.vy:.2f}, {c.wz:.2f}]", flush=True)

    def publish(self):
        message = TwistStamped()
        message.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="vehicle")
        message.twist.linear.x = float(self.command.vx)
        message.twist.linear.y = float(self.command.vy)
        message.twist.angular.z = float(self.command.wz)
        self.publisher.publish(message)


def main(args=None):
    if not sys.stdin.isatty():
        raise SystemExit("Run keyboard_velocity in an interactive terminal")
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = KeyboardVelocity()
    original = termios.tcgetattr(sys.stdin)
    print(f"W/Up: forward, X: backward {node.forward:.2f} m/s, A/D/Left/Right: lateral {node.lateral:.2f} m/s, "
          f"S/Down/Space: stop, Q/E: rotate CCW/CW {node.yaw:.2f} rad/s; "
          "each command holds until the next key. Ctrl+C: quit", flush=True)
    buffer = ""
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if select.select([sys.stdin], [], [], 0)[0]:
                buffer += os.read(sys.stdin.fileno(), 32).decode(errors="ignore")
                keys, buffer = decode_velocity_keys(buffer)
                if "\x03" in keys:
                    break
                for key in keys:
                    node.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
        node.command = Velocity()
        node.publish()
        rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
