"""Bridge the upstream ROS planner and a locomotion backend over local UDP."""

import json
import math
import socket
import time

import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header, String
from tf2_ros import TransformBroadcaster

from ..core import NavigationGate, Pose, Velocity, command_packet, yaw_from_wxyz


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class NavigationBridge(Node):
    def __init__(self):
        super().__init__("navigation_bridge")

        def parameter(name, default):
            return self.declare_parameter(name, default).value

        command_port = int(parameter("command_port", 9871))
        udp_port = int(parameter("udp_port", 9872))

        self.limits = (
            float(parameter("max_forward_speed", 0.5)),
            float(parameter("max_lateral_speed", 0.2)),
            float(parameter("max_yaw_rate", 0.5)),
        )

        self.gate = NavigationGate(
            float(parameter("goal_tolerance", 0.2)),
            float(parameter("sensor_timeout", 0.5)),
            float(parameter("command_timeout", 0.3)),
        )

        # UDP: 상태 수신 포트와 명령 전송 목적지 설정
        self.command_address = ("127.0.0.1", command_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", udp_port))
        self.socket.setblocking(False)

        self.sequence = 0
        self.pose = None

        self.pose_time = self.cloud_time = self.path_time = (
            self.command_time
        ) = -math.inf

        self.path_stamp = self.command_stamp = -math.inf
        self.command = Velocity()
        self.last_reason = None
        self.simulation_session = None

        # ROS publishers
        self.odom_pub = self.create_publisher(
            Odometry, "/state_estimation", 5
        )
        self.scan_pub = self.create_publisher(
            PointCloud2, "/registered_scan", 5
        )
        self.status_pub = self.create_publisher(
            String, "/go2/navigation_status", 10
        )
        self.velocity_pub = self.create_publisher(
            TwistStamped, "/go2/locomotion_command", 5
        )
        self.planner_goal_pub = self.create_publisher(
            PointStamped, "/go2/planner_way_point", 10
        )

        self.transforms = TransformBroadcaster(self)

        # ROS subscription callbacks
        self.create_subscription(
            PointStamped, "/way_point", self.CB_goal, 10
        )
        self.create_subscription(
            PointStamped,
            "/go2/waypoint_update",
            self.CB_goal_update,
            10,
        )
        self.create_subscription(
            Bool,
            "/go2/navigation_enabled",
            self.CB_enable,
            10,
        )
        self.create_subscription(
            Path, "/path", self.CB_path, 5
        )
        self.create_subscription(
            TwistStamped, "/cmd_vel", self.CB_command, 5
        )

        # Timer callbacks
        self.create_timer(0.01, self.receive_udp)
        self.create_timer(0.05, self.send_command)

    def CB_goal(self, msg):
        if msg.header.frame_id.lstrip("/") != "map":
            self.get_logger().warning("Waypoint must use the map frame")
            return

        try:
            self.gate.set_goal(
                msg.point.x,
                msg.point.y,
                stamp_seconds(msg.header.stamp),
            )
        except ValueError:
            self.get_logger().warning("Rejected invalid waypoint")
            return

        self.planner_goal_pub.publish(msg)

    def CB_goal_update(self, msg):
        if msg.header.frame_id.lstrip("/") != "map":
            return

        try:
            accepted = self.gate.continue_goal(
                msg.point.x,
                msg.point.y,
                stamp_seconds(msg.header.stamp),
            )
        except ValueError:
            return

        if accepted:
            self.planner_goal_pub.publish(msg)

    def CB_enable(self, msg):
        self.gate.lease(msg.data, time.monotonic())

    def CB_path(self, msg):
        if (
            len(msg.poses) > 1
            and msg.header.frame_id.lstrip("/") == "vehicle"
        ):
            self.path_time = time.monotonic()
            self.path_stamp = stamp_seconds(msg.header.stamp)
        else:
            self.path_time = -math.inf

    def CB_command(self, msg):
        if msg.header.frame_id.lstrip("/") != "vehicle":
            self.command_time = -math.inf
            return

        try:
            self.command = Velocity(
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.angular.z,
            ).bounded(*self.limits)
        except ValueError:
            self.command_time = -math.inf
            return

        self.command_time = time.monotonic()
        self.command_stamp = stamp_seconds(msg.header.stamp)

    def receive_udp(self):
        """UDP 상태 패킷을 수신하고 JSON으로 해석한다."""
        latest = None

        for _ in range(32):
            try:
                packet, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break

            try:
                data = json.loads(packet)
                age = time.monotonic() - float(data["monotonic"])

                if (
                    data["version"] == 1
                    and 0 <= age <= self.gate.sensor_timeout
                ):
                    latest = data

            except (ValueError, TypeError, KeyError):
                continue

        if latest is not None:
            self.publish_udp(latest)

    def publish_udp(self, data):
        """UDP로 받은 상태를 ROS 메시지와 TF로 변환하여 발행한다."""
        try:
            position = data["position"]
            quaternion = data["quaternion_wxyz"]
            linear = data["linear_velocity_body"]
            angular = data["angular_velocity_body"]

            if any(
                len(v) != n
                for v, n in (
                    (position, 3),
                    (quaternion, 4),
                    (linear, 3),
                    (angular, 3),
                )
            ):
                return

            if not all(
                math.isfinite(v)
                for values in (position, quaternion, linear, angular)
                for v in values
            ):
                return

            pose = Pose(*position, yaw_from_wxyz(quaternion))
            points = data["points"]

            if len(points) > 2000 or any(
                len(p) != 3
                or not all(math.isfinite(v) for v in p)
                for p in points
            ):
                return

        except (ValueError, TypeError, KeyError):
            return

        session = (
            data.get("session"),
            data.get("reset_counter", 0),
        )

        if (
            self.simulation_session is not None
            and session != self.simulation_session
        ):
            self.gate.goal = None
            self.gate.enabled = False
            self.path_time = self.command_time = -math.inf

            self.get_logger().info(
                "Locomotion session changed: set a new waypoint to resume"
            )

        self.simulation_session = session
        now = self.get_clock().now().to_msg()

        # UDP 상태 → ROS Odometry
        odom = Odometry()
        odom.header = Header(stamp=now, frame_id="map")
        odom.child_frame_id = "vehicle"

        (
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ) = position

        q = odom.pose.pose.orientation
        q.w, q.x, q.y, q.z = quaternion

        v = odom.twist.twist.linear
        w = odom.twist.twist.angular
        v.x, v.y, v.z = linear
        w.x, w.y, w.z = angular

        self.odom_pub.publish(odom)

        # UDP 위치·자세 → ROS TF
        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = odom.child_frame_id

        t = transform.transform.translation
        t.x, t.y, t.z = position
        transform.transform.rotation = q

        self.transforms.sendTransform(transform)

        # UDP 점 목록 → ROS PointCloud2
        fields = [
            PointField(
                name=name,
                offset=index * 4,
                datatype=PointField.FLOAT32,
                count=1,
            )
            for index, name in enumerate(
                ("x", "y", "z", "intensity")
            )
        ]

        cloud = point_cloud2.create_cloud(
            odom.header,
            fields,
            [(*point, 1.0) for point in points],
        )

        self.scan_pub.publish(cloud)

        self.pose = pose
        self.pose_time = self.cloud_time = time.monotonic()

    def send_command(self):
        """ROS에서 받은 명령을 검사한 뒤 UDP로 전송한다."""
        now = time.monotonic()

        reason = self.gate.reason(
            now,
            self.pose,
            self.pose_time,
            self.cloud_time,
            self.path_time,
            self.command_time,
            self.path_stamp,
            self.command_stamp,
        )

        command = (
            self.command
            if reason == "following waypoint"
            else Velocity()
        )

        session = (
            self.simulation_session[0]
            if self.simulation_session
            else None
        )

        # 속도 명령 → UDP 패킷 전송
        self.socket.sendto(
            command_packet(
                command,
                self.sequence,
                now,
                session,
            ),
            self.command_address,
        )
        self.sequence += 1

        # 전송한 명령을 ROS 토픽에도 발행
        message = TwistStamped()
        message.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id="vehicle",
        )
        message.twist.linear.x = command.vx
        message.twist.linear.y = command.vy
        message.twist.angular.z = command.wz

        self.velocity_pub.publish(message)

        if reason != self.last_reason:
            self.status_pub.publish(String(data=reason))
            self.get_logger().info(reason)
            self.last_reason = reason

    def close(self):
        session = (
            self.simulation_session[0]
            if self.simulation_session
            else None
        )

        self.socket.sendto(
            command_packet(
                Velocity(),
                self.sequence,
                time.monotonic(),
                session,
            ),
            self.command_address,
        )
        self.socket.close()


def main(args=None):
    rclpy.init(args=args)
    node = NavigationBridge()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()