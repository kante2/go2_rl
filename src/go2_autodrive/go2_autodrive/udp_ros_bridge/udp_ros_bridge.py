"""ROS 플래너와 보행 백엔드를 로컬 UDP로 연결하는 브리지."""

import json
import math
import socket
import time

import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header, String
from tf2_ros import TransformBroadcaster

from ..core import (
    NavigationGate,
    Pose,
    Velocity,
    command_packet,
    yaw_from_wxyz,
)


def stamp_seconds(stamp):
    """ROS 타임스탬프를 초 단위로 변환한다."""
    return stamp.sec + stamp.nanosec * 1e-9


class NavigationBridge(Node):
    def __init__(self):
        super().__init__("navigation_bridge")

        def parameter(name, default):
            return self.declare_parameter(name, default).value

        # --------------------------------------------------
        # 1. 파라미터
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 2. UDP 통신
        # --------------------------------------------------
        # ROS → UDP 명령 전송 목적지
        self.command_address = ("127.0.0.1", command_port)

        # UDP → ROS 상태 수신 소켓
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", udp_port))
        self.socket.setblocking(False)

        # --------------------------------------------------
        # 3. 내부 상태
        # --------------------------------------------------
        self.sequence = 0
        self.pose = None
        self.command = Velocity()

        self.pose_time = -math.inf
        self.cloud_time = -math.inf
        self.path_time = -math.inf
        self.command_time = -math.inf

        self.path_stamp = -math.inf
        self.command_stamp = -math.inf

        self.last_reason = None
        self.simulation_session = None

        # --------------------------------------------------
        # 4. ROS publishers
        # --------------------------------------------------
        self.odom_pub = self.create_publisher(
            Odometry,
            "/state_estimation",
            5,
        )

        self.scan_pub = self.create_publisher(
            PointCloud2,
            "/registered_scan",
            5,
        )

        self.status_pub = self.create_publisher(
            String,
            "/go2/navigation_status",
            10,
        )

        self.velocity_pub = self.create_publisher(
            TwistStamped,
            "/go2/locomotion_command",
            5,
        )

        self.planner_goal_pub = self.create_publisher(
            PointStamped,
            "/go2/planner_way_point",
            10,
        )

        self.transforms = TransformBroadcaster(self)

        # --------------------------------------------------
        # 5. ROS subscribers
        # --------------------------------------------------
        self.create_subscription(
            PointStamped,
            "/way_point",
            self.CB_ros_goal,
            10,
        )

        self.create_subscription(
            PointStamped,
            "/go2/waypoint_update",
            self.CB_ros_goal_update,
            10,
        )

        self.create_subscription(
            Bool,
            "/go2/navigation_enabled",
            self.CB_ros_enable,
            10,
        )

        self.create_subscription(
            Path,
            "/path",
            self.CB_ros_path,
            5,
        )

        self.create_subscription(
            TwistStamped,
            "/cmd_vel",
            self.CB_ros_command,
            5,
        )

        # --------------------------------------------------
        # 6. 타이머 콜백
        # --------------------------------------------------
        # UDP → ROS: 약 100Hz로 상태 수신 확인
        self.create_timer(
            0.01,
            self.receive_udp_state_and_publish_to_ros,
        )

        # ROS → UDP: 약 20Hz로 명령 전송
        self.create_timer(
            0.05,
            self.send_ros_command_to_udp,
        )

    # ======================================================
    # 데이터 검사
    # ======================================================
    @staticmethod
    def is_ros_frame_valid(msg, expected_frame):
        """ROS 메시지가 요구하는 좌표계를 사용하는지 확인한다."""
        return msg.header.frame_id.lstrip("/") == expected_frame

    @staticmethod
    def is_finite_vector(values, expected_length):
        """지정된 길이의 벡터이며 모든 값이 유한한 숫자인지 확인한다."""
        if not isinstance(values, (list, tuple)):
            return False

        if len(values) != expected_length:
            return False

        try:
            return all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in values
            )
        except (ValueError, TypeError, OverflowError):
            return False

    def is_udp_packet_valid(self, data):
        """UDP 패킷의 버전과 유효시간을 확인한다."""
        if not isinstance(data, dict):
            return False

        try:
            age = time.monotonic() - float(data["monotonic"])

            return (
                data["version"] == 1
                and 0 <= age <= self.gate.sensor_timeout
            )
        except (ValueError, TypeError, KeyError, OverflowError):
            return False

    def is_udp_robot_state_valid(self, data):
        """UDP 로봇 상태의 필수 필드, 벡터 형식, 숫자 값을 확인한다."""
        if not isinstance(data, dict):
            return False

        vector_fields = (
            ("position", 3),
            ("quaternion_wxyz", 4),
            ("linear_velocity_body", 3),
            ("angular_velocity_body", 3),
        )

        for field_name, expected_length in vector_fields:
            if not self.is_finite_vector(
                data.get(field_name),
                expected_length,
            ):
                return False

        points = data.get("points")

        if not isinstance(points, (list, tuple)):
            return False

        if len(points) > 2000:
            return False

        return all(
            self.is_finite_vector(point, 3)
            for point in points
        )

    # ======================================================
    # UDP 세션 변경 처리
    # ======================================================
    def handle_udp_session_change(self, data):
        """백엔드 세션이 바뀌면 기존 목표와 이동 활성 상태를 해제한다."""
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

            self.path_time = -math.inf
            self.command_time = -math.inf

            self.get_logger().info(
                "Locomotion session changed: "
                "set a new waypoint to resume"
            )

        self.simulation_session = session

    # ======================================================
    # ROS 수신 콜백
    # ======================================================
    def CB_ros_goal(self, msg):
        """ROS에서 새로운 목표를 받아 gate와 플래너에 전달한다."""
        if not self.is_ros_frame_valid(msg, "map"):
            self.get_logger().warning(
                "Waypoint must use the map frame"
            )
            return

        try:
            self.gate.set_goal(
                msg.point.x,
                msg.point.y,
                stamp_seconds(msg.header.stamp),
            )
        except ValueError:
            self.get_logger().warning(
                "Rejected invalid waypoint"
            )
            return

        self.planner_goal_pub.publish(msg)

    def CB_ros_goal_update(self, msg):
        """ROS에서 목표 갱신을 받아 허용된 경우 플래너에 전달한다."""
        if not self.is_ros_frame_valid(msg, "map"):
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

    def CB_ros_enable(self, msg):
        """ROS에서 이동 활성화 요청을 받아 gate에 반영한다."""
        self.gate.lease(
            msg.data,
            time.monotonic(),
        )

    def CB_ros_path(self, msg):
        """ROS 경로의 형식을 확인하고 수신 시각을 기록한다."""
        if (
            len(msg.poses) > 1
            and self.is_ros_frame_valid(msg, "vehicle")
        ):
            self.path_time = time.monotonic()
            self.path_stamp = stamp_seconds(msg.header.stamp)
        else:
            self.path_time = -math.inf

    def CB_ros_command(self, msg):
        """ROS 속도 명령을 받아 제한을 적용한 뒤 저장한다."""
        if not self.is_ros_frame_valid(msg, "vehicle"):
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

    # ======================================================
    # UDP → ROS
    # ======================================================
    def receive_udp_state_and_publish_to_ros(self):
        """UDP 상태를 수신·해석·검사하고 ROS 변환 함수로 전달한다."""
        latest = None

        # 대기 중인 패킷을 최대 32개 읽는다.
        for _ in range(32):
            try:
                packet, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break

            try:
                data = json.loads(packet)
            except (ValueError, UnicodeDecodeError):
                continue

            if self.is_udp_packet_valid(data):
                latest = data

        # 읽은 순서상 마지막 유효 패킷을 사용한다.
        if latest is not None:
            self.convert_udp_state_to_ros_and_publish(latest)

    def convert_udp_state_to_ros_and_publish(self, data):
        """UDP 상태를 ROS Odometry·TF·PointCloud2로 변환하고 발행한다."""
        if not self.is_udp_robot_state_valid(data):
            return

        position = data["position"]
        quaternion = data["quaternion_wxyz"]
        linear = data["linear_velocity_body"]
        angular = data["angular_velocity_body"]
        points = data["points"]

        # 검증된 숫자를 ROS 실수 필드에 맞게 변환한다.
        position = tuple(float(value) for value in position)
        quaternion = tuple(float(value) for value in quaternion)
        linear = tuple(float(value) for value in linear)
        angular = tuple(float(value) for value in angular)

        try:
            pose = Pose(
                *position,
                yaw_from_wxyz(quaternion),
            )
        except (ValueError, TypeError):
            return

        self.handle_udp_session_change(data)

        now = self.get_clock().now().to_msg()

        # 1. UDP 위치·자세·속도 → ROS Odometry
        odom = Odometry()
        odom.header = Header(
            stamp=now,
            frame_id="map",
        )
        odom.child_frame_id = "vehicle"

        (
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ) = position

        q = odom.pose.pose.orientation
        q.w, q.x, q.y, q.z = quaternion

        v = odom.twist.twist.linear
        v.x, v.y, v.z = linear

        w = odom.twist.twist.angular
        w.x, w.y, w.z = angular

        self.odom_pub.publish(odom)

        # 2. UDP 위치·자세 → ROS TF: map → vehicle
        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = odom.child_frame_id

        translation = transform.transform.translation
        (
            translation.x,
            translation.y,
            translation.z,
        ) = position

        transform.transform.rotation = q

        self.transforms.sendTransform(transform)

        # 3. UDP 점 목록 → ROS PointCloud2
        # points는 백엔드에서 이미 map 좌표계로 보내야 한다.
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

        # 4. 이동 허용 판단에 사용할 내부 상태 갱신
        self.pose = pose
        self.pose_time = self.cloud_time = time.monotonic()

    # ======================================================
    # ROS → UDP
    # ======================================================
    def send_ros_command_to_udp(self):
        """저장된 ROS 명령의 이동 조건을 검사하고 UDP로 전송한다."""
        now = time.monotonic()

        # 이동 가능 여부 판단은 NavigationGate에서 담당한다.
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

        # 1. ROS에서 받은 명령 → UDP 패킷 생성·전송
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

        # 2. 실제 전송한 명령을 ROS 토픽에도 발행
        message = TwistStamped()
        message.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id="vehicle",
        )

        message.twist.linear.x = command.vx
        message.twist.linear.y = command.vy
        message.twist.angular.z = command.wz

        self.velocity_pub.publish(message)

        # 3. 이동 상태가 변경된 경우에만 상태와 로그 출력
        if reason != self.last_reason:
            self.status_pub.publish(
                String(data=reason)
            )
            self.get_logger().info(reason)
            self.last_reason = reason

    # ======================================================
    # 종료 처리
    # ======================================================
    def close(self):
        """UDP로 정지 명령을 보내고 소켓을 닫는다."""
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