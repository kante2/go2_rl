"""키보드 방향을 유지하며 웨이포인트를 갱신하는 함수 기반 ROS 노드.

실행 흐름: main() → init_keyboard() → run_keyboard() → close().
ROS 노드와 이동 상태는 state 딕셔너리에 보관하고 함수에 전달한다.
"""

from functools import partial
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
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Int8

from ..core import yaw_from_wxyz


def init_keyboard(*, parameter_overrides=None):
    """기본 ROS 노드, 설정, 상태, 발행자와 콜백을 준비한다."""
    node = rclpy.create_node("keyboard_waypoint", parameter_overrides=parameter_overrides)
    try:
        distance = float(node.declare_parameter("waypoint_distance", 1.0).value)
        timeout = float(node.declare_parameter("pose_timeout", 0.5).value)
        continuous = bool(node.declare_parameter("continuous", True).value)
        update_period = float(node.declare_parameter("waypoint_update_period", 0.2).value)
        if any(not math.isfinite(v) or v <= 0 for v in (distance, timeout, update_period)):
            raise ValueError("waypoint_distance, pose_timeout, waypoint_update_period must be positive and finite")

        state = {
            "node": node,
            "distance": distance,
            "pose_timeout": timeout,
            "continuous": continuous,
            "pose": None,
            "pose_time": -math.inf,
            "frame": "map",
            "direction": None,
            "intent_frame": None,
            "enabled": False,
        }
        # 새 목표와 연속 이동 중의 목표 갱신을 별도 토픽으로 전달한다.
        state["goal_pub"] = node.create_publisher(PointStamped, "/way_point", 10)
        state["update_pub"] = node.create_publisher(PointStamped, "/go2/waypoint_update", 10)
        state["enable_pub"] = node.create_publisher(Bool, "/go2/navigation_enabled", 10)
        state["stop_pub"] = node.create_publisher(Int8, "/stop", 10)

        node.create_subscription(Odometry, "/state_estimation", partial(CB_pose, state), 10)
        node.create_timer(0.1, partial(CB_heartbeat, state))
        node.create_timer(update_period, partial(CB_refresh_waypoint, state))
        return state
    except BaseException:
        node.destroy_node()
        raise


# 위치 수신과 이동 가능 상태 검사
def CB_pose(state, msg):
    """위치·자세를 검사하고 최근 위치와 수신 시간을 저장한다."""
    q, p = msg.pose.pose.orientation, msg.pose.pose.position
    try:
        yaw = yaw_from_wxyz((q.w, q.x, q.y, q.z))
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z)) or not msg.header.frame_id:
            return
    except (ValueError, TypeError, OverflowError):
        return
    state["pose"] = {"x": p.x, "y": p.y, "z": p.z, "yaw": yaw}
    state["pose_time"] = time.monotonic()
    state["frame"] = msg.header.frame_id


def pose_is_current(state, *, check_frame=True):
    """최신 위치가 있으며 이동을 시작한 좌표계가 유지되는지 확인한다."""
    return (
        state["pose"] is not None
        and 0 <= time.monotonic() - state["pose_time"] <= state["pose_timeout"]
        and (not check_frame or state["intent_frame"] is None
             or state["frame"] == state["intent_frame"])
    )


def CB_heartbeat(state):
    """이동 활성 상태를 주기적으로 발행하고 위치를 잃으면 정지한다."""
    if state["enabled"] and not pose_is_current(state):
        stop_motion(state)
        state["node"].get_logger().warning(
            "위치 수신이 끊기거나 좌표계가 변경됐습니다. 복구 후 방향 키를 다시 누르세요."
        )
        return
    state["enable_pub"].publish(Bool(data=state["enabled"]))


# 방향 선택, 웨이포인트 발행, 정지
def set_direction(state, forward, left):
    """키 입력 순간의 몸체 방향을 월드 좌표계의 고정 방향으로 바꾼다."""
    length = math.hypot(forward, left)
    if not math.isfinite(length) or length == 0:
        raise ValueError("A finite, nonzero direction is required")
    yaw = state["pose"]["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    state["direction"] = (
        (c * forward - s * left) / length,
        (s * forward + c * left) / length,
    )
    state["intent_frame"] = state["frame"]


def publish_goal(state, publisher):
    """현재 위치에서 선택한 방향으로 일정 거리 앞의 목표를 발행한다."""
    pose = state["pose"]
    dx, dy = state["direction"]
    goal = PointStamped()
    goal.header.stamp = state["node"].get_clock().now().to_msg()
    goal.header.frame_id = state["frame"]
    goal.point.x = pose["x"] + state["distance"] * dx
    goal.point.y = pose["y"] + state["distance"] * dy
    goal.point.z = pose["z"]
    publisher.publish(goal)
    return goal.point.x, goal.point.y


def CB_refresh_waypoint(state):
    """연속 이동 중에는 선택한 방향을 유지하며 목표 위치만 갱신한다."""
    if state["continuous"] and state["enabled"]:
        if not pose_is_current(state):
            stop_motion(state)
            return
        publish_goal(state, state["update_pub"])


def stop_motion(state):
    """이동 방향을 해제하고 브리지와 경로 추종기에 정지를 전달한다."""
    state["enabled"] = False
    state["direction"] = None
    state["intent_frame"] = None
    state["enable_pub"].publish(Bool(data=False))
    state["stop_pub"].publish(Int8(data=2))


def handle_key(state, key):
    """입력 키를 처리한다. Q 또는 Ctrl+C이면 입력 반복 종료를 요청한다."""
    if key in (" ", "q", "\x03"):
        stop_motion(state)
        return key not in ("q", "\x03")

    directions = {"w": (1, 0), "s": (-1, 0), "a": (0, 1), "d": (0, -1)}
    if key not in directions:
        return True
    # 새 방향 키는 현재 좌표계에서 이동을 다시 시작할 수 있다.
    if not pose_is_current(state, check_frame=False):
        state["node"].get_logger().warning("최신 위치를 수신한 뒤 방향 키를 누르세요.")
        return True

    set_direction(state, *directions[key])
    x, y = publish_goal(state, state["goal_pub"])
    state["stop_pub"].publish(Int8(data=0))
    state["enabled"] = True
    CB_heartbeat(state)
    mode = "연속 이동" if state["continuous"] else "단일 웨이포인트"
    state["node"].get_logger().info(
        f"{mode}: {state['frame']} ({x:.2f}, {y:.2f}); Space로 정지"
    )
    return True


# 터미널 입력: 방향키의 여러 바이트가 나뉘어 도착해도 하나의 키로 해석한다.
def decode_keys(buffer):
    """W/S/A/D와 방향키를 해석하고 미완성 입력은 다음 읽기로 넘긴다."""
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
            if key in ("w", "a", "s", "d", "q", " ", "\x03"):
                keys.append(key)
    return keys, buffer


def run_keyboard(state):
    """ROS 콜백과 키 입력을 처리하고, 종료 시 터미널 설정을 복원한다."""
    node = state["node"]
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    mode = "Space를 누를 때까지 연속 이동" if state["continuous"] else "단일 웨이포인트"
    print(
        f"모드: {mode}\n"
        "W/↑: 전진, S/↓: 후진, A/←: 왼쪽, D/→: 오른쪽, Space: 정지, Q: 종료",
        flush=True,
    )
    buffer = ""
    try:
        tty.setcbreak(fd)
        while rclpy.ok(context=node.context):
            rclpy.spin_once(node, timeout_sec=0.02)
            if not select.select([fd], [], [], 0)[0]:
                continue
            chunk = os.read(fd, 32)
            if not chunk:
                stop_motion(state)
                break
            keys, buffer = decode_keys(buffer + chunk.decode(errors="ignore"))
            for key in keys:
                if not handle_key(state, key):
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def close(state):
    """정지 신호를 발행한 뒤 ROS 노드를 정리한다."""
    if state is None:
        return
    node = state["node"]
    try:
        if rclpy.ok(context=node.context):
            stop_motion(state)
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()


def main(args=None):
    """초기화 → 키보드 실행 → 종료 정리 순서로 호출한다."""
    if not sys.stdin.isatty():
        raise SystemExit("keyboard_waypoint는 키 입력이 가능한 터미널에서 실행하세요.")
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    state = None
    try:
        state = init_keyboard()
        run_keyboard(state)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            close(state)
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
