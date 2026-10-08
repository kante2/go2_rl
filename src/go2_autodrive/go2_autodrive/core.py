"""브리지와 시뮬레이션이 공유하는 상태·속도·UDP 검사·주행 허용 조건."""

from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class Velocity:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def bounded(self, linear: float, lateral: float, angular: float):
        if not all(math.isfinite(v) for v in (self.vx, self.vy, self.wz)):
            raise ValueError("Velocity must be finite")
        if any(not math.isfinite(v) or v <= 0 for v in (linear, lateral, angular)):
            raise ValueError("Velocity limits must be positive and finite")
        return Velocity(
            max(-linear, min(linear, self.vx)),
            max(-lateral, min(lateral, self.vy)),
            max(-angular, min(angular, self.wz)),
        )


def yaw_from_wxyz(quaternion):
    if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion):
        raise ValueError("Invalid quaternion")
    length = math.sqrt(sum(v * v for v in quaternion))
    if length < 1e-8:
        raise ValueError("Zero quaternion")
    w, x, y, z = (v / length for v in quaternion)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def command_packet(command: Velocity, sequence: int, now: float, session=None):
    if sequence < 0 or not math.isfinite(now):
        raise ValueError("Invalid command sequence or timestamp")
    return json.dumps({
        "version": 1, "sequence": sequence, "monotonic": now,
        "velocity": [command.vx, command.vy, command.wz],
        "session": session,
    }, allow_nan=False).encode()


class CommandWatchdog:
    """Only accept fresh, ordered body-velocity packets from the local ROS bridge."""

    def __init__(self, timeout=0.3, linear=0.5, lateral=0.2, angular=0.5):
        self.timeout = timeout
        self.limits = (linear, lateral, angular)
        self.command = Velocity()
        self.stamp = -math.inf
        self.sequence = -1

    def receive(self, packet, now):
        try:
            data = json.loads(packet)
            stamp = float(data["monotonic"])
            sequence = data["sequence"]
            if data["version"] != 1 or type(sequence) is not int or sequence < 0:
                return False
            if not isinstance(data["velocity"], list) or len(data["velocity"]) != 3:
                return False
            if not math.isfinite(stamp) or not -0.05 <= now - stamp <= self.timeout:
                return False
            # A restarted sender may reset its sequence only after the old lease expires.
            if now - self.stamp <= self.timeout and (sequence <= self.sequence or stamp < self.stamp):
                return False
            command = Velocity(*data["velocity"]).bounded(*self.limits)
        except (ValueError, TypeError, KeyError, OverflowError):
            return False
        self.command, self.stamp, self.sequence = command, stamp, sequence
        return True

    def sample(self, now):
        return self.command if 0 <= now - self.stamp <= self.timeout else Velocity()


class NavigationGate:
    """A waypoint and current planner/sensor data are required before walking."""

    def __init__(self, tolerance=0.2, sensor_timeout=0.5, command_timeout=0.3):
        self.tolerance = tolerance
        self.sensor_timeout = sensor_timeout
        self.command_timeout = command_timeout
        self.goal = None
        self.goal_stamp = -math.inf
        self.latest_goal_stamp = -math.inf
        self.finished = False
        self.enabled = False
        self.enable_time = -math.inf

    def set_goal(self, x, y, stamp):
        if not all(math.isfinite(v) for v in (x, y, stamp)):
            raise ValueError("Invalid goal")
        self.goal = (x, y)
        self.goal_stamp = stamp
        self.latest_goal_stamp = stamp
        self.finished = False

    def continue_goal(self, x, y, stamp):
        """Extend an active intent without invalidating its current fresh plan."""
        if not all(math.isfinite(v) for v in (x, y, stamp)):
            raise ValueError("Invalid goal update")
        if (self.goal is None or not self.enabled or self.finished
                or stamp <= self.latest_goal_stamp):
            return False
        self.goal = (x, y)
        self.latest_goal_stamp = stamp
        return True

    def lease(self, enabled, now):
        self.enabled, self.enable_time = bool(enabled), now

    def reason(self, now, pose, pose_time, cloud_time, path_time, command_time,
               path_stamp, command_stamp):
        if not self.enabled or now - self.enable_time > self.sensor_timeout:
            return "stopped"
        if self.goal is None:
            return "waiting for waypoint"
        if pose is None or now - pose_time > self.sensor_timeout:
            return "waiting for localization"
        if math.hypot(pose.x - self.goal[0], pose.y - self.goal[1]) <= self.tolerance:
            self.finished = True
        if self.finished:
            return "waypoint reached"
        if now - cloud_time > self.sensor_timeout:
            return "waiting for obstacle scan"
        if now - path_time > self.sensor_timeout or path_stamp < self.goal_stamp:
            return "waiting for planned path"
        if now - command_time > self.command_timeout or command_stamp < self.goal_stamp:
            return "waiting for planner velocity"
        return "following waypoint"
