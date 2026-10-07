import json
import math
import unittest

from go2_autonomous_driving.core import (
    CommandWatchdog, NavigationGate, Pose, Velocity, WaypointIntent, command_packet,
    relative_waypoint, yaw_from_wxyz,
    decode_keys,
)


class WaypointTests(unittest.TestCase):
    def test_continuous_goal_moves_beyond_the_first_metre(self):
        intent = WaypointIntent(1)
        self.assertEqual(intent.start(Pose(0, 0, 0.3, 0), 1, 0), (1, 0))
        self.assertEqual(intent.target(Pose(1.5, 0, 0.3, 0)), (2.5, 0))

    def test_selected_world_direction_survives_robot_turning(self):
        intent = WaypointIntent(1)
        intent.start(Pose(0, 0, 0.3, 0), 0, 1)
        # A left waypoint must not keep turning farther left as the robot turns.
        self.assertEqual(intent.target(Pose(1, 2, 0.3, math.pi / 2)), (1, 3))
        intent.start(Pose(1, 2, 0.3, math.pi / 2), 0, 1)
        target = intent.target(Pose(1, 2, 0.3, 0))
        self.assertAlmostEqual(target[0], 0)
        self.assertAlmostEqual(target[1], 2)

    def test_stop_clears_continuous_direction(self):
        intent = WaypointIntent()
        pose = Pose(0, 0, 0.3, 0)
        intent.start(pose, 1, 0)
        intent.stop()
        self.assertIsNone(intent.target(pose))

    def test_forward_uses_current_heading(self):
        point = relative_waypoint(Pose(2, 3, 0.3, math.pi / 2), 1, 0)
        self.assertAlmostEqual(point[0], 2)
        self.assertAlmostEqual(point[1], 4)

    def test_left_is_body_relative(self):
        point = relative_waypoint(Pose(2, 3, 0.3, math.pi / 2), 0, 1)
        self.assertAlmostEqual(point[0], 1)
        self.assertAlmostEqual(point[1], 3)

    def test_quaternion_order_is_wxyz(self):
        yaw = yaw_from_wxyz([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
        self.assertAlmostEqual(yaw, math.pi / 2)
        with self.assertRaises(ValueError):
            yaw_from_wxyz([0, 0, 0, 0])

    def test_fragmented_arrow_does_not_become_a_left_command(self):
        keys, pending = decode_keys("\x1b[")
        self.assertEqual(keys, [])
        keys, pending = decode_keys(pending + "A")
        self.assertEqual(keys, ["w"])
        self.assertEqual(pending, "")
        self.assertEqual(decode_keys("\x1b[B\x1b[C\x1b[DW ")[0], ["s", "d", "a", "w", " "])


class CommandTests(unittest.TestCase):
    def test_caps_and_timeout(self):
        watchdog = CommandWatchdog()
        self.assertTrue(watchdog.receive(command_packet(Velocity(9, -9, 9), 0, 10), 10))
        self.assertEqual(watchdog.sample(10.1), Velocity(0.5, -0.2, 0.5))
        self.assertEqual(watchdog.sample(10.31), Velocity())

    def test_old_and_replayed_packets_are_rejected(self):
        watchdog = CommandWatchdog()
        packet = command_packet(Velocity(0.1), 1, 10)
        self.assertTrue(watchdog.receive(packet, 10))
        self.assertFalse(watchdog.receive(packet, 10.1))
        self.assertFalse(watchdog.receive(command_packet(Velocity(0.2), 2, 9), 10))
        self.assertFalse(watchdog.receive(command_packet(Velocity(0.2), 3, 11), 10))

    def test_sender_can_restart_after_timeout(self):
        watchdog = CommandWatchdog()
        watchdog.receive(command_packet(Velocity(0.1), 100, 10), 10)
        self.assertTrue(watchdog.receive(command_packet(Velocity(0.2), 0, 11), 11))

    def test_malformed_and_nan_packets_do_not_replace_command(self):
        watchdog = CommandWatchdog()
        watchdog.receive(command_packet(Velocity(0.1), 1, 10), 10)
        for velocity in ([math.nan, 0, 0], [1, 2], [1, 2, 3, 4]):
            packet = json.dumps({"version": 1, "sequence": 2, "monotonic": 10,
                                 "velocity": velocity}).encode()
            self.assertFalse(watchdog.receive(packet, 10))
        self.assertFalse(watchdog.receive(b"invalid", 10))
        self.assertEqual(watchdog.sample(10), Velocity(0.1))


class NavigationTests(unittest.TestCase):
    def ready(self):
        gate = NavigationGate()
        gate.set_goal(1, 0, 100)
        gate.lease(True, 10)
        return gate

    def reason(self, gate, **overrides):
        values = dict(now=10, pose=Pose(0, 0, 0.3, 0), pose_time=10, cloud_time=10,
                      path_time=10, command_time=10, path_stamp=100, command_stamp=100)
        values.update(overrides)
        return gate.reason(**values)

    def test_new_goal_requires_a_new_plan(self):
        gate = self.ready()
        self.assertEqual(self.reason(gate, path_stamp=99), "waiting for planned path")
        self.assertEqual(self.reason(gate, command_stamp=99), "waiting for planner velocity")
        self.assertEqual(self.reason(gate), "following waypoint")

    def test_continuation_does_not_interrupt_current_fresh_plan(self):
        gate = self.ready()
        self.assertTrue(gate.continue_goal(2, 0, 101))
        self.assertEqual(self.reason(gate), "following waypoint")
        self.assertEqual(gate.goal, (2, 0))
        self.assertEqual(self.reason(gate, path_time=8), "waiting for planned path")
        gate.set_goal(-1, 0, 102)
        self.assertEqual(self.reason(gate), "waiting for planned path")
        self.assertFalse(gate.continue_goal(3, 0, 101))
        self.assertEqual(gate.goal, (-1, 0))

    def test_continuation_cannot_restart_after_stop_reset_or_arrival(self):
        gate = self.ready()
        gate.lease(False, 10)
        self.assertFalse(gate.continue_goal(2, 0, 101))
        gate.lease(True, 10)
        gate.goal = None  # Simulation reset clears the active intent.
        self.assertFalse(gate.continue_goal(2, 0, 101))
        gate.set_goal(0.1, 0, 101)
        self.assertEqual(self.reason(gate), "waypoint reached")
        self.assertFalse(gate.continue_goal(2, 0, 102))

    def test_stale_pose_scan_path_or_command_stops(self):
        gate = self.ready()
        for key in ("pose_time", "cloud_time", "path_time", "command_time"):
            self.assertNotEqual(self.reason(gate, **{key: 8}), "following waypoint")
        self.assertEqual(self.reason(gate, now=11), "stopped")

    def test_reaching_goal_latches_until_next_waypoint(self):
        gate = self.ready()
        self.assertEqual(self.reason(gate, pose=Pose(0.9, 0, 0.3, 0)), "waypoint reached")
        self.assertEqual(self.reason(gate), "waypoint reached")
        gate.set_goal(2, 0, 100)
        self.assertEqual(self.reason(gate), "following waypoint")

    def test_stop_overrides_a_fresh_planner_command(self):
        gate = self.ready()
        gate.lease(False, 10)
        self.assertEqual(self.reason(gate), "stopped")


if __name__ == "__main__":
    unittest.main()
