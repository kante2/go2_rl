"""Thread scheduling regressions; no ROS nodes or motor publishers are created."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('unitree_go')
from go2_autonomous_driving.real.real_robot import Backend


def make_backend():
    config = dict(command_port=0, telemetry_port=0, command_timeout=.3,
                  max_forward_speed=.25, max_lateral_speed=.15, max_yaw_rate=.35,
                  policy_timeout=.1, sensor_timeout=.3)
    mapping = SimpleNamespace(meta={'step_dt': .02},
                              observation=lambda *unused: np.zeros((1, 47), dtype=np.float32),
                              targets=lambda action: (np.zeros(12), np.zeros(12)))
    state = SimpleNamespace(imu_state=SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0]))
    robot = SimpleNamespace(config=config, mapping=mapping, sensor_fresh=lambda: True,
                            sensor_status=lambda: 'fake fresh sensors')
    def checked_state():
        robot.state_time = time.monotonic()
        return state, np.zeros(12), np.zeros(12)
    robot.checked_state = checked_state
    session = SimpleNamespace(run=lambda *unused: [np.zeros((1, 12))])
    return Backend(robot, session, 'obs')


def test_slow_cloud_work_does_not_block_policy_thread():
    backend = make_backend()
    inference = threading.Thread(target=backend.infer_loop)
    telemetry = threading.Thread(target=backend.telemetry_loop)
    def slow_cloud():
        time.sleep(.15)  # Old infer_loop executed this inside its 20 ms budget.
        backend.telemetry_time = time.monotonic()
    backend.telemetry = slow_cloud
    try:
        inference.start()
        telemetry.start()
        time.sleep(.4)
        assert backend.fault is None
        assert backend.policy_steps >= 10
        backend.check_target()
    finally:
        backend.stop.set()
        inference.join()
        telemetry.join()
        backend.close()


def test_expired_inference_is_never_published_as_a_fresh_target():
    backend = make_backend()
    def slow_policy(*unused):
        time.sleep(.12)
        return [np.zeros((1, 12))]
    backend.session.run = slow_policy
    try:
        backend.infer_loop()
        assert backend.target is None
        assert 'expired before use' in backend.fault
        with pytest.raises(RuntimeError, match='Preflight failed before release'):
            backend.preflight()
    finally:
        backend.close()


def test_navigation_fault_keeps_inference_alive_but_commands_zero():
    backend = make_backend()
    def broken_cloud():
        raise ValueError('invalid cloud')
    backend.telemetry = broken_cloud
    try:
        backend.telemetry_loop()
        assert backend.navigation_fault == 'invalid cloud'
        assert not backend.stop.is_set()
        np.testing.assert_array_equal(backend.command_sample(), [0, 0, 0])
    finally:
        backend.close()
