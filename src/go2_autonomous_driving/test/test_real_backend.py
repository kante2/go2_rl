"""Thread scheduling regressions; no ROS nodes or motor publishers are created."""
import hashlib
import json
import copy
from collections import deque
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('unitree_go')
from go2_autonomous_driving.real.real_robot import Backend, Robot, MotorCommandBuffer, spin_until_stopped, stop_executor
from go2_autonomous_driving.real.real_control import pack_low_command, wire_crc


@pytest.mark.parametrize('expected,actual', [(48, 47), (47, 48)])
def test_wrong_onnx_input_size_is_rejected_before_ros_init(tmp_path, monkeypatch, expected, actual):
    from go2_autonomous_driving.real import real_robot
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'policy.onnx').write_bytes(b'fake-onnx-for-shape-check')
    (tmp_path / 'model.pt').write_bytes(b'fake-checkpoint')
    (tmp_path / 'policy.json').write_text(json.dumps(dict(
        onnx_sha256=hashlib.sha256((tmp_path / 'policy.onnx').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((tmp_path / 'model.pt').read_bytes()).hexdigest())))
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(shape=[1, actual])],
                              get_outputs=lambda: [SimpleNamespace(shape=[1, 12])])
    monkeypatch.setattr(real_robot.ort, 'InferenceSession', lambda *a, **k: session)
    def forbidden(*a, **k):
        pytest.fail('ROS must not initialize when policy shape is incompatible')
    monkeypatch.setattr(real_robot.rclpy, 'init', forbidden)
    args = SimpleNamespace(config=tmp_path / 'config.json', policy_dir=tmp_path,
                           checkpoint=tmp_path / 'model.pt', execute=True)
    with pytest.raises(RuntimeError, match=f'Expected {expected} observations'):
        real_robot.run(args, requires_navigation=False,
                       mapping_factory=lambda meta: SimpleNamespace(observation_size=expected))


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


def test_executor_handles_are_destroyed_only_after_spin_exits():
    entered, wake, stop = threading.Event(), threading.Event(), threading.Event()
    calls = []

    class Executor:
        def spin_once(self, timeout_sec):
            entered.set()
            wake.wait(timeout_sec)
            calls.append('callback finished')

        def wake(self):
            wake.set()

        def shutdown(self, timeout_sec):
            assert not spin.is_alive()
            calls.append('shutdown')

    executor = Executor()
    spin = threading.Thread(target=spin_until_stopped, args=(executor, stop))
    spin.start()
    assert entered.wait(timeout=1)
    stop_executor(executor, spin, stop)
    assert calls[-1] == 'shutdown' and 'callback finished' in calls


def fake_motor_robot(publish):
    kp, kd = np.array([20, 20, 40] * 4), np.array([1, 1, 2] * 4)
    return SimpleNamespace(policy_kp=kp, policy_kd=kd,
                           motor_buffers={d: [MotorCommandBuffer(kp, kd, d) for _ in range(2)]
                                          for d in (False, True)}, motor_message_index=0,
                           peak_publish_timing=None, own_packets=deque(maxlen=128),
                           motor_publisher=SimpleNamespace(publish=publish))


def test_reused_motor_packets_match_original_fields_and_damping():
    from unitree_go.msg import LowCmd
    sent = []
    robot = fake_motor_robot(lambda msg: sent.append(copy.deepcopy(msg)))
    for index in range(6):
        target = np.tile([0., .7, -1.4], 4) + .01 * index
        damping = index % 2 == 1
        Robot.publish_motors(robot, target, damping=damping)
        # Independent construction of the original command contract.
        expected = LowCmd()
        expected.head, expected.level_flag = [0xFE, 0xEF], 0xFF
        for j, motor in enumerate(expected.motor_cmd):
            motor.mode = 1
            motor.q, motor.dq = 2.146e9, 16000.
            if j < 12:
                motor.q = float(target[j]) if not damping else 2.146e9
                motor.dq = 0.
                motor.kp = float(robot.policy_kp[j]) if not damping else 0.
                motor.kd = float(robot.policy_kd[j]) if not damping else 2.
        expected.crc = wire_crc(pack_low_command(expected))
        assert pack_low_command(sent[-1]) == pack_low_command(expected)
        assert robot.own_packets[-1] == pack_low_command(expected)


def test_cached_wire_matches_full_pack_for_random_float32_and_float64_targets():
    rng = np.random.default_rng(812)
    kp, kd = np.array([20., 20., 40.] * 4), np.array([1., 1., 2.] * 4)
    for damping in (False, True):
        buffer = MotorCommandBuffer(kp, kd, damping)
        for dtype in (np.float32, np.float64):
            for _ in range(50):
                targets = rng.uniform(-2.7, 1.1, 12).astype(dtype)
                buffer.set_targets(targets)
                packed = buffer.seal()
                assert packed == pack_low_command(buffer.message)
                assert wire_crc(packed) == buffer.message.crc


def test_next_publish_failure_preserves_last_successful_command():
    robot = fake_motor_robot(lambda msg: None)
    target = np.tile([0., .7, -1.4], 4)
    Robot.publish_motors(robot, target)
    previous = robot.last_motor_command[1]
    previous_bytes = pack_low_command(previous)
    def fail(msg):
        raise RuntimeError('injected DDS failure')
    robot.motor_publisher.publish = fail
    with pytest.raises(RuntimeError, match='injected DDS failure'):
        Robot.publish_motors(robot, target + .1)
    assert robot.last_motor_command[1] is previous
    assert pack_low_command(previous) == previous_bytes
    assert robot.last_publish_timing['dds_ms'] >= 0


def test_motor_qos_matches_robot_best_effort_depth_one():
    from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    recorded = []
    robot = SimpleNamespace(create_publisher=lambda *args: recorded.append(args))
    Robot.create_motor_publisher(robot)
    _, topic, qos = recorded[0]
    assert topic == '/lowcmd'
    assert qos.depth == 1 and qos.history == HistoryPolicy.KEEP_LAST
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.durability == DurabilityPolicy.VOLATILE
