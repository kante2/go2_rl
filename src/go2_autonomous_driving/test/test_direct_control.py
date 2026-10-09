import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from go2_autonomous_driving.core import Velocity
from go2_autonomous_driving.real.direct_control import (DirectCommandGate, KeyboardVelocityIntent,
                                                   MeasuredStateReceiver, decode_velocity_keys,
                                                   velocity_packet)


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / 'config/real_closed_loop.json').read_text())


def frame(stamp=100., session='backend', **changes):
    data = dict(version=1, kind='go2_measured_state', monotonic=stamp, session=session, sequence=0,
                checkpoint_sha256='checkpoint', q_policy=list(range(12)), dq_policy=[0.] * 12,
                gyro=[0.] * 3, quaternion_wxyz=[1., 0, 0, 0], command=[0.] * 3,
                ready=True, execute=False, phase='DRY READY')
    data.update(changes)
    return json.dumps(data).encode()


def test_direct_timeout_requires_new_key_even_when_packets_resume(config):
    gate = DirectCommandGate(config)
    gate.set_session('backend')
    def packet(seq, stamp, intent='w', session='backend'):
        return velocity_packet(Velocity(.2, 0, 0), seq, stamp, session, intent)
    assert not gate.receive(packet(0, 100, session='old-process'), 100)
    assert gate.receive(packet(0, 100), 100)
    assert gate.sample(100.1).vx == .2
    assert gate.sample(100.4) == Velocity()
    assert not gate.receive(packet(1, 100.41), 100.41)
    assert gate.sample(100.41) == Velocity()
    assert gate.receive(packet(2, 100.42, intent='new-key'), 100.42)
    assert gate.sample(100.42).vx == .2
    gate.set_session('new-policy-session')
    assert gate.sample(100.43) == Velocity()
    assert not gate.receive(packet(3, 100.44), 100.44)


def test_replay_and_waypoint_packets_do_not_drive_direct_control(config):
    gate = DirectCommandGate(config)
    gate.set_session('backend')
    payload = velocity_packet(Velocity(.2, 0, 0), 0, 100, 'backend', 'key')
    assert gate.receive(payload, 100)
    assert not gate.receive(payload, 100.1)
    assert not gate.receive(b'{"version": 1}', 100.2)
    assert gate.sample(100.4) == Velocity()


def test_keyboard_requires_ready_and_clears_direction_after_state_loss(config):
    key = KeyboardVelocityIntent(config)
    assert not key.handle_key('w', 100)
    assert key.state.receive(frame(ready=False), 100)
    assert not key.handle_key('w', 100)
    assert key.state.receive(frame(100.01, sequence=1), 100.01)
    assert key.handle_key('w', 100.01)
    assert key.command.vx == .2
    key.sync(100.4)
    assert key.command == Velocity()
    key.state.receive(frame(100.41, sequence=2), 100.41)
    key.sync(100.41)
    assert key.command == Velocity()
    assert key.handle_key('a', 100.42)
    assert key.command.vy == .12
    key.state.receive(frame(100.43, session='rearmed'), 100.43)
    key.sync(100.43)
    assert key.command == Velocity()


def test_measured_state_rejects_stale_wrong_model_nonfinite_and_replay():
    rx = MeasuredStateReceiver('checkpoint')
    assert not rx.receive(frame(), 101)
    assert not rx.receive(frame(checkpoint_sha256='wrong'), 100)
    assert not rx.receive(frame(q_policy=[float('nan')] * 12), 100)
    assert not rx.receive(frame(quaternion_wxyz=[0.] * 4), 100)
    assert rx.receive(frame(), 100.01)
    assert not rx.receive(frame(), 100.02)
    assert not rx.fresh(100.4)


def test_arrows_and_yaw_keys_are_distinct():
    assert decode_velocity_keys('\x1b[A\x1b[Djl q') == (['w', 'a', 'j', 'l', ' ', 'q'], '')
    assert decode_velocity_keys('\x1b[') == ([], '\x1b[')


def test_filtered_action_feedback_and_tracking_fault_record(config, tmp_path):
    pytest.importorskip('rclpy')
    pytest.importorskip('unitree_go')
    from go2_autonomous_driving.real.real_closed_loop import ClosedLoopBackend
    from go2_autonomous_driving.real.real_control import PolicyMapping
    project = Path(__file__).resolve().parents[3]
    mapping = PolicyMapping(json.loads((project / 'artifacts/go2_real/policy.json').read_text()))
    fault_path = tmp_path / 'tracking_fault.json'
    config.update(command_port=0, fault_log=str(fault_path))
    robot = SimpleNamespace(config=config, mapping=mapping)
    backend = ClosedLoopBackend(robot, None, None)
    q = mapping.to_sdk(mapping.default)
    obs = np.zeros((1, 47), dtype=np.float32)
    try:
        # A large but in-range raw target cannot jump directly to the actuators.
        previous, target = backend.policy_targets(np.full((1, 12), 3.2), obs, q, 100., True)
        np.testing.assert_allclose(target - q, .24, atol=1e-6)
        np.testing.assert_allclose(mapping.targets(previous)[1], target, atol=1e-6)
        actual_obs = mapping.observation(q, np.zeros(12), [1, 0, 0, 0], [0, 0, 0], [0, 0, 0], 1, previous)
        np.testing.assert_allclose(actual_obs[0, 35:47], .96, atol=1e-6)
        # A stuck joint must still trigger the original 0.65 rad limit.
        backend.policy_targets(np.full((1, 12), 3.2), obs, q, 100.02, False)
        _, target = backend.policy_targets(np.full((1, 12), 3.2), obs, q, 100.04, False)
        with pytest.raises(RuntimeError, match='joint=.*error=.*limit=0.650rad') as error:
            backend.check_tracking(target, q)
        backend.fault = str(error.value)
        # Invalid raw targets remain fatal, regardless of filtering.
        with pytest.raises(ValueError, match='joint limits'):
            backend.policy_targets(np.full((1, 12), 1000), obs, q, 100.06, False)
    finally:
        backend.close()
    recorded = json.loads(fault_path.read_text())
    assert recorded['tracking_failure'] is not None
    assert len(recorded['recent_policy_frames']) == 3
    assert 'requested_cmd=' in recorded['fault']


@pytest.mark.parametrize('quietpaw', [False, True])
def test_policy_reads_actual_joint_and_imu_feedback_without_navigation(config, quietpaw):
    pytest.importorskip('rclpy')
    pytest.importorskip('unitree_go')
    from go2_autonomous_driving.real.real_closed_loop import ClosedLoopBackend
    from go2_autonomous_driving.real.real_control import PolicyMapping, gravity_body
    project = Path(__file__).resolve().parents[3]
    metadata = json.loads((project / 'artifacts/go2_real/policy.json').read_text())
    if quietpaw:
        from go2_autonomous_driving.real.real_quietpaw import QuietPawMapping, QUIETPAW_OBSERVATIONS
        metadata.update(policy_family='quietpaw', observation_size=48,
                        observation_names=QUIETPAW_OBSERVATIONS,
                        noise_condition=dict(min_cost_limit=.02, max_cost_limit=.35))
        mapping = QuietPawMapping(metadata, quietness=.5)
    else:
        mapping = PolicyMapping(metadata)
    config.update(command_port=0)
    q = mapping.to_sdk(mapping.default)
    dq = np.arange(12) * .01
    state = SimpleNamespace(imu_state=SimpleNamespace(quaternion=[1., 0, 0, 0], gyroscope=[.1, .2, .3]))
    robot = SimpleNamespace(config=config, mapping=mapping, state=state, state_time=time.monotonic())
    def checked_state():
        robot.state_time = time.monotonic()
        return state, q.copy(), dq.copy()
    robot.checked_state = checked_state
    observations = []
    def infer(unused, inputs):
        obs = inputs['obs'].copy()
        assert obs.shape == (1, 48 if quietpaw else 47)
        if quietpaw:
            assert obs[0, -1] == pytest.approx(.185)
        observations.append(obs)
        # An observable policy response to sensor changes, within joint limits.
        output = np.zeros((1, 12), dtype=np.float32)
        output[0, 0] = obs[0, 0] + obs[0, 11]
        return [output]
    backend = ClosedLoopBackend(robot, SimpleNamespace(run=infer), 'obs')
    worker = threading.Thread(target=backend.infer_loop)
    try:
        worker.start()
        deadline = time.monotonic() + 1
        while len(observations) < 3 and time.monotonic() < deadline:
            time.sleep(.005)
        assert len(observations) >= 3
        first = backend.target[1].copy()
        q[mapping.order[0]] += .15
        state.imu_state.gyroscope = [.4, -.2, .1]
        state.imu_state.quaternion = [np.cos(.1), np.sin(.1), 0, 0]
        count = len(observations)
        while len(observations) < count + 3 and time.monotonic() < deadline:
            time.sleep(.005)
        assert len(observations) >= count + 3
        obs = observations[-1][0]
        np.testing.assert_allclose(obs[:3], state.imu_state.gyroscope, atol=1e-6)
        np.testing.assert_allclose(obs[3:6], gravity_body(state.imu_state.quaternion), atol=1e-6)
        np.testing.assert_allclose(obs[11:23], q[mapping.order] - mapping.default, atol=1e-6)
        np.testing.assert_allclose(obs[23:35], dq[mapping.order], atol=1e-6)
        assert np.max(np.abs(backend.target[1] - first)) > .05
        assert backend.fault is None
        # Viewer failure is diagnostic only, with no loss of control feedback.
        backend.navigation_fault = 'viewer unavailable'
        assert backend.navigation_ready()
        # rclpy fixed arrays contain np.float32; telemetry must serialize these.
        state.imu_state.quaternion = np.asarray(state.imu_state.quaternion, dtype=np.float32)
        state.imu_state.gyroscope = np.asarray(state.imu_state.gyroscope, dtype=np.float32)
        sent = []
        backend.sender.close()
        backend.sender = SimpleNamespace(sendto=lambda payload, address: sent.append(payload), close=lambda: None)
        backend.telemetry()
        assert len(sent) == 2
        packet = json.loads(sent[0])
        np.testing.assert_allclose(packet['q_policy'], q[mapping.order])
        assert MeasuredStateReceiver().receive(sent[0], time.monotonic())
    finally:
        backend.stop.set()
        worker.join(timeout=2)
        backend.close()


def test_closed_loop_eases_the_trained_walking_to_standing_boundary(config):
    pytest.importorskip('rclpy')
    pytest.importorskip('unitree_go')
    from go2_autonomous_driving.real.real_closed_loop import ClosedLoopBackend
    from go2_autonomous_driving.real.real_control import PolicyMapping
    project = Path(__file__).resolve().parents[3]
    mapping = PolicyMapping(json.loads((project / 'artifacts/go2_real/policy.json').read_text()))
    config = dict(config, command_port=0)
    backend = ClosedLoopBackend(SimpleNamespace(config=config, mapping=mapping), None, None)
    q = mapping.to_sdk(mapping.default)
    obs = mapping.observation(q, np.zeros(12), [1, 0, 0, 0], [0, 0, 0], [.1, 0, 0], 1, np.zeros(12))
    try:
        previous, walking = backend.policy_targets(np.full((1, 12), 3.2), obs, q, 100., True)
        standing_obs = mapping.observation(q, np.zeros(12), [1, 0, 0, 0], [0, 0, 0], [.08, 0, 0], 2, previous)
        np.testing.assert_allclose(standing_obs[0, 9:11], 0)
        previous, standing = backend.policy_targets(np.full((1, 12), -3.2), standing_obs, q, 100.02, False)
        np.testing.assert_allclose(standing - walking, -.06, atol=1e-6)
        np.testing.assert_allclose(mapping.targets(previous)[1], standing, atol=1e-6)
    finally:
        backend.close()
