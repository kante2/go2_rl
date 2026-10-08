import math
import struct

import numpy as np
import pytest

from go2_autonomous_driving.real.real_control import (PolicyMapping, gravity_body, next_period,
                                                remote_estop, require_independent_navigation, wire_crc)


@pytest.fixture
def mapping():
    return PolicyMapping(dict(
        policy_to_sdk=[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        default_joint_pos=[-.1, .9, -1.8, .1, .9, -1.8] * 2,
        action_scale=.25, joint_limits=[[-1, 1], [-1, 4], [-2.7, -.8]] * 4,
        observation_names=['base_ang_vel', 'projected_gravity', 'command', 'phase',
                           'joint_pos', 'joint_vel', 'actions'],
        step_dt=.02, phase_period=.6, clip_actions=None,
    ))


def test_crc_matches_independent_bitwise_reference():
    rng = np.random.default_rng(47)
    for size in (8, 812, 1180):
        packed = rng.bytes(size)
        expected = 0xFFFFFFFF
        for word in struct.unpack('<' + 'I' * (size // 4 - 1), packed[:-4]):
            expected ^= word
            for _ in range(32):
                expected = ((expected << 1) ^ (0x04C11DB7 if expected & 0x80000000 else 0)) & 0xFFFFFFFF
        assert wire_crc(packed) == expected


def test_observation_uses_sdk_to_policy_order(mapping):
    q = mapping.to_sdk(mapping.default) + np.arange(12, dtype=np.float32) / 100
    obs = mapping.observation(q, np.arange(12), [1, 0, 0, 0], [1, 2, 3], [.25, 0, 0], 0, np.zeros(12))[0]
    np.testing.assert_allclose(obs[:11], [1, 2, 3, 0, 0, -1, .25, 0, 0, 0, 1])
    np.testing.assert_allclose(obs[11:23], np.arange(12)[mapping.order] / 100, atol=1e-7)
    np.testing.assert_array_equal(obs[23:35], np.arange(12)[mapping.order])
    action, targets = mapping.targets(np.arange(12) / 100)
    np.testing.assert_allclose(targets[mapping.order], mapping.default + .25 * action)


def test_standing_phase_and_last_action(mapping):
    previous = np.arange(12)
    obs = mapping.observation(mapping.to_sdk(mapping.default), np.zeros(12), [1, 0, 0, 0],
                              [0, 0, 0], [0, 0, 0], 7, previous)[0]
    np.testing.assert_array_equal(obs[9:11], [0, 0])
    np.testing.assert_array_equal(obs[35:], previous)


def test_invalid_actions_and_quaternions_fail_closed(mapping):
    for action in (np.full(12, math.nan), np.full(12, 100), np.zeros(11)):
        with pytest.raises(ValueError):
            mapping.targets(action)
    for quaternion in ([0, 0, 0, 0], [math.nan, 0, 0, 1]):
        with pytest.raises(ValueError):
            gravity_body(quaternion)


def test_quaternion_conjugate_rotation():
    np.testing.assert_allclose(gravity_body([math.sqrt(.5), 0, math.sqrt(.5), 0]), [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(gravity_body([0, 0, 0, 1]), [0, 0, -1])


def test_remote_stop_requires_l2_and_b():
    assert remote_estop([0, 0, 0x20, 0x02])
    assert not remote_estop([0, 0, 0x20, 0])
    assert not remote_estop([0, 0, 0, 0x02])


def test_late_policy_slot_does_not_cause_catchup_burst():
    deadline, skipped = next_period(1., .02, 1.055)
    assert deadline == pytest.approx(1.06)
    assert skipped == 2
    deadline, skipped = next_period(deadline, .02, 1.061)
    assert deadline == pytest.approx(1.08)
    assert skipped == 0


@pytest.mark.parametrize('config', [
    {'odom_topic': '/utlidar/robot_odom', 'cloud_topic': '/independent/cloud'},
    {'odom_topic': '/independent/odom', 'cloud_topic': '/utlidar/cloud_deskewed'},
])
def test_sport_derived_navigation_is_rejected_before_release(config):
    with pytest.raises(RuntimeError, match='LIVE BLOCKED before releasing'):
        require_independent_navigation(config)
