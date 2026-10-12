"""QuietPaw's extra observation and baseline compatibility; no hardware I/O."""
from copy import deepcopy

import numpy as np
import pytest

from go2_autonomous_driving.real.real_control import PolicyMapping
from go2_autonomous_driving.real.real_quietpaw import (
    BASE_OBSERVATIONS, QUIETPAW_OBSERVATIONS, QuietPawMapping, cost_limit_from_quietness,
)


@pytest.fixture
def metadata():
    return dict(policy_family='quietpaw', observation_size=48,
                policy_to_sdk=[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
                default_joint_pos=[-.1, .9, -1.8, .1, .9, -1.8] * 2,
                action_scale=.25, joint_limits=[[-1, 1], [-1, 4], [-2.7, -.8]] * 4,
                observation_names=QUIETPAW_OBSERVATIONS.copy(),
                noise_condition=dict(min_cost_limit=.02, max_cost_limit=.35),
                step_dt=.02, phase_period=.6, clip_actions=None)


@pytest.mark.parametrize('quietness,expected', [(0, .35), (.5, .185), (1, .02)])
def test_last_input_is_saved_cost_limit_and_other_inputs_are_unchanged(metadata, quietness, expected):
    mapping = QuietPawMapping(metadata, quietness)
    baseline = PolicyMapping({**metadata, 'observation_names': BASE_OBSERVATIONS})
    args = (mapping.to_sdk(mapping.default) + np.arange(12) / 100, np.arange(12),
            [1, 0, 0, 0], [.1, -.2, .3], [.2, 0, .1], 7, np.arange(12) / 10)
    actual = mapping.observation(*args)
    assert actual.shape == (1, 48) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual[:, :47], baseline.observation(*args))
    assert actual[0, 47] == pytest.approx(expected)
    np.testing.assert_array_equal(actual[0, 35:47], np.asarray(args[-1], dtype=np.float32))
    action = np.arange(12) / 100
    np.testing.assert_array_equal(mapping.targets(action)[1], baseline.targets(action)[1])


def test_condition_uses_checkpoint_specific_limits(metadata):
    metadata['noise_condition'] = dict(min_cost_limit=.1, max_cost_limit=.9)
    mapping = QuietPawMapping(metadata, .25)
    assert mapping.cost_limit == pytest.approx(.7)


@pytest.mark.parametrize('bad', [-.01, 1.01, float('nan'), float('inf')])
def test_invalid_quietness_is_rejected(metadata, bad):
    with pytest.raises(ValueError, match='quietness'):
        QuietPawMapping(metadata, bad)


@pytest.mark.parametrize('low,high', [(.35, .02), (.1, .1), (-.1, .3), (0, 1.1), (0, float('nan'))])
def test_invalid_saved_ranges_are_rejected(low, high):
    with pytest.raises(ValueError, match='cost limits'):
        cost_limit_from_quietness(.5, dict(min_cost_limit=low, max_cost_limit=high))


def test_baseline_and_quietpaw_cannot_silently_load_each_others_layout(metadata):
    original = deepcopy(metadata)
    QuietPawMapping(metadata)
    assert metadata == original
    with pytest.raises(ValueError, match='Unsupported actor observations'):
        PolicyMapping(metadata)
    for field, value in [('policy_family', 'baseline'), ('observation_size', 47),
                         ('observation_names', BASE_OBSERVATIONS),
                         ('observation_names', ['noise_condition'] + BASE_OBSERVATIONS)]:
        with pytest.raises(ValueError, match='QuietPaw 48-input'):
            QuietPawMapping({**metadata, field: value})


def test_quietpaw_keeps_joint_limits_and_imu_validation(metadata):
    mapping = QuietPawMapping(metadata)
    with pytest.raises(ValueError, match='joint limits'):
        mapping.targets(np.full(12, 100.))
    with pytest.raises(ValueError, match='quaternion'):
        mapping.observation(mapping.to_sdk(mapping.default), np.zeros(12), [0, 0, 0, 0],
                            [0, 0, 0], [0, 0, 0], 0, np.zeros(12))
