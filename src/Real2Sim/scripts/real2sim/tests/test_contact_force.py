"""S3 recorded contact-vector direction must not invert the displayed load."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from real2sim.preprocessing import adapt


def test_s3_negative_z_is_positive_load_and_raw_vectors_are_unchanged():
    vector = np.array([[[1., 2., -30.], [0., 0., 0.], [2., 1., -45.], [0., 0., -12.]]])
    source = dict(sim_time=np.array([0.]), sim_foot_force_vec=vector.copy())
    mapped, metadata = adapt(source, dict(source='mujoco'))
    np.testing.assert_array_equal(mapped['sim_foot_fz_N'], [[30., 0., 45., 12.]])
    np.testing.assert_array_equal(source['sim_foot_force_vec'], vector)
    assert 'sim_foot_fz_N' not in source
    assert metadata['force_sign_to_load'] == -1
    again, _ = adapt(mapped, metadata)
    np.testing.assert_array_equal(again['sim_foot_fz_N'], mapped['sim_foot_fz_N'])


def test_unrecognized_positive_z_is_rejected_instead_of_fitted_or_rectified():
    source = dict(sim_time=np.array([0.]), sim_foot_force_vec=np.ones((1, 4, 3)))
    with pytest.raises(ValueError, match='contact-force convention'):
        adapt(source, dict(source='mujoco'))


def test_empty_s3_log_is_still_available_for_explicit_exclusion():
    source = dict(sim_time=np.empty(0), sim_foot_force_vec=np.empty((0, 4, 3)))
    mapped, _ = adapt(source, dict(source='mujoco'))
    assert mapped['sim_foot_fz_N'].shape == (0, 4)
