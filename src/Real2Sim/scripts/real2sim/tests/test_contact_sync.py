"""Touchdown diagnostics must retain residual sign and reject inverted loads."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diagnose_contact_sync import analyze_pair, match_onsets


def test_known_twenty_ms_residual_is_measured_without_applied_shift():
    t = np.arange(0., 8., .005)
    real = ((t % .6) < .3).astype(float)
    sim = (((t - .020) % .6) < .3).astype(float)
    data = dict(t=t, valid=(t > .2) & (t < 7.8), steady=(t >= .5) & (t <= 7.5),
        real_foot_sensor_raw=np.tile((10 + 30 * real)[:, None], (1, 4)),
        sim_foot_fz_N_before=np.tile((60 * sim)[:, None], (1, 4)))
    original = data['sim_foot_fz_N_before'].copy()
    sync = dict(passed=True, applied_lag_s=.045,
        calibration=dict(window_s=[1., 3.4]), validation=dict(window_s=[4.2, 6.6]))
    result, _ = analyze_pair(data, sync)
    for foot in result['feet'].values():
        for window in ('calibration', 'validation'):
            events = foot['windows'][window]['touchdown']
            assert events['matched'] >= 3
            assert events['median_ms'] == pytest.approx(20., abs=5.)
    np.testing.assert_array_equal(data['sim_foot_fz_N_before'], original)
    assert result['applied_joint_shift_ms'] == 45


def test_misleading_negative_load_cannot_be_used_for_contact_detection():
    data = dict(t=np.arange(1000) * .005, valid=np.ones(1000, bool), steady=np.ones(1000, bool),
        real_foot_sensor_raw=np.ones((1000, 4)), sim_foot_fz_N_before=-np.ones((1000, 4)))
    with pytest.raises(ValueError, match='negative'):
        analyze_pair(data, {})


def test_event_matching_does_not_reuse_sim_touchdown():
    result = match_onsets([1., 1.02], [1.01], [0., 2.])
    assert result['real_events'] == 2 and result['matched'] == 1
