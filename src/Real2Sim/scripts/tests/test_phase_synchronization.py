"""Constant phase fit must not use held-out events or extrapolate data."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase_synchronization import estimate_alignment, apply_alignment


def synthetic(end_lag=.12):
    t = np.arange(0, 16, .005)
    lag = np.where(t < 8., .12, end_lag)
    real = np.stack([3 + 50 * (((t - phase) % .6) < .28)
                     for phase in (0., .3, .3, 0.)], axis=1)
    sim = np.stack([70 * (((t - lag - phase) % .6) < .28)
                    for phase in (0., .3, .3, 0.)], axis=1)
    return dict(t=t, real_foot_sensor_raw=real, sim_foot_fz_N=sim,
                valid=np.ones(len(t), bool), steady=(t > 2) & (t < 14))


def test_constant_delay_recovered_from_start_events():
    result, events = estimate_alignment(synthetic(), 0., 15.)
    assert abs(result['applied_lag_s'] - .12) <= .005
    assert result['passed']
    assert result['held_out_abs_residual_p90_s'] <= .01
    assert {e['window'] for e in events} == {'calibration', 'validation'}


def test_end_drift_does_not_change_start_fit_and_is_flagged():
    baseline, _ = estimate_alignment(synthetic(), 0., 15.)
    drift, _ = estimate_alignment(synthetic(.20), 0., 15.)
    assert drift['applied_lag_s'] == baseline['applied_lag_s']
    assert not drift['passed']
    assert drift['status'] == 'REVIEW_REQUIRED'
    assert drift['held_out_abs_residual_p90_s'] > .025


def test_one_offset_all_sim_channels_and_overlap_only():
    t = np.arange(0, 1, .01)
    a = dict(t=t, real_q=np.stack([t, t * 2], axis=1),
             sim_q=np.stack([t - .12, 2 * (t - .12)], axis=1),
             sim_dq=np.stack([3 * (t - .12), 4 * (t - .12)], axis=1),
             valid=np.ones(len(t), bool))
    shifted = apply_alignment(a, .12)
    np.testing.assert_allclose(shifted['sim_q'], shifted['real_q'], atol=1e-12)
    np.testing.assert_allclose(shifted['sim_dq'][:, 0], shifted['t'] * 3, atol=1e-12)
    assert shifted['t'][-1] + .12 <= t[-1]
    np.testing.assert_array_equal(shifted['sim_q_unaligned'], a['sim_q'][:len(shifted['t'])])
