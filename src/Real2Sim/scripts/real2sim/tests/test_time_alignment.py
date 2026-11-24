"""Validate common lag recovery, held-out rejection, and periodic ambiguity."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from real2sim.time_alignment import estimate_recorded_alignment


def signals(early=.060, late=None, frequency=1.7):
    t = np.arange(0., 12., .005)
    lag = np.full(len(t), early)
    if late is not None: lag[t >= 6.] = late
    def wave(time, derivative=False):
        w = 2 * np.pi * frequency
        if derivative:
            return w * np.cos(w * time) + .3 * 2 * np.pi * 3.9 * np.cos(2 * np.pi * 3.9 * time + .2)
        return np.sin(w * time) + .3 * np.sin(2 * np.pi * 3.9 * time + .2)
    data = dict(t=t, valid=(t > .2) & (t < 11.8), steady=(t >= 1.) & (t <= 11.))
    for domain, time in (('real', t), ('sim', t - lag)):
        for quantity in ('q', 'dq'):
            data[domain + '_' + quantity] = np.tile(wave(time, quantity == 'dq')[:, None], (1, 12))
    return data


@pytest.mark.parametrize('lag', [.060, -.045, 0.])
def test_validated_common_shift_has_correct_sign(lag):
    result = estimate_recorded_alignment(signals(lag))
    assert result['passed']
    assert result['applied_lag_s'] == pytest.approx(lag, abs=.005)
    assert abs(result['held_out_lag_difference_s']) <= .005
    assert result['calibration']['window_s'][1] < result['validation']['window_s'][0]


def test_late_drift_rejects_early_shift_instead_of_applying_it():
    result = estimate_recorded_alignment(signals(.060, late=.120))
    assert not result['passed']
    assert result['applied_lag_s'] == 0.
    assert 'validation:lag_changed' in result['rejection_reasons']


def test_static_data_is_not_accepted_as_synchronized():
    data = signals()
    for domain in ('real', 'sim'):
        for quantity in ('q', 'dq'): data[domain + '_' + quantity][:] = 0.
    result = estimate_recorded_alignment(data)
    assert not result['passed']
    assert result['applied_lag_s'] == 0.


def test_ambiguous_short_period_is_not_accepted():
    data = signals(.060)
    for domain, t in (('real', data['t']), ('sim', data['t'] - .060)):
        data[domain + '_q'] = np.tile(np.sin(2 * np.pi * 5 * t)[:, None], (1, 12))
        data[domain + '_dq'] = np.tile((2 * np.pi * 5 * np.cos(2 * np.pi * 5 * t))[:, None], (1, 12))
    result = estimate_recorded_alignment(data)
    assert not result['passed']
    assert result['applied_lag_s'] == 0.
    assert any('ambiguous_period' in reason for reason in result['rejection_reasons'])


def weak_joint_shape_with_contacts(contact_lag=-.125, late_contact_lag=None, period=.6):
    data = signals(-.125)
    t = data['t']
    for quantity in ('q', 'dq'):
        values = data['sim_' + quantity]
        noise = np.column_stack([np.sin(2 * np.pi * 11.3 * t + j * .4) for j in range(12)])
        values += noise / noise.std(0) * values.std(0) * 1.2
    phase = np.full(len(t), contact_lag)
    if late_contact_lag is not None: phase[t >= 6.] = late_contact_lag
    real = ((t % period) < period * .5).astype(float)
    sim = (((t - phase) % period) < period * .5).astype(float)
    data['real_foot_sensor_raw'] = np.tile((10 + 40 * real)[:, None], (1, 4))
    data['sim_foot_fz_N'] = np.tile((70 * sim)[:, None], (1, 4))
    return data


def test_consistent_contacts_can_corroborate_weak_joint_shape():
    result = estimate_recorded_alignment(weak_joint_shape_with_contacts())
    assert result['passed']
    assert result['reference'] == 'contact_load_with_joint_consensus'
    assert result['applied_lag_s'] == pytest.approx(-.125, abs=.005)
    assert any('low_correlation' in reason for reason in result['joint_only_rejection_reasons'])


@pytest.mark.parametrize('kwargs', [dict(contact_lag=.100), dict(late_contact_lag=-.060)])
def test_contact_support_cannot_override_joint_conflict_or_late_drift(kwargs):
    result = estimate_recorded_alignment(weak_joint_shape_with_contacts(**kwargs))
    assert not result['passed']
    assert result['applied_lag_s'] == 0


def test_periodic_contact_ambiguity_does_not_override_weak_joints():
    result = estimate_recorded_alignment(weak_joint_shape_with_contacts(period=.2))
    assert not result['passed']
    assert result['applied_lag_s'] == 0


def long_phase_shift(lag=.315, early_signature=True):
    """Early transient selects a cycle; the late gait is exactly periodic."""
    data = signals()
    for domain, time in (('real', data['t']), ('sim', data['t'] - lag)):
        phase = 2 * np.pi * time / .6
        if early_signature:
            phase += 1.5 * np.exp(-((time - 3.) / .5) ** 2)
        q = np.sin(phase) + .2 * np.sin(2 * phase + .3)
        data[domain + '_q'] = np.tile(q[:, None], (1, 12))
        data[domain + '_dq'] = np.tile(np.gradient(q, .005)[:, None], (1, 12))
    return data


@pytest.mark.parametrize('lag', [.315, -.330])
def test_boundary_expansion_uses_early_cycle_and_validates_late_residual(lag):
    result = estimate_recorded_alignment(long_phase_shift(lag))
    assert result['passed']
    assert result['applied_lag_s'] == pytest.approx(lag, abs=.005)
    assert result['search_expansion']['initial_max_lag_s'] == .25
    assert result['config']['max_lag_s'] == .5
    assert not result['calibration']['ambiguous']
    assert result['validation']['ambiguous']  # Other cycles remain equally good.
    assert result['validation']['cycle_anchored']
    assert result['validation']['correlation_at_calibration_lag'] > .99
    assert abs(result['held_out_lag_difference_s']) <= .005


def test_expansion_rejects_a_periodic_signal_without_an_early_cycle_anchor():
    result = estimate_recorded_alignment(long_phase_shift(early_signature=False))
    assert 'search_expansion' in result
    assert not result['passed']
    assert result['applied_lag_s'] == 0
    assert 'calibration:ambiguous_period' in result['rejection_reasons']


def test_expansion_does_not_override_an_existing_drift_rejection():
    result = estimate_recorded_alignment(signals(.315, late=.390))
    assert not result['passed']
    assert result['applied_lag_s'] == 0
    assert 'validation:lag_changed' in result['rejection_reasons']
    assert 'search_expansion' not in result
