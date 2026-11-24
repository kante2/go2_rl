"""Check lag direction and fixed sample selection with known synthetic delays."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from real2sim.analyze_time_delay import sweep, estimate


@pytest.mark.parametrize('lag', [.058, -.042])
def test_recovers_known_lag_sign_in_independent_windows(lag):
    t = np.arange(0., 9., .005)
    def wave(time):
        return np.sin(2 * np.pi * 1.7 * time) + .3 * np.sin(2 * np.pi * 3.9 * time + .2)
    real, sim = wave(t)[:, None], wave(t - lag)[:, None]
    lags = np.arange(-250, 251) / 1000
    for window in ((1., 3.), (5., 7.)):
        curve, _ = sweep(t, real, sim, np.ones(len(t), bool), window, lags)
        result = estimate(curve[:, 0], lags)
        assert abs(result['lag_ms'] - lag * 1000) <= 1
        assert result['correlation_best'] > .999


def test_search_excludes_boundaries_and_entire_gap_neighborhood():
    t = np.arange(0., 5., .005)
    signal = np.sin(t)[:, None]
    valid = np.ones(len(t), bool)
    valid[(t >= 2.) & (t <= 2.1)] = False
    _, mask = sweep(t, signal, signal, valid, (0., 5.), np.arange(-250, 251) / 1000)
    for lag in (-.25, 0., .25):
        query = t[mask] + lag
        assert query.min() >= t[0] and query.max() <= t[-1]
        assert np.all(np.interp(query, t, valid.astype(float)) > .999)
