"""Physical-unit inversion, causal sampling and predictor history isolation."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_converter import ROOT, POLICY_HASH, PolicyMapping, hold_indices, targets_to_action, action_to_targets
from analyze_converter import causal_grf_inputs, histories


def mapping():
    return PolicyMapping(json.loads((ROOT / 'artifacts/go2_real/baseline' / POLICY_HASH / 'policy.json').read_text()))


def test_target_action_inverse_preserves_nontrivial_sdk_order():
    m = mapping()
    action = np.arange(24, dtype=float).reshape(2, 12) / 50
    target = action_to_targets(action, m)
    np.testing.assert_allclose(target[:, m.order], m.default + m.scale * action)
    np.testing.assert_allclose(targets_to_action(target, m), action, atol=1e-12)


def test_receive_clock_hold_never_uses_future_and_rejects_stale():
    i, valid = hold_indices([1., 2., 3.], [.9, 1., 1.01, 1.5, 2.], .1)
    assert i.tolist() == [0, 0, 0, 0, 1]
    assert valid.tolist() == [False, True, True, False, True]
    with pytest.raises(ValueError):
        hold_indices([1., 1.], [1.], .1)


def test_observation_inverse_uses_absolute_q_not_normalized_values():
    m = mapping(); q = np.linspace(-.2, .2, 12)
    obs = m.observation(q, np.zeros(12), [1, 0, 0, 0], [0, 0, 0], [.2, 0, 0], 0, np.zeros(12))[0]
    np.testing.assert_allclose(m.to_sdk(obs[11:23] + m.default), q, atol=1e-7)


def test_grf_preprocessing_does_not_leak_future_samples():
    t = np.arange(0, 10, .005)
    raw = dict(ls_t=t, ls_q=np.ones((len(t), 12)), ls_dq=np.zeros((len(t), 12)),
               sim_foot_fz_N=np.zeros((len(t), 4)))
    changed = {k: v.copy() for k, v in raw.items()}
    changed['ls_q'][t >= 6.] += 10
    changed['ls_dq'][t >= 6.] += 20
    query = np.arange(.1, 9.9, .02)
    original, _, _ = causal_grf_inputs(raw, 0., query, simulated=True)
    later, _, _ = causal_grf_inputs(changed, 0., query, simulated=True)
    np.testing.assert_array_equal(original[query < 6.], later[query < 6.])


def test_history_rejects_every_window_crossing_invalid_frame():
    x = np.arange(36).reshape(12, 3)
    valid = np.ones(12, bool); valid[6] = False
    history, keep = histories(x, valid, 6)
    assert history.shape == (7, 6, 3)
    assert keep.tolist() == [True, False, False, False, False, False, False]
