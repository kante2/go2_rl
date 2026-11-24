"""Whole-dataset summaries must weight samples and retain incomparable units."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from real2sim.dataset_summary import _physical, _force, write_summary


def test_pooled_rmse_weights_samples_instead_of_averaging_trial_rmse():
    rows = []
    for n, error in ((2, 1.), (1, 3.)):
        a = np.zeros((n, 12))
        rows.append(dict(data=dict(steady=np.ones(n, bool), real_q_before=a, sim_q_before=a+error)))
    result = _physical(rows, 'q')
    assert result['n'] == 36
    assert result['rmse'] == pytest.approx(np.sqrt(11/3))
    assert result['mae'] == result['bias'] == pytest.approx(5/3)
    assert result['centered_rmse'] == pytest.approx(0.)


def test_force_comparison_uses_each_trials_units_and_weighted_shape():
    rows = []
    for n, sign in ((100, 1.), (300, -1.)):
        wave = np.sin(2*np.pi*np.arange(n)/100)[:, None] * np.ones((1,4))
        real, sim = 15+3*wave, 60+20*sign*wave
        reference = dict(real_mean=real.mean(0), real_std=real.std(0), sim_mean=sim.mean(0), sim_std=sim.std(0))
        rows.append(dict(data=dict(steady=np.ones(n,bool), real_foot_sensor_raw=real, sim_foot_fz_N_before=sim),
                         detail=dict(before=dict(force=reference))))
    result = _force(rows)
    assert result['r'] == pytest.approx(-.5)
    assert result['zrmse'] == pytest.approx(np.sqrt(3))
    assert result['real_mean'] == pytest.approx(15)
    assert result['sim_mean'] == pytest.approx(60)


def test_changed_or_missing_source_preserves_previous_txt(tmp_path):
    bundle = tmp_path/'bundle'; bundle.mkdir()
    (bundle/'manifest.json').write_text(json.dumps(dict(status='complete', trials=[dict(id='one',realPath=str(tmp_path/'missing.npz'))],excludedTrials=[])))
    output=tmp_path/'analysis.txt'; output.write_text('previous report')
    with pytest.raises(FileNotFoundError):
        write_summary(bundle, output)
    assert output.read_text() == 'previous report'
