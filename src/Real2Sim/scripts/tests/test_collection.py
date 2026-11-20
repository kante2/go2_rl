"""Command replay, checkpoint identity and output protection without mjlab."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_sim_baseline import CommandSchedule, save_npz, DEFAULT_REFERENCE, DEFAULT_CHECKPOINT, sha256


def test_command_replay_is_causal_and_does_not_ramp_twice():
    s = CommandSchedule([.1, .14, .18], [[0, 0, 0], [.02, 0, 0], [.2, 0, 0]], 1.)
    assert s.at(0).tolist() == [0, 0, 0]
    assert s.at(.139).tolist() == [0, 0, 0]
    assert s.at(.14).tolist() == [.02, 0, 0]
    assert s.at(.179).tolist() == [.02, 0, 0]
    assert s.at(.18).tolist() == [.2, 0, 0]


def test_reference_preserves_timing_and_policy_identity():
    digest = sha256(DEFAULT_CHECKPOINT)
    s, meta = CommandSchedule.from_reference(DEFAULT_REFERENCE, digest)
    with np.load(DEFAULT_REFERENCE) as z:
        origin = z['ev_t'][z['ev_label'] == 'record_start'][0]
        np.testing.assert_allclose(s.t, z['rl_t'] - origin)
        np.testing.assert_array_equal(s.command, z['rl_command'])
    assert meta['vx'] == .2
    assert 25 < s.duration < 26
    with pytest.raises(ValueError, match='SHA256'):
        CommandSchedule.from_reference(DEFAULT_REFERENCE, 'wrong-checkpoint')


def test_generated_ramp_and_post_stop_interval():
    s = CommandSchedule.generated(.2, 1., 2., 1., .02,
        dict(command_acceleration=[.5, .4, 1.], command_deceleration=[.5, .4, 1.]))
    assert s.at(.9)[0] == 0
    assert s.at(1.02)[0] < .2
    assert s.at(2.)[0] == pytest.approx(.2)
    assert s.at(3.8)[0] == 0
    assert s.duration == pytest.approx(4.4)
    assert np.max(np.abs(np.diff(s.command[:, 0]))) <= .01000001


def test_output_is_not_silently_overwritten(tmp_path):
    p = tmp_path / 'sim' / '0.2_1.npz'
    save_npz(p, dict(meta=np.array(json.dumps(dict(completed=True)))))
    original = p.read_bytes()
    with pytest.raises(FileExistsError):
        save_npz(p, dict(q=np.zeros((1, 12))))
    assert p.read_bytes() == original


@pytest.mark.parametrize('times', [[0., 0.], [1., 0.], [0., float('nan')]])
def test_invalid_command_clock_is_rejected(times):
    with pytest.raises(ValueError):
        CommandSchedule(times, np.zeros((2, 3)), 1.)
