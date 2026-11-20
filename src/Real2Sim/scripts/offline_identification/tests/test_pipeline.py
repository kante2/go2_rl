"""Contract, causality and physics checks for identification, using synthetic logs."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_common import command_at, hysteresis, interpolate_quaternion, load_dataset
from prepare_dataset import prepare
from evaluate_real2sim import match_events


def raw_recording(path):
    t = 100 + np.arange(0, 16, .002)
    n = len(t)
    q = np.tile([.1, .9, -1.8, -.1, .9, -1.8, .1, .9, -1.8, -.1, .9, -1.8], (n, 1))
    arrays = {"ls_t": t, "ls_tick": np.arange(n) * 2, "lc_t": t - .001,
              "ls_quaternion": np.tile([1., 0., 0., 0.], (n, 1)),
              "ls_ff": np.full((n, 4), 40.), "ls_q": q, "lc_q": q}
    arrays.update({"ls_" + k: np.zeros((n, 12 if k in ("dq", "tau") else 3))
                   for k in ("dq", "tau", "acc", "gyro")})
    arrays.update({"lc_" + k: np.full((n, 12), v) for k, v in (("dq", 0.), ("tau", 0.), ("kp", 20.), ("kd", 1.))})
    np.savez(path, **arrays)


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "raw.npz"
    raw_recording(raw)
    out = tmp_path / "prepared.npz"
    prepare(raw, out)
    return load_dataset(out)


def test_commands_are_causal_and_delay_uses_history():
    times = np.array([1., 2., 3.])
    assert command_at(times, [1., 1.99, 2.]).tolist() == [0, 0, 1]
    assert command_at(times, [2., 3.], .01).tolist() == [0, 1]
    with pytest.raises(ValueError, match="history"):
        command_at(times, [1.], .01)


def test_rotation_interpolation_handles_quaternion_sign():
    q = np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
    result = interpolate_quaternion([0., 1.], q, [.25, .5, .75])
    np.testing.assert_allclose(abs(result[:, 0]), 1.)


def test_contact_dwell_and_event_missing():
    values = np.tile(np.array([0, 2, 0, 2, 2, 2, 0, 0, 0])[:, None], (1, 4))
    c = hysteresis(values, np.full(4, 1.), np.full(4, .5), .002)
    assert c[:, 0].tolist() == [False, False, False, False, False, True, True, True, False]
    matched = match_events([0., 1.], [.01, .03, 2.])
    assert (matched["matched"], matched["missed"], matched["extra"]) == (1, 1, 2)


def test_clips_and_delay_history_do_not_cross_split_boundaries(dataset):
    data, meta = dataset
    for clip, split in zip(data["clip_bounds"], data["clip_split"]):
        first, end = clip
        a, b = data["t"][[first, end - 1]]
        matching = [(lo, hi) for lo, hi, s in meta["split_blocks_clock_s"]
                    if s == split and a - meta["max_delay_s"] >= lo - 1e-9 and b < hi]
        assert len(matching) == 1


def test_gap_is_not_interpolated(tmp_path):
    raw = tmp_path / "gap.npz"
    raw_recording(raw)
    with np.load(raw) as z:
        arrays = dict(z)
    # Remove 100ms from BOTH state and input streams in a train block.
    for prefix in ("ls", "lc"):
        keep = ~((arrays[prefix + "_t"] > 101.) & (arrays[prefix + "_t"] < 101.1))
        for key in arrays:
            if key.startswith(prefix + "_"):
                arrays[key] = arrays[key][keep]
    np.savez(raw, **arrays)
    output = tmp_path / "prepared.npz"
    prepare(raw, output)
    data, meta = load_dataset(output)
    for start, end in data["clip_bounds"]:
        a, b = data["t"][[start, end - 1]]
        assert b <= 101. or a - meta["max_delay_s"] >= 101.1


def test_free_rollout_reproducible_without_teacher_forcing(dataset):
    pytest.importorskip("mujoco")
    from mujoco_replay import Replay, DEFAULT_PARAMETERS
    data, _ = dataset
    engine = Replay()
    first, _ = engine.rollout(data, 0)
    start, end = data["clip_bounds"][0]
    data["q"][start+1:end] += .5  # Future measurements must not drive simulation.
    data["gyro"][start+1:end] += .5
    second, _ = engine.rollout(data, 0)
    np.testing.assert_array_equal(first["q"], second["q"])
    np.testing.assert_array_equal(first["quaternion"], second["quaternion"])
    assert first["force_world"][1:, :, 2].sum(axis=1).mean() > 0
    # Updating model constants must not contaminate a later nominal replay.
    engine.rollout(data, 0, DEFAULT_PARAMETERS | {"base_mass_scale": 1.2})
    third, _ = engine.rollout(data, 0)
    np.testing.assert_allclose(first["q"], third["q"], atol=1e-12)


def test_delay_changes_motion_and_not_initial_state(dataset):
    pytest.importorskip("mujoco")
    from mujoco_replay import Replay, DEFAULT_PARAMETERS
    data, _ = dataset
    start, _ = data["clip_bounds"][0]
    change = data["command_t"] >= data["t"][start] + .02
    data["command_q"][change, 1] += .15
    engine = Replay()
    nominal, _ = engine.rollout(data, 0)
    delayed, _ = engine.rollout(data, 0, DEFAULT_PARAMETERS | {"delay_s": .02})
    np.testing.assert_array_equal(nominal["q"][0], delayed["q"][0])
    assert np.max(abs(nominal["q"] - delayed["q"])) > 1e-4


def test_known_parameter_has_lower_prediction_loss(dataset):
    pytest.importorskip("mujoco")
    from mujoco_replay import Replay, DEFAULT_PARAMETERS, errors, prediction_cost
    data, _ = dataset
    engine = Replay()
    p = DEFAULT_PARAMETERS | {"gain_thigh": .65, "gain_calf": .75}
    truth, _ = engine.rollout(data, 0, p)
    start, end = data["clip_bounds"][0]
    for key in ("q", "dq", "gyro", "quaternion"):
        data[key][start:end] = truth[key]
    recovered, _ = engine.rollout(data, 0, p)
    wrong, _ = engine.rollout(data, 0)
    assert prediction_cost(errors(recovered, data, 0)) < 1e-15
    assert prediction_cost(errors(wrong, data, 0)) > 1e-4


def test_contact_thresholds_use_train_only(tmp_path):
    raw = tmp_path / "raw.npz"
    raw_recording(raw)
    first = tmp_path / "first.npz"
    meta = prepare(raw, first)
    with np.load(raw) as z:
        arrays = dict(z)
    held_out = np.zeros(len(arrays["ls_t"]), dtype=bool)
    for a, b, split in meta["split_blocks_clock_s"]:
        if split != "train":
            held_out |= (arrays["ls_t"] >= a) & (arrays["ls_t"] < b)
    arrays["ls_ff"][held_out] += 1000.
    np.savez(raw, **arrays)
    second_meta = prepare(raw, tmp_path / "second.npz")
    assert meta["contact"] == second_meta["contact"]
