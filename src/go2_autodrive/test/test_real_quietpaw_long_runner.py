"""LongStride 모델 선택 및 제어 연결 회귀 검사. ROS/로봇 연결 없음."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNNER_DIR = ROOT / "src/go2_autodrive/go2_autodrive/real"
sys.path.insert(0, str(RUNNER_DIR))
sys.path.insert(0, str(ROOT / "src/go2_autonomous_driving"))
spec = importlib.util.spec_from_file_location(
    "long_runner", RUNNER_DIR / "go2_real_udp_runner_queitpaw_long.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_checkpoint_selection_uses_numeric_iteration(tmp_path):
    for name in ("model_9.pt", "model_100.pt", "model_20.pt", "model_bad.pt"):
        (tmp_path / name).touch()
    assert runner.select_checkpoint(tmp_path).name == "model_100.pt"


def test_config_paths_overrides_and_pt_only(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "model_100.pt").touch()
    (folder / "model_200.pt").touch()
    path = tmp_path / "real.json"
    path.write_text(json.dumps(dict(
        model_dir="models", checkpoint="model_100.pt", quietness=.8,
        real={"max_forward_speed": .3},
    )))
    args = runner.parse_args(["--config", str(path)])
    config, checkpoint, source, quietness = runner.load_config(args)
    assert checkpoint == folder / "model_100.pt"
    assert source is None and quietness == .8
    assert config["max_forward_speed"] == .3
    (folder / "policy.onnx").touch()
    args.checkpoint = folder / "model_200.pt"
    args.quietness = .2
    assert runner.load_config(args)[1:] == (folder / "model_200.pt", folder / "policy.onnx", .2)
    args.export_onnx = True
    assert runner.load_config(args)[2] is None


@pytest.fixture
def prepared(tmp_path):
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.write_bytes(b"checkpoint")
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"onnx")
    source = tmp_path / "imported.onnx"
    source.write_bytes(b"onnx")
    metadata = dict(
        policy_family="quietpaw", policy_variant="quietpaw_long_stride",
        observation_size=48,
        observation_names=["base_ang_vel", "projected_gravity", "command", "phase",
                           "joint_pos", "joint_vel", "actions", "noise_condition"],
        policy_to_sdk=[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        default_joint_pos=[-.1, .9, -1.8, .1, .9, -1.8] * 2,
        action_scale=.25, joint_limits=[[-1, 1], [-1, 4], [-2.7, -.8]] * 4,
        noise_condition=dict(min_cost_limit=.02, max_cost_limit=.35),
        step_dt=.02, phase_period=.7, stride={"gait_period": .7}, clip_actions=None,
        checkpoint_sha256=runner.sha256(checkpoint), onnx_sha256=runner.sha256(policy),
        source_onnx_sha256=runner.sha256(source),
    )
    (tmp_path / "policy.json").write_text(json.dumps(metadata))
    state = dict(checkpoint=checkpoint, policy_dir=tmp_path, source_onnx=source,
                 quietness=.5, config_path=tmp_path / "real.json", temp_directory=None)
    return state, metadata


@pytest.mark.parametrize("name", ["checkpoint", "source_onnx", "policy"])
def test_changed_files_fail_before_robot_connection(prepared, name):
    state, _ = prepared
    runner.check_policy(state)
    path = state["policy_dir"] / "policy.onnx" if name == "policy" else state[name]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        runner.check_policy(state)


def test_wrong_phase_period_is_rejected(prepared):
    state, metadata = prepared
    metadata["phase_period"] = .6
    (state["policy_dir"] / "policy.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="주기"):
        runner.check_policy(state)


def test_run_connects_longstride_mapping_and_keeps_arm_options(prepared, monkeypatch):
    state, _ = prepared
    metadata = runner.check_policy(state)
    backend = types.ModuleType("go2_autonomous_driving.real.real_closed_loop")
    backend.ClosedLoopBackend = object()
    control = types.ModuleType("go2_autonomous_driving.real.real_robot")
    control.run = Mock()
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, control.__name__, control)
    runner.run_robot(argparse.Namespace(execute=False, seconds=2), state)
    args = control.run.call_args.args[0]
    options = control.run.call_args.kwargs
    assert args.execute is False and args.seconds == 2
    assert options["requires_navigation"] is False and options["send_telemetry"] is True
    mapping = options["mapping_factory"](metadata)
    obs = mapping.observation(
        mapping.to_sdk(mapping.default), np.zeros(12), [1, 0, 0, 0], [0, 0, 0],
        [.2, 0, 0], 7, np.zeros(12),
    )
    assert obs.shape == (1, 48)
    assert obs[0, -1] == pytest.approx(.185)
    np.testing.assert_allclose(obs[0, 9:11], [np.sin(2*np.pi*.14/.7), np.cos(2*np.pi*.14/.7)], atol=1e-6)


@pytest.mark.parametrize("mode", ["--prepare", "--check-only"])
def test_non_robot_modes_never_start_control(mode, monkeypatch):
    state = {"temp_directory": None}
    monkeypatch.setattr(runner, "init", lambda args: state)
    run = Mock()
    monkeypatch.setattr(runner, "run_robot", run)
    runner.main([mode])
    run.assert_not_called()


def test_failed_parity_never_publishes(prepared, monkeypatch):
    state, _ = prepared
    # 임시 저장소 안에 fixture를 두어 Docker 경로 변환까지 검사한다.
    monkeypatch.setattr(runner, "ROOT", state["checkpoint"].parent)
    error = runner.subprocess.CalledProcessError(1, "parity")
    calls = Mock(side_effect=[None, error])
    monkeypatch.setattr(runner.subprocess, "run", calls)
    with pytest.raises(runner.subprocess.CalledProcessError):
        runner.prepare_policy(state)
    assert calls.call_count == 2
    assert (state["policy_dir"] / "policy.onnx").read_bytes() == b"onnx"
    assert not list(runner.ROOT.glob(".go2-long-prepare-*"))
