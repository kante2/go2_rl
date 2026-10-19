"""실로봇 모델 선택/매핑/실행 모드 검증. ROS와 모터 I/O는 대체한다."""

import argparse
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/go2_autodrive/go2_autodrive/real"))
sys.path.insert(0, str(ROOT / "src/go2_autonomous_driving"))
import go2_real_udp_runner as runner


@pytest.fixture
def settings(tmp_path):
    for folder, _, _ in runner.MODELS.values():
        directory = tmp_path / "model_pt" / folder
        directory.mkdir(parents=True)
        for name in ("model_9.pt", "model_100.pt"):
            (directory / name).write_bytes(name.encode())
        (directory / "policy.onnx").write_bytes(b"source")
    path = tmp_path / "real.json"
    path.write_text(json.dumps(dict(
        model="quietpaw_long", model_root="model_pt", checkpoint="model_9.pt",
        quietness=.5, real={"command_port": 9911, "keyboard_forward_speed": .1},
    )))
    return path


@pytest.mark.parametrize("model", runner.MODELS)
def test_model_directory_and_checkpoint_overrides(settings, model):
    args = runner.parse_args(["--config", str(settings), "--model", model])
    config, selected, checkpoint, source, quietness = runner.load_config(args)
    assert selected == model and config["command_port"] == 9911 and quietness == .5
    assert checkpoint.parent.name == runner.MODELS[model][0]
    assert checkpoint.name == ("model_9.pt" if model == "quietpaw_long" else "model_100.pt")
    assert source.name == "policy.onnx"
    args.export_onnx = True
    assert runner.load_config(args)[3] is None


def metadata_for(model):
    quietpaw = model != "baseline"
    metadata = dict(
        policy_family="quietpaw" if quietpaw else "baseline",
        policy_variant=runner.MODELS[model][1], observation_size=48 if quietpaw else 47,
        policy_to_sdk=[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        default_joint_pos=[-.1, .9, -1.8, .1, .9, -1.8] * 2,
        action_scale=.25, joint_limits=[[-1, 1], [-1, 4], [-2.7, -.8]] * 4,
        observation_names=["base_ang_vel", "projected_gravity", "command", "phase",
                           "joint_pos", "joint_vel", "actions"],
        step_dt=.02, phase_period=.7 if model == "quietpaw_long" else .6, clip_actions=None,
    )
    if quietpaw:
        metadata["observation_names"].append("noise_condition")
        metadata["noise_condition"] = dict(min_cost_limit=.02, max_cost_limit=.35)
    if model == "quietpaw_long":
        metadata["stride"] = {"gait_period": .7}
    return metadata


@pytest.mark.parametrize("model", runner.MODELS)
@pytest.mark.parametrize("execute", [False, True])
def test_backend_receives_correct_mapping_and_arm_flag(tmp_path, model, execute, monkeypatch):
    backend = types.ModuleType("go2_autonomous_driving.real.real_closed_loop")
    backend.ClosedLoopBackend = object()
    control = types.ModuleType("go2_autonomous_driving.real.real_robot")
    control.run = Mock()
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, control.__name__, control)
    state = dict(model=model, quietness=.5, checkpoint=tmp_path / "model.pt",
                 policy_dir=tmp_path, config_path=tmp_path / "real.json")
    runner.run_robot(argparse.Namespace(execute=execute, seconds=3), state)
    args = control.run.call_args.args[0]
    options = control.run.call_args.kwargs
    assert args.execute is execute and args.seconds == 3
    assert options["requires_navigation"] is False and options["send_telemetry"] is True
    mapping = options["mapping_factory"](metadata_for(model))
    obs = mapping.observation(mapping.to_sdk(mapping.default), np.zeros(12), [1, 0, 0, 0],
                              [0, 0, 0], [.2, 0, 0], 7, np.zeros(12))
    assert obs.shape == (1, runner.MODELS[model][2])
    angle = 2 * np.pi * .14 / metadata_for(model)["phase_period"]
    np.testing.assert_allclose(obs[0, 9:11], [np.sin(angle), np.cos(angle)], atol=1e-6)
    if model != "baseline":
        assert obs[0, -1] == pytest.approx(.185)


@pytest.mark.parametrize("mode", ["--execute", "--keyboard"])
def test_noninteractive_input_rejected_before_robot_init(mode, monkeypatch):
    monkeypatch.setattr(runner.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    init = Mock()
    monkeypatch.setattr(runner, "init", init)
    with pytest.raises(SystemExit) as error:
        runner.main([mode])
    assert error.value.code == 1
    init.assert_not_called()


def test_keyboard_shares_merged_config_without_model_loading(settings, monkeypatch):
    monkeypatch.setattr(runner.sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    check = Mock(side_effect=AssertionError("Keyboard must not load policy"))
    monkeypatch.setattr(runner, "check_policy", check)
    captured = []
    def keyboard(state):
        captured.append(state["config_path"])
        config = json.loads(state["config_path"].read_text())
        assert config["command_port"] == 9911 and config["keyboard_forward_speed"] == .1
        assert state["checkpoint"] is None
    monkeypatch.setattr(runner, "run_keyboard", keyboard)
    runner.main(["--config", str(settings), "--keyboard"])
    assert captured and not captured[0].exists()
    check.assert_not_called()


def test_failed_validation_never_publishes(settings, monkeypatch):
    args = runner.parse_args(["--config", str(settings), "--model", "baseline"])
    _, model, checkpoint, source, _ = runner.load_config(args)
    monkeypatch.setattr(runner, "ROOT", settings.parent)
    calls = Mock(side_effect=[None, runner.subprocess.CalledProcessError(1, "parity")])
    monkeypatch.setattr(runner.subprocess, "run", calls)
    output = settings.parent / "artifacts/baseline"
    with pytest.raises(runner.subprocess.CalledProcessError):
        runner.prepare_policy(dict(model=model, checkpoint=checkpoint, policy_dir=output, source_onnx=source))
    assert calls.call_count == 2 and not output.exists()
    assert not list(settings.parent.glob(".go2-real-prepare-*"))


def test_wrong_model_metadata_rejected_before_inference(tmp_path):
    checkpoint, policy = tmp_path / "model.pt", tmp_path / "policy.onnx"
    checkpoint.write_bytes(b"pt")
    policy.write_bytes(b"onnx")
    metadata = metadata_for("quietpaw_long")
    metadata.update(checkpoint_sha256=runner.sha256(checkpoint), onnx_sha256=runner.sha256(policy))
    (tmp_path / "policy.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="종류"):
        runner.check_policy(dict(model="baseline", checkpoint=checkpoint, policy_dir=tmp_path,
                                 source_onnx=None, quietness=.5))


@pytest.mark.parametrize("mode", ["--execute", "--keyboard", "--check-only"])
def test_speed_overrides_preserve_limits_and_json(settings, mode):
    original = settings.read_bytes()
    args = runner.parse_args(["--config", str(settings), mode,
                              "--speed", ".4", "--lateral-speed", ".1", "--yaw-rate", ".3"])
    config = runner.load_config(args)[0]
    assert [config[key] for _, key, _ in runner.SPEED_OPTIONS] == [.4, .1, .3]
    defaults = json.loads(runner.REAL_CONFIG.read_text())
    for _, _, limit in runner.SPEED_OPTIONS:
        assert config[limit] == defaults[limit]
    assert settings.read_bytes() == original


@pytest.mark.parametrize("flag", ["--speed", "--lateral-speed", "--yaw-rate"])
@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf", "100"])
def test_invalid_or_excessive_speed_is_rejected(settings, flag, value):
    args = runner.parse_args(["--config", str(settings), flag, value])
    with pytest.raises(ValueError, match="이하"):
        runner.load_config(args)


def test_keyboard_explicit_options_reach_keyboard_process(settings, monkeypatch):
    args = runner.parse_args(["--config", str(settings), "--keyboard", "--speed", ".4"])
    state = runner.init(args)
    execute = Mock()
    monkeypatch.setattr(runner.subprocess, "run", execute)
    try:
        runner.run_keyboard(state)
        command = execute.call_args.args[0]
        assert command[command.index("--speed") + 1] == "0.4"
        assert "--lateral-speed" not in command and "--yaw-rate" not in command
        merged = json.loads(state["config_path"].read_text())
        assert merged["keyboard_forward_speed"] == .4
    finally:
        runner.close(state)
