"""QuietPaw LongStride 모델 선택/입력 검사와 UDP timeout 회귀 검사."""

import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_autodrive.core import CommandWatchdog, Velocity, command_packet
from go2_autodrive.simulation import go2_mujuco_udp_runner as runner


@pytest.fixture
def config_path(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    for name in ("model_9.pt", "model_100.pt", "model_20.pt", "model_wrong.pt"):
        (folder / name).touch()
    path = tmp_path / "simulation.json"
    path.write_text(json.dumps(dict(
        model_dir="models", checkpoint=None, quietness=.5,
        telemetry_port=9872,
    )))
    return path


def test_selects_highest_iteration_and_cli_quietness(config_path):
    config, checkpoint = runner.load_config(runner.parse_args([
        "--config", str(config_path), "--quietness", "0.8",
    ]))
    assert checkpoint == config_path.parent / "models/model_100.pt"
    assert config["quietness"] == .8
    assert config["udp_port"] == 9872


def test_config_and_cli_checkpoint_override(config_path):
    config = json.loads(config_path.read_text())
    config["checkpoint"] = "model_9.pt"
    config_path.write_text(json.dumps(config))
    args = runner.parse_args(["--config", str(config_path)])
    assert runner.load_config(args)[1].name == "model_9.pt"
    args.checkpoint = config_path.parent / "models/model_20.pt"
    assert runner.load_config(args)[1].name == "model_20.pt"


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_quietness(config_path, value):
    args = runner.parse_args(["--config", str(config_path)])
    args.quietness = value
    with pytest.raises(ValueError, match="quietness"):
        runner.load_config(args)


def test_missing_checkpoint(config_path):
    args = runner.parse_args(["--config", str(config_path), "--checkpoint", "/missing/model.pt"])
    with pytest.raises(FileNotFoundError):
        runner.load_config(args)


def checkpoint_shape(inputs=48):
    return dict(
        infos=dict(
            quietpaw={"stride": {"gait_period": .7}},
            quietpaw_agent={"actor": {"class_name": "MLPModel", "hidden_dims": [64, 32]}},
        ),
        actor_state_dict={
            "mlp.0.weight": SimpleNamespace(shape=(64, inputs)),
            "mlp.4.weight": SimpleNamespace(shape=(12, 32)),
        },
    )


def test_saved_network_sizes_are_used():
    runner.check_policy_shape(checkpoint_shape())


def test_baseline_inputs_rejected():
    with pytest.raises(ValueError, match="48-input"):
        runner.check_policy_shape(checkpoint_shape(inputs=47))


def test_non_longstride_checkpoint_rejected():
    state = checkpoint_shape()
    del state["infos"]["quietpaw"]["stride"]
    with pytest.raises(ValueError, match="LongStride"):
        runner.check_policy_shape(state)


@pytest.mark.parametrize("model,folder", [
    ("baseline", "baseline"), ("quietpaw", "quietpaw"), ("quietpaw_long", "quiet_paw_long"),
])
def test_model_selects_its_own_folder(config_path, model, folder):
    directory = config_path.parent / "model_pt" / folder
    directory.mkdir(parents=True)
    (directory / "model_300.pt").touch()
    config, checkpoint = runner.load_config(runner.parse_args([
        "--config", str(config_path), "--model", model,
    ]))
    assert checkpoint == directory / "model_300.pt"
    assert config["model"] == model


def test_quietpaw_and_longstride_cannot_be_interchanged():
    state = checkpoint_shape()
    with pytest.raises(ValueError, match="LongStride"):
        runner.check_policy_shape(state, "quietpaw")
    del state["infos"]["quietpaw"]["stride"]
    runner.check_policy_shape(state, "quietpaw")
    with pytest.raises(ValueError, match="LongStride"):
        runner.check_policy_shape(state, "quietpaw_long")


def test_baseline_accepts_47_inputs_only():
    state = dict(infos={}, actor_state_dict={
        "mlp.0.weight": SimpleNamespace(shape=(512, 47)),
        "mlp.6.weight": SimpleNamespace(shape=(12, 128)),
    })
    runner.check_policy_shape(state, "baseline")
    state["actor_state_dict"]["mlp.0.weight"].shape = (512, 48)
    with pytest.raises(ValueError, match="47-input"):
        runner.check_policy_shape(state, "baseline")


def test_udp_command_and_timeout(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.setblocking(False)
        state = dict(receiver=receiver, watchdog=CommandWatchdog(timeout=.3))
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(command_packet(Velocity(.2, 0, .1), 1, 10), receiver.getsockname())
        monkeypatch.setattr(runner.time, "monotonic", lambda: 10)
        assert runner.receive_udp_command_for_mujoco(state) == Velocity(.2, 0, .1)
        monkeypatch.setattr(runner.time, "monotonic", lambda: 10.5)
        assert runner.receive_udp_command_for_mujoco(state) == Velocity()
