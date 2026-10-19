"""모델별 JSON이 시뮬과 실로봇에서 같은 정책 설정을 선택하는지 검사한다."""

import json
from pathlib import Path
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from go2_autodrive.model_config import MODEL_FOLDERS, read_model_config
from go2_autodrive.real import go2_real_udp_runner as real
from go2_autodrive.simulation import go2_mujuco_udp_runner as sim


@pytest.mark.parametrize("model", MODEL_FOLDERS)
def test_default_profile_selects_same_policy(model):
    args = ["--model", model]
    sim_config, sim_pt = sim.load_config(sim.parse_args(args))
    real_config, selected, real_pt, _, quietness = real.load_config(real.parse_args(args))
    assert selected == sim_config["model"] == model
    assert real_pt == sim_pt
    assert real_pt.parent.name == MODEL_FOLDERS[model]
    assert quietness == sim_config["quietness"]
    path, profile = read_model_config(PACKAGE / "config", model)
    assert path.name == f"{model}.json"
    assert sim_config["command_port"] == profile["command_port"]
    assert real_config["command_port"] == profile["command_port"]
    for key in ("max_forward_speed", "max_lateral_speed", "max_yaw_rate", "command_timeout"):
        assert real_config[key] == sim_config[key] == profile[key]


@pytest.fixture
def profile(tmp_path):
    for name in ("model_9.pt", "model_100.pt"):
        (tmp_path / name).touch()
    data = dict(model="quietpaw", model_dir=".", checkpoint="model_9.pt", quietness=.8,
                telemetry_port=9812, command_port=9811, max_forward_speed=.4)
    path = tmp_path / "quietpaw.json"
    path.write_text(json.dumps(data))
    return path


def test_same_profile_preserves_pinned_checkpoint_and_cli_overrides(profile):
    args = ["--config", str(profile), "--model", "quietpaw"]
    config, pt = sim.load_config(sim.parse_args(args))
    actual, model, real_pt, _, quietness = real.load_config(real.parse_args(args))
    assert pt == real_pt == profile.parent / "model_9.pt"
    assert quietness == config["quietness"] == .8
    assert actual["command_port"] == config["command_port"] == 9811
    assert actual["max_forward_speed"] == config["max_forward_speed"] == .4
    args += ["--checkpoint", str(profile.parent / "model_100.pt"), "--quietness", ".3"]
    config, pt = sim.load_config(sim.parse_args(args))
    _, _, real_pt, _, quietness = real.load_config(real.parse_args(args))
    assert pt == real_pt == profile.parent / "model_100.pt"
    assert quietness == config["quietness"] == .3


@pytest.mark.parametrize("runner", [sim, real])
def test_mismatched_model_profile_is_rejected(profile, runner):
    with pytest.raises(ValueError, match="모델과 JSON"):
        runner.load_config(runner.parse_args(["--config", str(profile), "--model", "baseline"]))


@pytest.mark.parametrize("section", ["simulation", "real"])
def test_policy_settings_cannot_diverge_between_sections(profile, section):
    data = json.loads(profile.read_text())
    data[section] = {"quietness": .1}
    profile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="최상위"):
        read_model_config(profile.parent, path=profile)
