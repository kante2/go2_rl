"""ROS launch 전달 인자와 ARM용 터미널 보존 검사. 실제 제어기는 실행하지 않는다."""

import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import shlex
import signal
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

launch = pytest.importorskip("launch")
from launch import LaunchContext


ROOT = Path(__file__).resolve().parents[3]
LAUNCH_PATH = ROOT / "src/go2_autodrive/go2_autodrive/launch/real.launch.py"
spec = importlib.util.spec_from_file_location("real_launch", LAUNCH_PATH)
real_launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_launch)


@pytest.fixture
def project(tmp_path):
    config = tmp_path / "src/go2_autodrive/config"
    config.mkdir(parents=True)
    (config / "quietpaw_long.json").write_text(json.dumps(dict(
        model="quietpaw_long", model_dir="model_pt/quiet_paw_long",
        checkpoint=None, quietness=.2,
    )))
    (tmp_path / "scripts").mkdir()
    return tmp_path


@pytest.mark.parametrize("mode,flag", real_launch.MODE_FLAGS.items())
def test_launch_routes_modes_and_interface(project, mode, flag, monkeypatch):
    context = LaunchContext()
    context.launch_configurations.update(dict(
        project_dir=str(project), model="quietpaw_long", mode=mode, config="",
        checkpoint="", quietness="0.8", seconds="3", export_onnx="false", interface="test_eth",
        speed="0.4", lateral_speed="0.1", yaw_rate="0.3",
    ))
    monkeypatch.setattr(real_launch.sys.stdin, "isatty", lambda: True)
    execute = Mock(side_effect=RuntimeError("test replacement"))
    monkeypatch.setattr(real_launch.os, "execvpe", execute)
    with pytest.raises(RuntimeError, match="test replacement"):
        real_launch.start_controller(context)
    executable, command, env = execute.call_args.args
    assert command[command.index("--config") + 1] == str(project / "src/go2_autodrive/config/quietpaw_long.json")
    assert executable == "bash" and env["GO2_NETWORK_INTERFACE"] == "test_eth"
    if flag:
        assert flag in command
    assert ("--execute" in command) == (mode == "live")
    assert command[command.index("--quietness") + 1] == "0.8"
    for flag, value in (("--speed", "0.4"), ("--lateral-speed", "0.1"), ("--yaw-rate", "0.3")):
        assert command[command.index(flag) + 1] == value


def test_live_launch_keeps_real_stdin_tty(project):
    # 대상 wrapper를 모터 I/O가 없는 입력 확인 프로그램으로 대체한다.
    probe = (
        "import sys; print('TTY=' + str(sys.stdin.isatty()), flush=True); "
        "print('STUB ARM:', flush=True); print('ACCEPTED=' + input(), flush=True)"
    )
    wrapper = project / "scripts/run_go2_real_docker.sh"
    wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -c " + shlex.quote(probe) + "\n")
    master, slave = pty.openpty()
    process = subprocess.Popen(
        ["ros2", "launch", str(LAUNCH_PATH), "mode:=live", "project_dir:=" + str(project)],
        stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
    )
    os.close(slave)
    output = b""
    sent = False
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if select.select([master], [], [], .1)[0]:
                try:
                    chunk = os.read(master, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
                if b"STUB ARM:" in output and not sent:
                    os.write(master, b"ARM\n")
                    sent = True
            if process.poll() is not None and b"ACCEPTED=ARM" in output:
                break
        process.wait(timeout=5)
        assert process.returncode == 0, output.decode(errors="replace")
        assert b"TTY=True" in output and b"ACCEPTED=ARM" in output
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        os.close(master)
