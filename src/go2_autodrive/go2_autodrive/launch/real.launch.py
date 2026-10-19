"""모델을 선택해 실로봇 정책 준비/검사/제어/키보드를 실행한다.

ros2 launch go2_autodrive real.launch.py model:=quietpaw_long mode:=prepare
ros2 launch go2_autodrive real.launch.py model:=quietpaw_long mode:=check
ros2 launch go2_autodrive real.launch.py model:=quietpaw_long mode:=live interface:=enp3s0

단일 제어 프로세스로 교체 실행해 현재 터미널의 stdin을 보존한다.
ROS ExecuteProcess의 출력용 PTY로는 input('ARM')을 받을 수 없기 때문이다.
모터 제어기의 ARM/비상정지/종료 damping은 그대로 사용한다.
"""

import os
from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


from go2_autodrive.model_config import MODEL_FOLDERS, read_model_config, is_model_config
MODE_FLAGS = {
    "prepare": "--prepare",
    "check": "--check-only",
    "dry": None,
    "live": "--execute",
    "keyboard": "--keyboard",
}


def default_project_dir():
    if os.environ.get("GO2_PROJECT_ROOT"):
        return os.environ["GO2_PROJECT_ROOT"]
    for directory in Path(__file__).resolve().parents:
        if (directory / "scripts/run_go2_real_docker.sh").is_file():
            return str(directory)
    return "/workspace/unitree_rl_mjlab"


def start_controller(context):
    project = Path(LaunchConfiguration("project_dir").perform(context)).resolve()
    model = LaunchConfiguration("model").perform(context)
    mode = LaunchConfiguration("mode").perform(context)
    if model not in MODEL_FOLDERS or mode not in MODE_FLAGS:
        raise ValueError("Unknown model or mode")
    if mode in ("live", "keyboard") and not sys.stdin.isatty():
        raise RuntimeError("ARM/키보드 입력을 받을 수 있는 대화형 터미널에서 실행하세요")

    config_arg = LaunchConfiguration("config").perform(context)
    config_path, config = read_model_config(
        project / "src/go2_autodrive/config", model, config_arg or None,
    )
    command = [
        "bash", str(project / "scripts/run_go2_real_docker.sh"),
        str(project / "src/go2_autodrive/go2_autodrive/real/go2_real_udp_runner.py"),
        "--config", str(config_path), "--model", model,
    ]
    if MODE_FLAGS[mode]:
        command.append(MODE_FLAGS[mode])
    checkpoint = LaunchConfiguration("checkpoint").perform(context)
    if checkpoint:
        if is_model_config(config):
            model_dir = config_path.parent / config["model_dir"]
        else:
            model_dir = config_path.parent / config["model_root"] / MODEL_FOLDERS[model]
        path = (model_dir / checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        command += ["--checkpoint", str(path)]
    for name in ("quietness", "seconds", "speed", "lateral_speed", "yaw_rate"):
        value = LaunchConfiguration(name).perform(context)
        if value:
            command += ["--" + name.replace("_", "-"), value]
    if LaunchConfiguration("export_onnx").perform(context).lower() == "true":
        if mode != "prepare":
            raise ValueError("export_onnx:=true는 mode:=prepare에서만 사용하세요")
        command.append("--export-onnx")

    interface = LaunchConfiguration("interface").perform(context)
    env = dict(os.environ, GO2_NETWORK_INTERFACE=interface)
    print(f"Go2 real: model={model}, mode={mode}, config={config_path}, interface={interface}", flush=True)
    # shell 문자열로 합치지 않고 인자 배열을 전달한다. stdin/TTY는 그대로 상속한다.
    os.execvpe(command[0], command, env)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("project_dir", default_value=default_project_dir()),
        DeclareLaunchArgument("model", default_value="quietpaw_long", choices=list(MODEL_FOLDERS)),
        DeclareLaunchArgument("mode", default_value="dry", choices=list(MODE_FLAGS),
                              description="prepare/check는 로봇 연결 없음, dry는 센서·추론, live는 ARM 후 모터 제어"),
        DeclareLaunchArgument("checkpoint", default_value="",
                              description="모델 폴더 안의 PT 파일명. 생략하면 가장 큰 반복 번호"),
        DeclareLaunchArgument("config", default_value="", description="모델별 JSON 경로. 기본 config/<model>.json"),
        DeclareLaunchArgument("quietness", default_value="", description="QuietPaw 소음 제약 0~1"),
        DeclareLaunchArgument("seconds", default_value="", description="제어 루프 실행 시간"),
        DeclareLaunchArgument("speed", default_value="",
                              description="W/S 전후 목표 속도 [m/s]. JSON 상한 이내"),
        DeclareLaunchArgument("lateral_speed", default_value="",
                              description="A/D 좌우 목표 속도 [m/s]. JSON 상한 이내"),
        DeclareLaunchArgument("yaw_rate", default_value="",
                              description="J/L 회전 목표 속도 [rad/s]. JSON 상한 이내"),
        DeclareLaunchArgument("interface", default_value=os.environ.get("GO2_NETWORK_INTERFACE", "enp3s0"),
                              description="실로봇에 연결된 네트워크 인터페이스"),
        DeclareLaunchArgument("export_onnx", default_value="false", choices=["true", "false"],
                              description="가져온 ONNX 대신 PT에서 다시 변환"),
        OpaqueFunction(function=start_controller),
    ])
