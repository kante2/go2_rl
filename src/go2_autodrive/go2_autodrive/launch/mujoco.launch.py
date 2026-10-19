"""모델 종류를 선택해 MuJoCo, 웹 뷰어, 경로 플래너, UDP 브리지를 실행한다.

ros2 launch go2_autodrive mujoco.launch.py model:=quietpaw_long
ros2 launch go2_autodrive mujoco.launch.py model:=baseline checkpoint:=model_34997.pt
키보드는 별도 터미널에서 ros2 run go2_autodrive keyboard_waypoint로 실행한다.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, ExecuteProcess, IncludeLaunchDescription,
    LogInfo, OpaqueFunction, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import FindExecutable, LaunchConfiguration


from go2_autodrive.model_config import MODEL_FOLDERS, read_model_config, is_model_config


def default_project_dir():
    """소스/심볼릭 링크 설치와 Docker 설치 경로를 모두 지원한다."""
    if os.environ.get("GO2_PROJECT_ROOT"):
        return os.environ["GO2_PROJECT_ROOT"]
    for directory in Path(__file__).resolve().parents:
        if (directory / "scripts/run_go2_docker.sh").is_file():
            return str(directory)
    return "/workspace/unitree_rl_mjlab"


def start_simulation(context):
    project = Path(LaunchConfiguration("project_dir").perform(context)).resolve()
    package = project / "src/go2_autodrive"
    model = LaunchConfiguration("model").perform(context)
    if model not in MODEL_FOLDERS:
        raise ValueError(f"model은 {', '.join(MODEL_FOLDERS)} 중 선택하세요")

    config_arg = LaunchConfiguration("config").perform(context)
    config_path, config = read_model_config(package / "config", model, config_arg or None)
    if is_model_config(config):
        model_dir = (config_path.parent / config["model_dir"]).resolve()
    else:
        model_dir = config_path.parent / "model_pt" / MODEL_FOLDERS[model]
    checkpoint = LaunchConfiguration("checkpoint").perform(context)
    if checkpoint:
        checkpoint_path = (model_dir / checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    elif not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # ROS Python과 ML Python의 버전이 다르므로 ML 전용 프로세스로 실행한다.
    command = [
        FindExecutable(name="go2-python"), "-m",
        "src.go2_autodrive.go2_autodrive.simulation.go2_mujuco_udp_runner",
        "--config", str(config_path), "--model", model,
        "--viewer", LaunchConfiguration("viewer"),
        "--viewer-port", LaunchConfiguration("viewer_port"),
        "--monitor-port", LaunchConfiguration("monitor_port"),
    ]
    if checkpoint:
        command += ["--checkpoint", str(checkpoint_path)]
    for argument in ("quietness", "device", "steps"):
        value = LaunchConfiguration(argument).perform(context)
        if value:
            command += ["--" + argument, value]
    simulator = ExecuteProcess(cmd=command, cwd=str(project), output="screen")

    actions = [
        LogInfo(msg=f"Go2 model={model}, config={config_path}"),
        # 뷰어 종료/모델 로딩 실패 후 플래너만 남아 실행되지 않도록 함께 종료한다.
        RegisterEventHandler(OnProcessExit(
            target_action=simulator,
            on_exit=[EmitEvent(event=Shutdown(reason="MuJoCo process exited"))],
        )),
        simulator,
    ]
    if LaunchConfiguration("navigation").perform(context).lower() == "true":
        navigation = package / "go2_autodrive/launch/simulation.launch.py"
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(navigation)),
            launch_arguments={
                "command_port": str(config["command_port"]),
                "udp_port": str(config.get("udp_port", config.get("telemetry_port"))),
                "max_forward_speed": str(config["max_forward_speed"]),
                "max_lateral_speed": str(config["max_lateral_speed"]),
                "max_yaw_rate": str(config["max_yaw_rate"]),
                "command_timeout": str(config["command_timeout"]),
            }.items(),
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("project_dir", default_value=default_project_dir(),
                              description="go2_rl 저장소 경로"),
        DeclareLaunchArgument("model", default_value="quietpaw_long",
                              choices=list(MODEL_FOLDERS), description="보행 모델 종류"),
        DeclareLaunchArgument("checkpoint", default_value="",
                              description="모델 폴더 안의 PT 파일명. 생략하면 가장 큰 반복 번호"),
        DeclareLaunchArgument("config", default_value="", description="모델별 JSON 경로. 기본 config/<model>.json"),
        DeclareLaunchArgument("quietness", default_value="",
                              description="QuietPaw의 조용함 0~1. 생략하면 config 값"),
        DeclareLaunchArgument("viewer", default_value="viser", choices=["viser", "none"]),
        DeclareLaunchArgument("viewer_port", default_value="8080"),
        DeclareLaunchArgument("monitor_port", default_value="9910",
                              description="보행 UI UDP 포트. 0이면 측정 비활성화"),
        DeclareLaunchArgument("device", default_value="", description="예: cuda:0"),
        DeclareLaunchArgument("steps", default_value="", description="생략하면 종료할 때까지 실행"),
        DeclareLaunchArgument("navigation", default_value="true", choices=["true", "false"],
                              description="플래너/브리지도 실행. 이미 실행 중이면 false"),
        OpaqueFunction(function=start_simulation),
    ])
