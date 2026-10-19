from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import ExecutableInPackage


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("http_port", default_value="8081"),
        DeclareLaunchArgument("udp_port", default_value="9910"),
        ExecuteProcess(cmd=[
            ExecutableInPackage(package="go2_gait_monitor", executable="dashboard"),
            "--http-port", LaunchConfiguration("http_port"),
            "--udp-port", LaunchConfiguration("udp_port"),
        ], output="screen"),
    ])
