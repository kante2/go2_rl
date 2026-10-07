"""Use the upstream localPlanner/pathFollower without the wheeled motor driver."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package = get_package_share_directory("go2_autonomous_driving")
    planner = get_package_share_directory("local_planner")
    speed = 0.5
    limits = {"maxSpeed": speed, "autonomySpeed": speed, "autonomyMode": True,
              "twoWayDrive": False, "sensorOffsetX": 0.0, "sensorOffsetY": 0.0}
    return LaunchDescription([
        DeclareLaunchArgument("command_port", default_value="9871"),
        DeclareLaunchArgument("telemetry_port", default_value="9872"),
        Node(package="local_planner", executable="localPlanner", name="localPlanner",
             remappings=[("/way_point", "/go2/planner_way_point")],
             output="screen", parameters=[dict(limits, **{
                 "pathFolder": planner + "/paths", "vehicleLength": 0.65,
                 "vehicleWidth": 0.4, "useTerrainAnalysis": False,
                 "checkObstacle": True, "minRelZ": -0.2, "maxRelZ": 0.5,
                 "pathScale": 0.8, "minPathScale": 0.5,
                 "goalClearRange": 0.2, "omniDirGoalThre": 0.5,
             })]),
        Node(package="local_planner", executable="pathFollower", name="pathFollower",
             output="screen", parameters=[dict(limits, **{
                 "realRobot": False, "pubSkipNum": 4, "maxAccel": 0.4,
                 "maxYawRate": 28.0, "yawRateGain": 1.0, "stopYawRateGain": 1.0,
                 # The trained phase observation becomes zero below 0.1 m/s.
                 # Slow down close enough that the 0.2 m goal gate fires first.
                 "slowDwnDisThre": 0.7, "stopDisThre": 0.15, "omniDirGoalThre": 0.5,
                 "omniDirDiffThre": 1.5, "noRotAtGoal": True,
             })]),
        Node(package="go2_autonomous_driving", executable="navigation_bridge",
             name="navigation_bridge", output="screen",
             parameters=[package + "/config/navigation.yaml", {
                 "command_port": ParameterValue(LaunchConfiguration("command_port"), value_type=int),
                 "telemetry_port": ParameterValue(LaunchConfiguration("telemetry_port"), value_type=int),
                 "max_forward_speed": speed,
                 "max_lateral_speed": 0.2,
                 "max_yaw_rate": 0.5,
             }]),
    ])
