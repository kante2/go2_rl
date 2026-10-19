"""Go2 시뮬레이션용 경로 플래너, 경로 추종기, UDP↔ROS 브리지를 실행한다.

MuJoCo와 학습 모델, 키보드 입력은 각각 별도 터미널에서 실행한다.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    planner = get_package_share_directory("local_planner")
    # 플래너의 목표 속도와 브리지의 전후 속도 제한을 동일하게 설정한다.
    speed = ParameterValue(LaunchConfiguration("max_forward_speed"), value_type=float)
    limits = {
        "maxSpeed": speed,
        "autonomySpeed": speed,
        "autonomyMode": True,
        "twoWayDrive": False,
        "sensorOffsetX": 0.0,
        "sensorOffsetY": 0.0,
    }

    return LaunchDescription([
        # command_port: 브리지 → MuJoCo 속도 명령의 목적지 포트.
        # udp_port: MuJoCo → 브리지 상태 패킷의 수신 포트.
        # MuJoCo 실행 설정의 command_port 및 udp_port(또는 telemetry_port)와 맞춘다.
        DeclareLaunchArgument(
            "command_port", default_value="9871",
            description="브리지가 속도 명령을 보낼 MuJoCo UDP 포트",
        ),
        DeclareLaunchArgument(
            "udp_port", default_value="9872",
            description="브리지가 MuJoCo 상태를 받을 UDP 포트",
        ),
        DeclareLaunchArgument("max_forward_speed", default_value="0.5"),
        DeclareLaunchArgument("max_lateral_speed", default_value="0.2"),
        DeclareLaunchArgument("max_yaw_rate", default_value="0.5"),
        DeclareLaunchArgument("command_timeout", default_value="0.3"),

        # 1. localPlanner: 장애물을 피하며 목표로 향하는 이동 경로를 선택한다. -------------------------
        # 입력: /state_estimation(위치·자세), /registered_scan(장애물 점),
        #       /go2/planner_way_point(브리지가 검사한 웨이포인트).
        # 출력: /path(선택한 이동 경로).
        # 원래 구독 이름 /way_point를 브리지의 목표 토픽으로 연결한다.
        Node(
            package="local_planner",
            executable="localPlanner",
            name="localPlanner",
            remappings=[("/way_point", "/go2/planner_way_point")],
            output="screen",
            parameters=[{
                **limits,
                "pathFolder": planner + "/paths",
                "vehicleLength": 0.65,
                "vehicleWidth": 0.4,
                "useTerrainAnalysis": False,
                "checkObstacle": True,
                "minRelZ": -0.2,
                "maxRelZ": 0.5,
                "pathScale": 0.8,
                "minPathScale": 0.5,
                "goalClearRange": 0.2,
                "omniDirGoalThre": 0.5,
            }],
        ),

        # 2. pathFollower: 현재 자세와 경로를 보고 이동·회전 속도를 계산한다. ----------------------------
        # 입력: /state_estimation(위치·자세), /path(이동 경로).
        # 출력: /cmd_vel(TwistStamped, 몸체 좌표계의 vx·vy·wz).
        # realRobot=False로 외부 플래너의 직렬 모터 송신 기능을 비활성화한다.
        Node(
            package="local_planner",
            executable="pathFollower",
            name="pathFollower",
            output="screen",
            parameters=[{
                **limits,
                "realRobot": False,
                "pubSkipNum": 4,
                "maxAccel": 0.4,
                "maxYawRate": 28.0,  # 외부 플래너가 사용하는 단위: deg/s.
                "yawRateGain": 1.0,
                "stopYawRateGain": 1.0,
                # 정책의 보행 위상 입력은 속도 0.1m/s 미만에서 0이 된다.
                # 목표 근처의 감속 구간과 브리지의 도착 판정 거리(0.2m)를 맞춘다.
                "slowDwnDisThre": 0.7,
                "stopDisThre": 0.15,
                "omniDirGoalThre": 0.5,
                "omniDirDiffThre": 1.5,
                "noRotAtGoal": True,
            }],
        ),

        # 3. navigation_bridge: 새 패키지의 UDP↔ROS 변환 노드. ----------------------------------------------
        # UDP 상태 → /state_estimation, /registered_scan, TF(map → vehicle).
        # 키보드의 목표·활성 신호를 받고 /go2/planner_way_point로 목표를 전달한다.
        # /cmd_vel → 속도 제한과 주행 조건 확인 → MuJoCo로 UDP 명령 전송.
        # 입력 상태·경로·명령이 오래되거나 이동이 비활성화되면 0 속도를 보낸다.
        Node(
            package="go2_autodrive",
            executable="udp_ros_bridge",
            name="navigation_bridge",
            output="screen",
            parameters=[{
                "command_port": ParameterValue(
                    LaunchConfiguration("command_port"), value_type=int,
                ),
                "udp_port": ParameterValue(
                    LaunchConfiguration("udp_port"), value_type=int,
                ),
                "max_forward_speed": speed,
                "max_lateral_speed": ParameterValue(
                    LaunchConfiguration("max_lateral_speed"), value_type=float,
                ),
                "max_yaw_rate": ParameterValue(
                    LaunchConfiguration("max_yaw_rate"), value_type=float,
                ),  # 브리지에서 사용하는 단위: rad/s.
                "goal_tolerance": 0.2,
                "sensor_timeout": 0.5,
                "command_timeout": ParameterValue(
                    LaunchConfiguration("command_timeout"), value_type=float,
                ),
            }],
        ),
    ])
