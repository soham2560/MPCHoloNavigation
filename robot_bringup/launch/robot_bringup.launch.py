import datetime
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown, SetEnvironmentVariable, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, AndSubstitution, NotSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='False',
            description='Launch in simulation mode.'))
    declared_arguments.append(
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='namespace'))
    declared_arguments.append(
        DeclareLaunchArgument(
            'record',
            default_value='False',
            description='Record in rosbag'))
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_rviz',
            default_value='False',
            description='Launch RVIZ on startup'))
    declared_arguments.append(
        SetEnvironmentVariable(
            'RCUTILS_COLORIZED_OUTPUT', '1'))

    # Initialize Arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    namespace = LaunchConfiguration('namespace')
    record = LaunchConfiguration('record')
    use_rviz = LaunchConfiguration('use_rviz')

    # Package Path
    package_path = get_package_share_directory('robot_bringup')

    # set log output path
    get_current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_full_path = os.path.join('/ros2_ws/src/records/', get_current_timestamp)
    rosbag_full_path = os.path.join(log_full_path, 'rosbag')

    # Get URDF via xacro
    xacro_path = PathJoinSubstitution(
        [package_path, 'urdf', 'mecanum_drive.xacro.urdf']
    )
    
    # Set the robot controller file
    robot_controllers = PathJoinSubstitution([package_path, 'config', 'mecanum_drive_controller.yaml'])
    
    # Params
    controller_manager_timeout = ['--controller-manager-timeout', '30']
    controller_manager_node_name = ['--controller-manager', 'controller_manager']
    
    # Spawn Robot
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        namespace=namespace,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_sim_time),
        arguments=[
            '-name', 'robot',
            '-x', str(0.0),
            '-y', str(0.0),
            '-z', str(0.0),
            '-Y', str(0.0),
            '-topic', 'robot_description'],
    )

    # Gazebo Environment
    gazebo = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(get_package_share_directory('ros_gz_sim'),
                              'launch', 'gz_sim.launch.py')]),
            launch_arguments=[('gz_args', ['-r v 4 empty.sdf'])],
            condition=IfCondition(use_sim_time))
    
    # Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        namespace=namespace,
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time}],
        condition=IfCondition(use_sim_time),
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    )

    # Nodes
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        namespace=namespace,
        parameters=[
            robot_controllers,
            {'use_sim_time': use_sim_time},
        ],
        condition=UnlessCondition(use_sim_time),
        output='both',
        remappings=[
            ('~/robot_description', 'robot_description'),
        ],
        on_exit=Shutdown(),
    )

    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        namespace=namespace,
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'frame_prefix': [namespace, '/'],
                'robot_description': ParameterValue(
                    Command(['xacro ', xacro_path, ' ',
                            'USE_WITH_SIM:=', use_sim_time, ' ',
                            'NAMESPACE:=', namespace, ' ',
                            'YAML_PATH:=', robot_controllers]), value_type=str),
            }
        ]
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
        ],
        arguments=['joint_state_broadcaster'] + controller_manager_node_name
        + controller_manager_timeout,
    )

    mecanum_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
        ],
        arguments=['mecanum_drive_controller'] + controller_manager_node_name
        + controller_manager_timeout,
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        namespace=namespace,
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-d', package_path + '/rviz/robot.rviz'],
        on_exit=Shutdown(),
        condition=IfCondition(use_rviz),
    )

    # Delay start of rviz after `joint_state_broadcaster`
    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        )
    )

    # Delay start of robot_controller after `joint_state_broadcaster`
    delay_mecanum_drive_controller_spawner_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[mecanum_drive_controller_spawner],
        )
    )

    rosbag_recorder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [package_path, '/launch/rosbag_recorder.launch.py']
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rosbag_storage_dir': rosbag_full_path,
        }.items(),
        condition=IfCondition(record),
    )

    nodes = [
        gz_spawn_entity,
        gazebo,
        bridge,
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        delay_rviz_after_joint_state_broadcaster_spawner,
        delay_mecanum_drive_controller_spawner_after_joint_state_broadcaster_spawner,
        rosbag_recorder_launch
    ]

    return LaunchDescription(declared_arguments + nodes)
