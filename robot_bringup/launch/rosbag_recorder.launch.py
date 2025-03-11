# ROS2 Launch file for rosbag recording of topics listed in a yaml file
from datetime import datetime
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
# from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from yaml import safe_load


def generate_launch_description():
    # Get path to package
    robot_bringup_dir = get_package_share_directory('robot_bringup')

    # Get path to config file
    config_dir = os.path.join(robot_bringup_dir, 'config')
    rosbag_record_yaml = os.path.join(config_dir, 'rosbag_record.yaml')
    topics_for_recording = safe_load(open(rosbag_record_yaml, 'r'))['topics']

    # Get path to rosbag file directory /records absolute path
    recordings_path = os.path.join('/records')

    # Get path to rosbag file
    # Name of rosbag file contains the timestamp of when the rosbag was started
    # rosbag_<timestamp>.bag
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    folderName = 'rosbag_' + timestamp
    rosbag_dir = os.path.join(recordings_path, folderName)

    # Declare launch arguments
    rosbag_storage_dir_arg = DeclareLaunchArgument(
        'rosbag_storage_dir', default_value=rosbag_dir, description='Path to rosbag file')

    # add rosbag folder to the rosbaf_storage_dir
    rosbag_storage_dir = LaunchConfiguration('rosbag_storage_dir')

    # Start rosbag recording
    # Arguments:
    #   -o: Destination of the bagfile to create
    #   -a: Record all topics
    #   -c: Record only topics listed in yaml file
    #   -s: mcap mode
    rosbag_cmd = ['ros2', 'bag', 'record']
    if topics_for_recording:
        for topic in topics_for_recording:
            rosbag_cmd.append(topic)
    rosbag_cmd.extend(['-o', rosbag_storage_dir, '-s', 'mcap'])
    start_rosbag_yaml = ExecuteProcess(
        cmd=rosbag_cmd,
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(rosbag_storage_dir_arg)
    ld.add_action(start_rosbag_yaml)
    return ld