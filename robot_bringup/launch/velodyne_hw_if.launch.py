import os
import yaml

import ament_index_python.packages
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    share_dir = ament_index_python.packages.get_package_share_directory('robot_bringup')
    params_file = os.path.join(share_dir, 'config', 'velodyne.yaml')
    with open(params_file, 'r') as f:
        config = yaml.safe_load(f) or {}
        driver_params = config.get('velodyne_driver_node', {}).get('ros__parameters', {})
        laserscan_params = config.get('velodyne_laserscan_node', {}).get('ros__parameters', {})
        convert_params = config.get('velodyne_transform_node', {}).get('ros__parameters', {})

    convert_share_dir = ament_index_python.packages.get_package_share_directory('velodyne_pointcloud')
    convert_params['calibration'] = os.path.join(convert_share_dir, 'params', 'VLP16db.yaml')

    container = ComposableNodeContainer(
            name='velodyne_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='velodyne_driver',
                    plugin='velodyne_driver::VelodyneDriver',
                    name='velodyne_driver_node',
                    parameters=[driver_params]),
                ComposableNode(
                    package='velodyne_pointcloud',
                    plugin='velodyne_pointcloud::Transform',
                    name='velodyne_transform_node',
                    parameters=[convert_params]),
                ComposableNode(
                    package='velodyne_laserscan',
                    plugin='velodyne_laserscan::VelodyneLaserScan',
                    name='velodyne_laserscan_node',
                    parameters=[laserscan_params]),
            ],
            output='both',
    )

    return LaunchDescription([container])
