import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import launch

def generate_launch_description():
    pcl_pbject_detection_node = Node(
        package='pcl_object_detection',
        executable='pcl_object_detection_node',
        name='pcl_pbject_detection',
        output={
            'stdout': 'screen',
            'stderr': 'screen',
        }
    )

    return LaunchDescription([
        pcl_pbject_detection_node
    ])
