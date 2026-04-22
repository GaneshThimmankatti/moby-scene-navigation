from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Path to the YAML file
    package_name = 'deep_sort_tracker'
    params_file = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'params.yaml'
    )

    return LaunchDescription([
        Node(
            package=package_name,
            executable='deep_sort_tracker',
            name='deep_sort_tracker',
            parameters=[params_file],
            output='screen'
        )
    ])
