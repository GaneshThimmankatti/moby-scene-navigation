import launch
from launch import LaunchDescription
from launch.actions import LogInfo, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare configurable parameters.
    configurable_parameters = [
        DeclareLaunchArgument(
            "trajectory_topic",
            default_value="/plan",
            description="Topic to subscribe for trajectory messages."
        ),
        DeclareLaunchArgument(
            "detection_topic",
            default_value="/detection_3d_array",
            description="Topic to subscribe for detection messages."
        ),
        DeclareLaunchArgument(
            "cmd_vel_topic",
            default_value="/moby_hardware_controller/cmd_vel",
            description="Topic to subscribe for cmd_vel messages."
        ),
        DeclareLaunchArgument(
            "target_frame",
            default_value="base_link",
            description="Target frame for transforming detections."
        ),
        DeclareLaunchArgument(
            "max_detection_distance",
            default_value="4.0",
            description="Max distance (m) to consider a detection for interaction."
        )
    ]

    # Define the trajectory watcher node with debug log level.
    trajectory_watcher_node = Node(
        package='trajectory_watcher',
        executable='trajectory_watcher_node',
        name='trajectory_watcher_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'trajectory_topic': LaunchConfiguration('trajectory_topic')},
            {'detection_topic': LaunchConfiguration('detection_topic')},
            {'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic')},
            {'target_frame': LaunchConfiguration('target_frame')},
            {'max_detection_distance': LaunchConfiguration('max_detection_distance')},
        ],
        arguments=['--ros-args', '--log-level', 'trajectory_watcher:=debug']
    )

    ld = LaunchDescription(configurable_parameters)
    ld.add_action(LogInfo(msg="Launching Trajectory Watcher Node with DEBUG log level..."))
    ld.add_action(trajectory_watcher_node)
    return ld
