from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='human_3d_locator',
            executable='human_3d_locator',
            name='human_3d_locator',
            output='screen',
            parameters=[{
                'depth_image_topic': '/camera/camera/depth/image_rect_raw',
                'camera_info_topic': '/camera/camera/depth/camera_info',
                'yolo_result_topic': '/yolo_result',
                'depth_scale': 0.001,
                'fixed_width': 0.6,
                'fixed_height': 1.7,
                'fixed_depth': 0.4,
                'detection_frame': 'camera_color_optical_frame',
                'roi_point_skip': 2,
            }],
        ),
    ])
