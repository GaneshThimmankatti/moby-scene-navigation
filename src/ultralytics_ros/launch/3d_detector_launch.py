#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    depth_image_topic_arg = DeclareLaunchArgument(
        'depth_image_topic',
        default_value='/camera/camera/depth/image_rect_raw',
        description='Depth image topic'
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera/depth/camera_info',
        description='Camera info topic'
    )
    yolo_result_topic_arg = DeclareLaunchArgument(
        'yolo_result_topic',
        default_value='/yolo_result',
        description='YOLO result topic'
    )
    depth_scale_arg = DeclareLaunchArgument(
        'depth_scale',
        default_value='0.001',
        description='Depth scale (mm to meters)'
    )
    fixed_width_arg = DeclareLaunchArgument(
        'fixed_width',
        default_value='0.6',
        description='Fixed bounding box width for humans'
    )
    fixed_height_arg = DeclareLaunchArgument(
        'fixed_height',
        default_value='1.7',
        description='Fixed bounding box height for humans'
    )
    fixed_depth_arg = DeclareLaunchArgument(
        'fixed_depth',
        default_value='0.4',
        description='Fixed bounding box depth for humans'
    )
    ground_y_arg = DeclareLaunchArgument(
        'ground_y',
        default_value='2.0',
        description='Ground level (y coordinate) in the camera frame'
    )
    detection_frame_arg = DeclareLaunchArgument(
        'detection_frame',
        default_value='camera_color_optical_frame',
        description='Frame in which detections are reported'
    )
    roi_point_skip_arg = DeclareLaunchArgument(
        'roi_point_skip',
        default_value='1',
        description='Subsampling factor for ROI extraction (1 = no subsampling)'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time if running from a rosbag'
    )

    tracker_node_3d = Node(
        package='ultralytics_ros',
        executable='tracker_node_3d.py',
        name='tracker_node_3d',
        output='screen',
        parameters=[{
            'depth_image_topic': LaunchConfiguration('depth_image_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'yolo_result_topic': LaunchConfiguration('yolo_result_topic'),
            'depth_scale': LaunchConfiguration('depth_scale'),
            'fixed_width': LaunchConfiguration('fixed_width'),
            'fixed_height': LaunchConfiguration('fixed_height'),
            'fixed_depth': LaunchConfiguration('fixed_depth'),
            'ground_y': LaunchConfiguration('ground_y'),
            'detection_frame': LaunchConfiguration('detection_frame'),
            'roi_point_skip': LaunchConfiguration('roi_point_skip'),
        }]
    )

    ld = LaunchDescription()
    ld.add_action(depth_image_topic_arg)
    ld.add_action(camera_info_topic_arg)
    ld.add_action(yolo_result_topic_arg)
    ld.add_action(depth_scale_arg)
    ld.add_action(fixed_width_arg)
    ld.add_action(fixed_height_arg)
    ld.add_action(fixed_depth_arg)
    ld.add_action(ground_y_arg)
    ld.add_action(detection_frame_arg)
    ld.add_action(roi_point_skip_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(tracker_node_3d)

    return ld
