# ultralytics_ros

ROS 2 (Humble) package for real-time object detection and 3D localization using the Ultralytics YOLO model, enabling flexible integration with robotics applications.

This package was forked from [Alpaca-zip/ultralytics_ros](https://github.com/Alpaca-zip/ultralytics_ros) and extended with a depth-image-based 2D→3D detection node (`tracker_node_3d.py`) for use on the MOBY robot platform with an Intel RealSense D455 camera.

## Nodes

| Node | Description |
|------|-------------|
| `tracker_node.py` | Real-time YOLOv8 detection + tracking on ROS 2 image stream, publishes `YoloResult` |
| `tracker_with_cloud_node` (C++) | 3D object detection by fusing 2D detections with LiDAR PointCloud2 via frustum projection |
| `tracker_node_3d.py` | 2D→3D lifting via depth image ROI median + RealSense `rs2_deproject_pixel_to_point` |
| `3d_tracker_node.py` | Alternative PointCloud frustum extractor with DBSCAN clustering (Open3D) |

## Setup

```bash
cd ~/colcon_ws/src
git clone https://github.com/<your-username>/ultralytics_ros.git
python3 -m pip install -r ultralytics_ros/requirements.txt
rosdep install -r -y -i --from-paths .
colcon build --packages-select ultralytics_ros
```

## Usage

### YOLOv8 detection only
```bash
ros2 launch ultralytics_ros tracker.launch.xml input_topic:=/camera/camera/color/image_raw
```

### YOLOv8 + depth-image 3D localization (RealSense D455)
```bash
ros2 launch ultralytics_ros 3d_detector_launch.py
```

### YOLOv8 + LiDAR PointCloud fusion
```bash
ros2 launch ultralytics_ros tracker_with_cloud.launch.xml
```

## Custom Message

```
# ultralytics_ros/YoloResult
std_msgs/Header header
vision_msgs/Detection2DArray detections
sensor_msgs/Image[] masks
```

## Topics (tracker_node_3d)

| Topic | Type | Direction |
|-------|------|-----------|
| `/camera/camera/depth/image_rect_raw` | `sensor_msgs/Image` | Subscribed |
| `/camera/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | Subscribed |
| `/yolo_result` | `ultralytics_ros/YoloResult` | Subscribed |
| `/detection_3d_array` | `vision_msgs/Detection3DArray` | Published |
| `/human_3d_markers` | `visualization_msgs/MarkerArray` | Published |
| `/detection_point_cloud` | `sensor_msgs/PointCloud2` | Published |

## License

Original package: AGPL-3.0 (Alpaca-zip)  
Extensions: Sphaira Medical GmbH / RWTH Aachen University
