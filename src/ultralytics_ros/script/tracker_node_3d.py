#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import pyrealsense2 as rs
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from ultralytics_ros.msg import YoloResult
from visualization_msgs.msg import Marker, MarkerArray
import message_filters
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
from rclpy.duration import Duration
import math
import tf_transformations
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose


def create_point_cloud2_msg(header, points):
    """
    Create a sensor_msgs/PointCloud2 message from a (N,3) numpy array of points.
    """
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
         PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
         PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
         PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12  # 3 floats, 4 bytes each
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    return msg

def compute_3d_point_from_roi(depth_image, bbox, intrin, depth_scale=1.0):
    """
    Compute a representative 3D point from the ROI defined by bbox in the depth image.
    This function computes the median depth from the ROI and then deprojects the center pixel.
    
    Parameters:
      depth_image (np.ndarray): Depth image (aligned with the RGB image).
      bbox (tuple): Bounding box as (center_x, center_y, width, height).
      intrin (rs.intrinsics): RealSense intrinsics.
      depth_scale (float): Factor to convert raw depth units to meters.
      
    Returns:
      np.ndarray: 3D point [x, y, z] or None if no valid depth is found.
    """
    center_x, center_y, width, height = bbox
    x_min = int(max(center_x - width / 2, 0))
    x_max = int(min(center_x + width / 2, depth_image.shape[1]))
    y_min = int(max(center_y - height / 2, 0))
    y_max = int(min(center_y + height / 2, depth_image.shape[0]))
    
    roi = depth_image[y_min:y_max, x_min:x_max]
    valid_depths = roi[roi > 0]
    if valid_depths.size == 0:
        return None

    median_depth = np.median(valid_depths) * depth_scale
    pixel = [center_x, center_y]
    point_3d = rs.rs2_deproject_pixel_to_point(intrin, pixel, median_depth)
    return np.array(point_3d)

class Human3DLocator(Node):
    def __init__(self):
        super().__init__('human_3d_locator')
        self.get_logger().info("Initializing Human3DLocator node")
        
        # Topics and parameters
        self.declare_parameter('depth_image_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('yolo_result_topic', '/yolo_result')
        self.declare_parameter('depth_scale', 0.001)  # e.g., convert mm to meters
        # Fixed bounding box dimensions (in meters)
        self.declare_parameter('fixed_width', 0.6)
        self.declare_parameter('fixed_height', 1.7)
        self.declare_parameter('fixed_depth', 0.4)
        # Ground level (y coordinate in camera frame corresponding to the floor)
        self.declare_parameter('ground_y', 2.0)  # adjust as needed
        self.declare_parameter('detection_frame', 'camera_color_optical_frame')
        # Subsample factor for the ROI point extraction (1 means no subsampling)
        self.declare_parameter('roi_point_skip', 2)
        #self.declare_parameter('use_sim_time', False)

        
        depth_image_topic = self.get_parameter('depth_image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        yolo_result_topic = self.get_parameter('yolo_result_topic').value

        self.bridge = CvBridge()
        self.get_logger().info("Subscribing to topics:\n - Depth Image: {}\n - Camera Info: {}\n - YOLO Result: {}"
                               .format(depth_image_topic, camera_info_topic, yolo_result_topic))
        
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.depth_sub = message_filters.Subscriber(self, Image, depth_image_topic)
        self.cam_info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic)
        self.yolo_sub = message_filters.Subscriber(self, YoloResult, yolo_result_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.cam_info_sub, self.yolo_sub],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.marker_pub = self.create_publisher(MarkerArray, 'human_3d_markers', 10)
        self.detection_cloud_pub = self.create_publisher(PointCloud2, 'detection_point_cloud', 10)
        self.detection3d_pub = self.create_publisher(Detection3DArray, 'detection_3d_array', 10)

        self.get_logger().info("Publishers created for markers and detection point cloud.")
    
    def callback(self, depth_msg, cam_info_msg, yolo_msg):
        self.get_logger().info("Callback triggered with synchronized messages.")
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            self.get_logger().info("Depth image converted successfully. Shape: {}".format(depth_image.shape))
        except Exception as e:
            self.get_logger().error("Failed to convert depth image: {}".format(e))
            return

        intrin = rs.intrinsics()
        intrin.width = cam_info_msg.width
        intrin.height = cam_info_msg.height
        intrin.fx = cam_info_msg.k[0]
        intrin.fy = cam_info_msg.k[4]
        intrin.ppx = cam_info_msg.k[2]
        intrin.ppy = cam_info_msg.k[5]
        intrin.model = rs.distortion.none
        intrin.coeffs = [0, 0, 0, 0, 0]
        self.get_logger().info("Camera intrinsics set: width={}, height={}, fx={}, fy={}, ppx={}, ppy={}"
                               .format(intrin.width, intrin.height, intrin.fx, intrin.fy, intrin.ppx, intrin.ppy))
        
        depth_scale = self.get_parameter('depth_scale').value
        fixed_width = self.get_parameter('fixed_width').value
        fixed_height = self.get_parameter('fixed_height').value
        fixed_depth = self.get_parameter('fixed_depth').value
        ground_y = self.get_parameter('ground_y').value
        roi_point_skip = self.get_parameter('roi_point_skip').value

        marker_array = MarkerArray()
        delete_all_marker = Marker()
        delete_all_marker.header = depth_msg.header
        delete_all_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_all_marker)
        marker_id = 0
        
        all_detection_points = []
        detection_frame = self.get_parameter('detection_frame').value
        target_frame = depth_msg.header.frame_id

        detection3d_array = Detection3DArray()
        detection3d_array.header = depth_msg.header

        self.get_logger().info("Processing {} detections from YOLO result.".format(len(yolo_msg.detections.detections)))
        for idx, detection in enumerate(yolo_msg.detections.detections):
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y
            bbox = (center_x, center_y, width, height)
            self.get_logger().info("Detection {}: bbox center=({:.2f}, {:.2f}), size=({:.2f}, {:.2f})"
                                   .format(idx, center_x, center_y, width, height))
            
            point_3d = compute_3d_point_from_roi(depth_image, bbox, intrin, depth_scale)
            if point_3d is None:
                self.get_logger().warn("No valid depth for detection {} with bbox: {}".format(idx, bbox))
                continue

            self.get_logger().info("Detection {}: Computed 3D point: x={:.2f}, y={:.2f}, z={:.2f}"
                                   .format(idx, point_3d[0], point_3d[1], point_3d[2]))
            
            point_stamped = PointStamped()
            point_stamped.header.frame_id = detection_frame
            point_stamped.header.stamp = depth_msg.header.stamp
            point_stamped.point.x = point_3d[0]
            point_stamped.point.y = point_3d[1]
            point_stamped.point.z = point_3d[2]
            
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    detection_frame,
                    point_stamped.header.stamp,
                    timeout=Duration(seconds=1.0))
                transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
                point_3d_transformed = np.array([transformed_point.point.x,
                                                 transformed_point.point.y,
                                                 transformed_point.point.z])
                self.get_logger().info("Detection {}: Transformed 3D point (in {}): x={:.2f}, y={:.2f}, z={:.2f}"
                                       .format(idx, target_frame, point_3d_transformed[0],
                                               point_3d_transformed[1], point_3d_transformed[2]))
            except Exception as e:
                self.get_logger().error("Failed to transform detection {}: {}".format(idx, e))
                continue    
            
            # Use the transformed point for the marker.

            qx, qy, qz, qw = tf_transformations.quaternion_from_euler(-math.pi/2, 0, 0)  

            marker = Marker()
            marker.header = depth_msg.header
            marker.ns = "human_3d_box"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = point_3d_transformed[0]
            marker.pose.position.y = point_3d_transformed[1]
            marker.pose.position.z = point_3d_transformed[2]

            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw
            
            marker.scale.x = fixed_width
            marker.scale.y = fixed_depth
            marker.scale.z = fixed_height

            marker.color.a = 0.8
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            self.get_logger().info("Detection {}: Fixed 3D bounding box marker created.".format(idx))
            marker_array.markers.append(marker)

            # Publish a Detection3D message for this detection.
            detection_msg = Detection3D()
            # Use the transformed 3D point as the center (you can adjust if needed)
            detection_msg.bbox.center.position.x = point_3d_transformed[0]
            detection_msg.bbox.center.position.y = point_3d_transformed[1]
            detection_msg.bbox.center.position.z = point_3d_transformed[2]
            # Use the computed quaternion for the bounding box orientation
            detection_msg.bbox.center.orientation.x = qx
            detection_msg.bbox.center.orientation.y = qy
            detection_msg.bbox.center.orientation.z = qz
            detection_msg.bbox.center.orientation.w = qw
            # Set the fixed bounding box size
            detection_msg.bbox.size.x = fixed_width
            detection_msg.bbox.size.y = fixed_depth
            detection_msg.bbox.size.z = fixed_height
            # Create a hypothesis using YOLO detection results (assuming at least one result exists)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(detection.results[0].hypothesis.class_id)
            hypothesis.hypothesis.score = detection.results[0].hypothesis.score
            detection_msg.results.append(hypothesis)
            detection3d_array.detections.append(detection_msg)

            # Extract ROI points with subsampling.
            x_min = int(max(center_x - width / 2, 0))
            x_max = int(min(center_x + width / 2, depth_image.shape[1]))
            y_min = int(max(center_y - height / 2, 0))
            y_max = int(min(center_y + height / 2, depth_image.shape[0]))
            
            detection_points_count = 0
            for v in range(y_min, y_max):
                # Skip rows based on the subsampling factor.
                if (v - y_min) % roi_point_skip != 0:
                    continue
                for u in range(x_min, x_max):
                    if (u - x_min) % roi_point_skip != 0:
                        continue
                    d = depth_image[v, u]
                    if d > 0:
                        d_m = d * depth_scale
                        point = rs.rs2_deproject_pixel_to_point(intrin, [u, v], d_m)
                        all_detection_points.append(point)
                        detection_points_count += 1
            self.get_logger().info("Detection {}: Extracted {} valid 3D points from ROI.".format(idx, detection_points_count))
        
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)-1} bounding box markers (excluding delete-all marker).")
        
        # After processing all detections:
        if all_detection_points:
            points_np = np.array(all_detection_points)
            detection_cloud_msg = create_point_cloud2_msg(depth_msg.header, points_np)
            self.detection_cloud_pub.publish(detection_cloud_msg)
            self.get_logger().info(f"Published detection point cloud with {points_np.shape[0]} points.")
        else:
            # Create and publish an empty point cloud to clear any previous detection clouds.
            empty_points = np.empty((0, 3))
            detection_cloud_msg = create_point_cloud2_msg(depth_msg.header, empty_points)
            self.detection_cloud_pub.publish(detection_cloud_msg)
            self.get_logger().info("Published empty detection point cloud to clear previous detections.")
    

def main(args=None):
    rclpy.init(args=args)
    node = Human3DLocator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down node.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
