#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import open3d as o3d
import tf2_ros
import tf_transformations
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
import message_filters
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial import Delaunay
import pyrealsense2 as rs
from rclpy.duration import Duration


# --- Helper Functions ---
class_map = {"person": 1, "car": 2}

def convert_ros_to_open3d(cloud_msg, point_skip=1):
    points = []
    for i, p in enumerate(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)):
        if i % point_skip == 0:
            points.append([float(p[0]), float(p[1]), float(p[2])])
    pcd = o3d.geometry.PointCloud()
    if points:
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
    return pcd

def downsample_point_cloud(points_np, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    downsampled = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled.points)

def lookup_transform(tf_buffer, target_frame, source_frame, stamp):
    try:
        transform = tf_buffer.lookup_transform(target_frame, source_frame, stamp, timeout=Duration(seconds=1.0))
        return transform
    except Exception as e:
        print("TF lookup failed:", e)
        return None

def transform_point_cloud(points, transform_matrix):
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))
    transformed = (transform_matrix @ points_hom.T).T
    return transformed[:, :3]


def compute_obb_pca(points):
    if points.shape[0] == 0:
        return None, None, None
    center = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    sort_idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sort_idx]
    points_local = (points - center) @ eig_vecs
    min_vals, max_vals = np.min(points_local, axis=0), np.max(points_local, axis=0)
    extent = max_vals - min_vals
    obb_center = center + eig_vecs @ ((min_vals + max_vals) / 2.0)
    return obb_center, extent, eig_vecs

# If you want to use RealSense deprojection:
def camera_info_to_rs_intrinsics(camera_info_msg):
    intrin = rs.intrinsics()
    intrin.width = camera_info_msg.width
    intrin.height = camera_info_msg.height
    intrin.fx = camera_info_msg.k[0]
    intrin.fy = camera_info_msg.k[4]
    intrin.ppx = camera_info_msg.k[2]
    intrin.ppy = camera_info_msg.k[5]
    intrin.model = rs.distortion.none
    intrin.coeffs = [0, 0, 0, 0, 0]
    return intrin

def deproject_pixel_to_point_rs2(pixel, depth, intrin):
    point = rs.rs2_deproject_pixel_to_point(intrin, pixel, depth)
    return np.array(point)

def bbox_2D_to_frustum(bbox_2D, intrin, z_min, z_max, expand_ratio):
    x1, y1, x2, y2 = bbox_2D
    dx = (x2 - x1) * expand_ratio / 2
    dy = (y2 - y1) * expand_ratio / 2
    x1 -= dx
    y1 -= dy 
    x2 += dx 
    y2 += dy
    frustum_points = []
    for z in [z_min, z_max]:
        frustum_points.append(deproject_pixel_to_point_rs2([x1, y1], z, intrin))
        frustum_points.append(deproject_pixel_to_point_rs2([x2, y1], z, intrin))
        frustum_points.append(deproject_pixel_to_point_rs2([x2, y2], z, intrin))
        frustum_points.append(deproject_pixel_to_point_rs2([x1, y2], z, intrin))
    return np.array(frustum_points)

#def filter_points_inside_frustum(point_cloud, frustum_points):
#    """
#    Filter the point cloud (Nx3 numpy array) to keep only the points
#    that are inside the convex hull of the frustum points.
#    """
#    if point_cloud.shape[0] == 0:
#        return np.empty((0, 3))
#    delaunay = Delaunay(frustum_points)
#    inside_mask = delaunay.find_simplex(point_cloud) >= 0
#    return point_cloud[inside_mask]

def extract_closest_cluster_open3d(points_np, eps=0.3, min_points=10):
    """
    Clusters the input 3D points using Open3D's DBSCAN clustering and
    returns the cluster whose centroid is closest to the sensor origin.
    
    :param points_np: (N, 3) numpy array of 3D points.
    :param eps: Maximum distance between two points to be considered in the same cluster.
    :param min_points: Minimum number of points required to form a cluster.
    :return: A numpy array representing the closest cluster (or the original points if clustering fails).
    """
    # Convert numpy array to an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    
    # Run DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    # If no clusters are found (all points marked as noise, i.e. label == -1), return original points
    if labels.max() < 0:
        return points_np

    best_cluster = None
    min_distance = float('inf')
    
    # Iterate through each cluster (ignoring noise with label -1)
    for label in np.unique(labels):
        if label == -1:
            continue
        # Get indices for the current cluster
        indices = np.where(labels == label)[0]
        cluster_points = points_np[indices]
        # Compute centroid of the cluster
        centroid = np.mean(cluster_points, axis=0)
        distance = np.linalg.norm(centroid)
        # Select the cluster with the smallest centroid distance to the sensor origin
        if distance < min_distance:
            min_distance = distance
            best_cluster = cluster_points

    return best_cluster if best_cluster is not None else points_np

def project3d_to_pixel(pt, K):
    """
    Project a 3D point (x,y,z) to 2D pixel coordinates using a pinhole camera model.
    K is a 3x3 intrinsic matrix.
    """
    X, Y, Z = pt
    if Z <= 0:
        return np.array([-1, -1])
    u = (K[0, 0] * X / Z) + K[0, 2]
    v = (K[1, 1] * Y / Z) + K[1, 2]
    return np.array([u, v])

def filter_points_with_bbox(points, bbox, K):
    """
    Filter 3D points by projecting them into 2D and checking if they fall within the given bbox.
    bbox: [center_x, center_y, size_x, size_y] in pixel coordinates.
    K: 3x3 intrinsic matrix.
    """
    cx, cy, size_x, size_y = bbox
    x_min = cx - size_x / 2
    x_max = cx + size_x / 2
    y_min = cy - size_y / 2
    y_max = cy + size_y / 2
    filtered = []
    for pt in points:
        if pt[2] <= 0:
            continue  # skip invalid depth
        uv = project3d_to_pixel(pt, K)
        if x_min <= uv[0] <= x_max and y_min <= uv[1] <= y_max:
            filtered.append(pt)
    if len(filtered) == 0:
        return np.empty((0, 3))
    return np.array(filtered)

def create_point_cloud2_msg(header, points):
    """
    Convert a (N,3) numpy array of points to a sensor_msgs/PointCloud2 message.
    """
    from sensor_msgs.msg import PointCloud2, PointField
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
    msg.point_step = 12
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    return msg


# --- Node Definition ---
class FrustumExtractorNode(Node):
    def __init__(self):
        super().__init__('frustum_extractor_node')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('lidar_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('yolo_result_in_topic', '/yolo_result')
        self.declare_parameter('yolo_result_topic', 'detection_3d_array')
        self.declare_parameter('z_min', 0.5)
        self.declare_parameter('z_max', 10.0)
        self.declare_parameter('expand_ratio', 0.1)
        self.declare_parameter('min_confidence', 0.45)
        self.declare_parameter('point_skip', 1)
        self.declare_parameter('voxel_leaf_size', 0.25)
        self.declare_parameter('cluster_eps', 0.2)
        self.declare_parameter('cluster_min_points', 10)
        self.declare_parameter('use_box_filter', True)
        
        cam_topic = self.get_parameter('camera_info_topic').value
        lidar_topic = self.get_parameter('lidar_topic').value
        yolo_in_topic = self.get_parameter('yolo_result_in_topic').value
        yolo_out_topic = self.get_parameter('yolo_result_topic').value
        
        self.camera_info_sub = message_filters.Subscriber(self, CameraInfo, cam_topic)
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, lidar_topic)
        self.yolo_result_sub = message_filters.Subscriber(self, YoloResult, yolo_in_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_info_sub, self.lidar_sub, self.yolo_result_sub],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        
        self.marker_pub = self.create_publisher(MarkerArray, "detection_marker", 1)
        self.detection3d_pub = self.create_publisher(Detection3DArray, yolo_out_topic, 1)
        self.detection_cloud_pub = self.create_publisher(PointCloud2, "detection_cloud", 1)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
    def sync_callback(self, camera_info_msg, cloud_msg, yolo_result_msg):
        self.get_logger().info("Processing synchronized messages...")
        
        # Use RealSense intrinsics conversion
        intrin = camera_info_to_rs_intrinsics(camera_info_msg)
        K = np.array([[intrin.fx, 0, intrin.ppx],
                      [0, intrin.fy, intrin.ppy],
                      [0, 0, 1]])
        
        # Convert ROS PointCloud2 to Open3D and downsample
        cloud = convert_ros_to_open3d(cloud_msg, self.get_parameter('point_skip').value)
        points_np = np.asarray(cloud.points)
        self.get_logger().info(f"Original point cloud has {points_np.shape[0]} points")
        voxel_size = self.get_parameter('voxel_leaf_size').value
        downsampled_points = downsample_point_cloud(points_np, voxel_size)
        
        # Optional: Transform point cloud to target frame if needed (e.g., "camera_frame")
        target_frame = "camera_color_optical_frame"
        transform_stamped = None
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                target_frame,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().warn(f"Transform lookup failed: {e}")
        if transform_stamped is not None:
            t = transform_stamped.transform.translation
            q = transform_stamped.transform.rotation
            transform_matrix = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            transform_matrix[:3, 3] = [t.x, t.y, t.z]
            downsampled_points = transform_point_cloud(downsampled_points, transform_matrix)
            self.get_logger().info("Transformed point cloud to target frame")

        aggregated_detection_points = []
        
        # Process each YOLO detection
        detection3d_array = Detection3DArray()
        detection3d_array.header = cloud_msg.header
        marker_array = MarkerArray()
        for detection in yolo_result_msg.detections.detections:
            if detection.results[0].hypothesis.score < self.get_parameter('min_confidence').value:
                continue
            
            # Compute 2D bounding box
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            size_x = detection.bbox.size_x
            size_y = detection.bbox.size_y
            x1 = center_x - size_x / 2
            y1 = center_y - size_y / 2
            x2 = center_x + size_x / 2
            y2 = center_y + size_y / 2
            
            # Generate 3D frustum using RealSense deprojection
            if self.get_parameter('use_box_filter').value:
                # Box filter: project each 3D point to 2D and check if inside bbox
                bbox = [center_x, center_y, size_x, size_y]
                filtered_points = filter_points_with_bbox(downsampled_points, bbox, K)
            else:
                # Otherwise, use frustum-based filtering (using deprojection)
                frustum_points = bbox_2D_to_frustum([x1, y1, x2, y2], intrin,
                                                    self.get_parameter('z_min').value,
                                                    self.get_parameter('z_max').value,
                                                    self.get_parameter('expand_ratio').value)
                filtered_points = filter_points_inside_frustum(downsampled_points, frustum_points)
            
            if filtered_points.shape[0] == 0:
                continue
            
            
            # Cluster to extract the relevant object points
            cluster_eps = self.get_parameter('cluster_eps').value
            cluster_min_points = self.get_parameter('cluster_min_points').value
            closest_cluster = extract_closest_cluster_open3d(filtered_points, eps=cluster_eps, min_points=cluster_min_points)
            if closest_cluster.shape[0] == 0:
                continue

            aggregated_detection_points.append(closest_cluster)
            
            # Compute oriented bounding box using PCA
            obb_center, extent, eig_vecs = compute_obb_pca(closest_cluster)
            if obb_center is None:
                continue
            T = np.eye(4)
            T[:3, :3] = eig_vecs
            q = tf_transformations.quaternion_from_matrix(T)
            
            # Create detection message and marker (similar to the C++ node)
            detection_msg = Detection3D()
            detection_msg.bbox.center.position.x = obb_center[0]
            detection_msg.bbox.center.position.y = obb_center[1]
            detection_msg.bbox.center.position.z = obb_center[2]
            detection_msg.bbox.center.orientation.x = q[0]
            detection_msg.bbox.center.orientation.y = q[1]
            detection_msg.bbox.center.orientation.z = q[2]
            detection_msg.bbox.center.orientation.w = q[3]
            detection_msg.bbox.size.x = extent[0]
            detection_msg.bbox.size.y = extent[1]
            detection_msg.bbox.size.z = extent[2]
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(detection.results[0].hypothesis.class_id)
            hypothesis.hypothesis.score = detection.results[0].hypothesis.score
            detection_msg.results.append(hypothesis)
            detection3d_array.detections.append(detection_msg)
            
            # Create an RViz marker for visualization
            marker = Marker()
            marker.header = cloud_msg.header
            marker.ns = "detection"
            marker.id = class_map.get(detection.results[0].hypothesis.class_id, 0)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = obb_center[0]
            marker.pose.position.y = obb_center[1]
            marker.pose.position.z = obb_center[2]
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            marker.scale.x = extent[0]
            marker.scale.y = extent[1]
            marker.scale.z = extent[2]
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 0.5
            marker_array.markers.append(marker)
        
        if aggregated_detection_points:
            aggregated_points = np.vstack(aggregated_detection_points)
            detection_cloud_msg = create_point_cloud2_msg(cloud_msg.header, aggregated_points)
            self.detection_cloud_pub.publish(detection_cloud_msg)
        
        # Publish your messages
        self.detection3d_pub.publish(detection3d_array)
        self.marker_pub.publish(marker_array)
        # Optionally publish a PointCloud2 for the detections as well

def main(args=None):
    rclpy.init(args=args)
    node = FrustumExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
