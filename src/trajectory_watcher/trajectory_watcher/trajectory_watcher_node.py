import rclpy
from rclpy.node import Node
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Twist, TwistStamped
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from shapely.geometry import LineString, Point
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from tf2_geometry_msgs import do_transform_point
from rclpy.duration import Duration
from std_msgs.msg import Bool


class TrajectoryWatcher(Node):
    def __init__(self):
        super().__init__('trajectory_watcher')
        
        self.declare_parameter('trajectory_topic', '/plan')
        self.declare_parameter('detection_topic', '/detection_3d_array')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('max_detection_distance', 10.0)
        self.max_dist = self.get_parameter('max_detection_distance') \
                   .get_parameter_value().double_value

        trajectory_topic = self.get_parameter('trajectory_topic').get_parameter_value().string_value
        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        cmd_vel_topic   = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        
        # Subscribe to the trajectory topic (change topic name if necessary)
        self.trajectory_sub = self.create_subscription(
            Path,
            trajectory_topic,
            self.trajectory_callback,
            10)
        self.get_logger().info("Subscribed to 'trajectory_topic'.")
        # Subscribe to the 3D detections topic
        self.detection_sub = self.create_subscription(
            Detection3DArray,
            detection_topic,
            self.detection_callback,
            10)
        self.get_logger().info("Subscribed to '/detection_3d_array'.")

        self.cmd_vel_sub = self.create_subscription(
            TwistStamped,
            cmd_vel_topic,
            self.cmd_vel_callback,
            10)
        self.get_logger().info("Subscribed to 'cmd_vel'.")
        # This variable will store the buffered polygon created from the trajectory

        self.human_detected_pub = self.create_publisher(
            Bool, '/is_human_detected', 10)
        
                # Publisher for the first 5 m of the trajectory in base_link
        self.trajectory_ahead_pub = self.create_publisher(
            Path, '/trajectory_ahead', 10)

        self.trajectory_polygon = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info("TF2 buffer and listener created.")

    def trajectory_callback(self, msg: Path):
        self.get_logger().info("Received trajectory message.")
        if not msg.poses:
            self.get_logger().warn("Trajectory message contains no poses.")
            return

        points = []
        latest_time = rclpy.time.Time()  # zero → ask TF for the latest available

        # 1) Transform every PoseStamped into base_link
        for i, pose_stamped in enumerate(msg.poses):
            pose_stamped.header.stamp = latest_time.to_msg()

            if not self.tf_buffer.can_transform(
                self.target_frame,
                pose_stamped.header.frame_id,
                latest_time,
                timeout=Duration(seconds=0.1)
            ):
                self.get_logger().warn(f"TF not available for pose {i}, skipping...")
                continue

            try:
                transformed_pose = self.tf_buffer.transform(
                    pose_stamped,
                    self.target_frame,
                    timeout=Duration(seconds=0.5)
                )
            except (LookupException, ExtrapolationException) as ex:
                self.get_logger().error(f"TF transformation failed for pose {i}: {ex}")
                continue

            pos = transformed_pose.pose.position
            points.append((pos.x, pos.y))
            self.get_logger().debug(f"Transformed Pose {i}: ({pos.x:.2f}, {pos.y:.2f})")

        if len(points) < 2:
            self.get_logger().warn("Not enough points to create a valid trajectory line.")
            return

        # ─── New: Publish only the first 5 m of the path in base_link frame ───
        seg_pts = []
        dist_accum = 0.0
        max_ahead = 5.0
        seg_pts.append(points[0])
        for prev, curr in zip(points, points[1:]):
            d = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
            if dist_accum + d > max_ahead:
                break
            seg_pts.append(curr)
            dist_accum += d

        ahead_path = Path()
        ahead_path.header.frame_id = self.target_frame
        ahead_path.header.stamp = self.get_clock().now().to_msg()
        for x, y in seg_pts:
            ps = PoseStamped()
            ps.header = ahead_path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            ahead_path.poses.append(ps)

        self.trajectory_ahead_pub.publish(ahead_path)
        # ────────────────────────────────────────────────────────────────

        # 2) Build the full 1 m buffer polygon around the entire path
        line = LineString(points)
        polygon = line.buffer(1.0)
        self.trajectory_polygon = polygon
        self.get_logger().info("Trajectory polygon updated successfully in base_link frame.")

    
    def detection_callback(self, msg: Detection3DArray):
        self.get_logger().info("Received detection message.")
        if self.trajectory_polygon is None:
            self.get_logger().warn("Trajectory polygon not yet defined; skipping detection processing.")
            return

        human_detected = False
        self.get_logger().info(f"Processing {len(msg.detections)} detections...")

        for i, detection in enumerate(msg.detections):
            center = detection.bbox.center.position
            self.get_logger().debug(f"Original detection {i} center in detection frame: ({center.x}, {center.y}, {center.z})")

            point_stamped = PointStamped()
            point_stamped.header.stamp = msg.header.stamp
            src = detection.header.frame_id.strip()
            if not src:
                src = 'camera_depth_optical_frame'
            point_stamped.header.frame_id = src
            point_stamped.point.x = center.x
            point_stamped.point.y = center.y
            point_stamped.point.z = center.z

            self.get_logger().debug(
                f"Trying TF transform from '{src}' to '{self.target_frame}' "
                f"at time {point_stamped.header.stamp.sec}.{point_stamped.header.stamp.nanosec}"
             )


            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    src,                         # now guaranteed non‐empty
                    point_stamped.header.stamp,
                    timeout=Duration(seconds=1.0)
                )
                transformed_point_stamped = do_transform_point(point_stamped, transform)
                transformed_point = transformed_point_stamped.point

                dx = transformed_point.x
                dy = transformed_point.y
                dist = (dx*dx + dy*dy)**0.5
                if dist > self.max_dist:
                    self.get_logger().debug(
                        f"Detection {i} at {dist:.1f} m beyond max {self.max_dist:.1f} m, skipping"
                    )
                    continue

                self.get_logger().debug(
                    f"Transformed detection {i} center in {self.target_frame}: "
                    f"({transformed_point.x:.2f}, {transformed_point.y:.2f}, {transformed_point.z:.2f})"
                )
            except (LookupException, ExtrapolationException) as ex:
                self.get_logger().warn(f"Skipping detection {i} due to TF failure: {ex}")
                continue

            detection_point = Point(transformed_point.x, transformed_point.y)
            if self.trajectory_polygon.contains(detection_point):
                human_detected = True
                self.get_logger().info(f"Detection {i} at ({transformed_point.x:.2f}, {transformed_point.y:.2f}) is INSIDE the area.")
                break
            else:
                self.get_logger().info(f"Detection {i} at ({transformed_point.x:.2f}, {transformed_point.y:.2f}) is OUTSIDE the area.")

        self.human_detected_pub.publish(Bool(data=human_detected))

    
    def cmd_vel_callback(self, msg):
        # Try to extract linear and angular components.
        try:
            # Assume msg is of type Twist.
            lin = msg.linear
            ang = msg.angular
        except AttributeError:
            # If AttributeError occurs, assume msg is of type TwistStamped.
            lin = msg.twist.linear
            ang = msg.twist.angular

        #self.get_logger().info(
         #   f"Received cmd_vel: linear=({lin.x:.2f}, {lin.y:.2f}, {lin.z:.2f}), "
          #  f"angular=({ang.x:.2f}, {ang.y:.2f}, {ang.z:.2f})"
        #)
            

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryWatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
