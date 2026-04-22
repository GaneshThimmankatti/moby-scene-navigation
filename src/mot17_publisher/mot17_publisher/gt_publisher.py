import rclpy
from rclpy.node import Node
from ultralytics_ros.msg import YoloResult 
from vision_msgs.msg import Detection2D, Detection2DArray, Point2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pandas as pd
import cv2
import os


class GTPublisherWithImages(Node):
    def __init__(self):
        super().__init__('gt_publisher_node')

        # Parameters
        self.declare_parameter('gt_file_path', '/home/ganesh/TrackEval/data/gt/mot_challenge/MOT17-train/MOT17-13-FRCNN/gt/gt.txt')
        self.declare_parameter('image_dir', '/home/ganesh/TrackEval/data/gt/mot_challenge/MOT17-train/MOT17-13-FRCNN/img1')  # Directory with MOT17 frames
        self.declare_parameter('publish_rate', 10.0)  # Simulated frame rate
        self.declare_parameter('detection_topic', '/yolo_result')
        self.declare_parameter('image_topic', '/image_raw')

        # Load parameters
        self.gt_file_path = self.get_parameter('gt_file_path').get_parameter_value().string_value
        self.image_dir = self.get_parameter('image_dir').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        # Load GT detections
        self.gt_data = self.load_gt_data(self.gt_file_path)
        self.current_frame = 1

        # Publishers
        self.detection_pub = self.create_publisher(YoloResult, self.detection_topic, 10)
        self.image_pub = self.create_publisher(Image, self.image_topic, 10)

        # Timer for publishing detections and images
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_frame)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        self.get_logger().info("GT Publisher with Images initialized.")

    def load_gt_data(self, gt_file_path):
        """
        Load ground truth detections from gt.txt.
        """
        try:
            gt_data = pd.read_csv(
                gt_file_path,
                header=None,
                names=[
                    'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x_world', 'y_world'
                ]
            )
            self.get_logger().info(f"Loaded {len(gt_data)} detections from {gt_file_path}")
            return gt_data
        except Exception as e:
            self.get_logger().error(f"Failed to load GT file: {e}")
            return pd.DataFrame()

    def publish_frame(self):
        """
        Publish the current frame's detections and image.
        """
        if self.current_frame > self.gt_data['frame'].max():
            self.get_logger().info("All frames processed. Stopping publisher.")
            self.timer.cancel()  # Stop the timer
            return

        # Publish detections for the current frame
        self.publish_detections()

        # Publish the corresponding image for the current frame
        self.publish_image()

        # Increment to the next frame
        self.current_frame += 1

        # Loop back to the first frame if we reach the end of the GT data
        #if self.current_frame > self.gt_data['frame'].max():
        #    self.get_logger().info("Reached the end of the GT data. Looping back to frame 1.")
        #    self.current_frame = 1

    def publish_detections(self):
        """
        Publish detections for the current frame as YoloResult.
        """
        # Filter detections for the current frame
        frame_data = self.gt_data[self.gt_data['frame'] == self.current_frame]

        if frame_data.empty:
            self.get_logger().debug(f"No detections for frame {self.current_frame}.")
            return

        # Prepare YoloResult message
        yolo_result = YoloResult()
        yolo_result.header = Header()
        yolo_result.header.stamp = self.get_clock().now().to_msg()
        yolo_result.header.frame_id = str(self.current_frame)  # Use the frame number as frame_id

        for _, row in frame_data.iterrows():
            # Convert bounding box to YOLO format (center_x, center_y, width, height)
            x_center = row['bb_left'] + (row['bb_width'] / 2)
            y_center = row['bb_top'] + (row['bb_height'] / 2)
            width = row['bb_width']
            height = row['bb_height']
            confidence = row['conf']  # Assuming confidence is already in the GT

            # Create a single detection
            detection = Detection2D()
            detection.header = yolo_result.header
            detection.bbox.center.position.x = float(x_center)
            detection.bbox.center.position.y = float(y_center)
            detection.bbox.size_x = float(width)
            detection.bbox.size_y = float(height)

            # Create ObjectHypothesisWithPose
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = "person"  # Assuming all GT objects are persons
            hypothesis.hypothesis.score = float(confidence)

            # Add hypothesis to detection
            detection.results.append(hypothesis)

            # Add detection to the YoloResult detections
            yolo_result.detections.detections.append(detection)

    # Publish detections
        self.detection_pub.publish(yolo_result)
        self.get_logger().info(f"Published {len(yolo_result.detections.detections)} detections for frame {self.current_frame}")




    def publish_image(self):
        """
        Publish the corresponding image for the current frame.
        """
        # Construct the image file path
        image_file = os.path.join(self.image_dir, f"{self.current_frame:06d}.jpg")

        # Check if the image exists
        if not os.path.exists(image_file):
            self.get_logger().error(f"Image not found: {image_file}")
            return

        # Read and publish the image
        image = cv2.imread(image_file)
        if image is None:
            self.get_logger().error(f"Failed to load image: {image_file}")
            return

        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = str(self.current_frame)  # Use the frame number as frame_id
            self.image_pub.publish(image_msg)
            self.get_logger().info(f"Published image for frame {self.current_frame}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image to ROS message: {e}")




def main(args=None):
    rclpy.init(args=args)
    node = GTPublisherWithImages()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
