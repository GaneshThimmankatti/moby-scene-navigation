import os
import numpy as np
from ament_index_python.packages import get_package_share_directory
import torch
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from ultralytics_ros.msg import YoloResult
from .deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from sensor_msgs.msg import Image
from derived_object_msgs.msg import Object
from geometry_msgs.msg import Pose, Twist, Accel
from shape_msgs.msg import SolidPrimitive


CLASS_ID_MAPPING = {
    'person': 0 
    #'bicycle': 1, 'car': 2, 'motorbike': 3, 'aeroplane': 4,
    #'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9,
    #'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
    #'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19,
    #'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24,
    #'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29,
    #'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
    #'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38,
    #'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43,
    #'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48,
    #'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53,
    #'donut': 54, 'cake': 55, 'chair': 56, 'sofa': 57, 'pottedplant': 58,
    #'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63,
    #'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68,
    #'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73,
    #'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78,
    #'toothbrush': 79
}

class DeepSortTracker(Node):
    def __init__(self):
        super().__init__('deep_sort_tracker_node')

        self.declare_parameter('model_path', '/opt/sphaira/models/ml/ckpt.t7')
        self.declare_parameter('max_dist', 0.2)
        self.declare_parameter('max_iou_distance', 0.7)
        self.declare_parameter('max_age', 70)
        self.declare_parameter('n_init', 3)
        self.declare_parameter('nn_budget', 100)
        self.declare_parameter('time_interval', 1.0 / 30.0)  # Assuming 30 FPS

        self.declare_parameter('detection_topic', '/yolo_result')
        self.declare_parameter('image_topic', 'my_camera/pylon_ros2_camera_node/image_raw')
        self.declare_parameter('tracked_objects_topic', '/tracked_objects')
        self.declare_parameter('annotated_image_topic', '/annotated_image')
        self.declare_parameter('output_file_path', '/output/file/path')
        self.declare_parameter('enable_save_detections', False)

        # Retrieve parameters from YAML, with defaults if missing
        
        self.max_dist = self.get_parameter('max_dist').get_parameter_value().double_value
        self.max_iou_distance = self.get_parameter('max_iou_distance').get_parameter_value().double_value
        self.max_age = self.get_parameter('max_age').get_parameter_value().integer_value
        self.n_init = self.get_parameter('n_init').get_parameter_value().integer_value
        self.nn_budget = self.get_parameter('nn_budget').get_parameter_value().integer_value 
        self.time_interval = self.get_parameter('time_interval').get_parameter_value().double_value  # Assuming 30 FPS
        self.output_file_path = self.get_parameter('output_file_path').get_parameter_value().string_value
        self.enable_save_detections = self.get_parameter('enable_save_detections').get_parameter_value().bool_value



        if self.enable_save_detections:
            os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
            self.output_handle = open(self.output_file_path, 'w')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")




        # Topics for subscribers and publishers
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.tracked_objects_topic = self.get_parameter('tracked_objects_topic').get_parameter_value().string_value
        self.annotated_image_topic = self.get_parameter('annotated_image_topic').get_parameter_value().string_value

        # Locate the model file dynamically
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        # Validate if the model exists
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found at {self.model_path}.")
            raise FileNotFoundError(f"Model file not found at {self.model_path}.")

        
        
        # Initialize Deep SORT
        self.deepsort = DeepSort(
            model_path=self.model_path,
            max_dist=self.max_dist,
            max_iou_distance=self.max_iou_distance,
            max_age=self.max_age,
            n_init=self.n_init,
            nn_budget=self.nn_budget,
            use_cuda=(self.device == 'cuda')
        )

        self.bridge = CvBridge()

        # Track previous positions for velocity computation
        self.previous_positions = {}  # {track_id: (x, y)}
        

        # Placeholder for the latest received image
        self.latest_image = None
        self.draw_image = None

        # Subscribers
        self.detection_sub = self.create_subscription(
            YoloResult, self.detection_topic, self.track_objects, 30
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic,
            self.image_callback, 30
        )

        # Publisher: tracked objects with velocities
        self.tracker_pub = self.create_publisher(
            Object, self.tracked_objects_topic, 10
        )

        self.annotated_image_pub = self.create_publisher(
            Image, self.annotated_image_topic, 10
        )

        self.get_logger().info("DeepSortTracker initialized with parameters:")
        self.log_parameters()

    def log_parameters(self):
        """Log loaded parameters for debugging."""
        self.get_logger().info(f"model_path: {self.model_path}")
        self.get_logger().info(f"max_dist: {self.max_dist}")
        self.get_logger().info(f"max_iou_distance: {self.max_iou_distance}")
        self.get_logger().info(f"max_age: {self.max_age}")
        self.get_logger().info(f"n_init: {self.n_init}")
        self.get_logger().info(f"nn_budget: {self.nn_budget}")
        self.get_logger().info(f"time_interval: {self.time_interval}")
        self.get_logger().info(f"detection_topic: {self.detection_topic}")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"tracked_objects_topic: {self.tracked_objects_topic}")

    def image_callback(self, msg: Image):
        """
        Callback to store the latest image from the camera topic.
        """
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def track_objects(self, msg: YoloResult):
        """
        Callback to process YOLO detections, run DeepSORT, and publish tracked objects with velocities.
        """
        
        # Ensure Latest image is availible 
        if self.latest_image is None:
            self.get_logger().error("No image available for tracking. Ensure /image_raw is being published.")
            return
       
        detections = []
        confidences = []
        classes = []

        # Collect bounding boxes, confidences, and classes
        for detection in msg.detections.detections:
            bbox = detection.bbox
            x_center, y_center = bbox.center.position.x, bbox.center.position.y
            w, h = bbox.size_x, bbox.size_y
            confidence = detection.results[0].hypothesis.score
            class_label = detection.results[0].hypothesis.class_id
            class_id = CLASS_ID_MAPPING.get(class_label, -1)

            
            if class_id == -1:
                self.get_logger().warning(f"Unknown class label '{class_label}'. Skipping detection.")
                continue
            
            detections.append([x_center, y_center, w, h])
            confidences.append(confidence)
            classes.append(class_id)

        detections = np.array(detections)
        confidences = np.array(confidences)
        classes = np.array(classes)

        if detections.size == 0:
            self.get_logger().debug("No detections received.")
            # Publish the latest image without annotations.
            if self.latest_image is not None:
                self.annotate_and_publish_image([], self.latest_image)
            return


        

        # Run DeepSORT tracking
        self.draw_image = self.latest_image.copy()
        try:
            tracked_objects = self.deepsort.update(
                bbox_xywh=detections,
                confidences=confidences,
                classes=classes,
                ori_img=self.draw_image
            )
            tracked_objects = tracked_objects[0]     # Use only the first element as it is a tuble with first element as an array with tracked object data and second element is an empty array
                    # Log outputs from SORT
            self.get_logger().debug("Outputs from SORT:")
            for track in tracked_objects:
                xtl, ytl, xbr, ybr, class_id, track_id = track
                self.get_logger().debug(f"Tracked Object: xtl={xtl}, ytl={ytl}, xbr={xbr}, ybr={ybr}, class_id={class_id}, track_id={track_id}")

            self.calculate_and_publish_velocities(tracked_objects, msg.header)
        
        except Exception as e:
            self.get_logger().error(f"Error during DeepSORT update: {e}\n"
                                    f"Inputs: bbox_xywh={detections}, confidences={confidences}, classes={classes}."
            )

    def annotate_and_publish_image(self, tracked_objects_with_velocities, frame):
        """
        Draw bounding boxes with track_id and velocities on the image and publish it.
        """
        if frame is None:
            self.get_logger().warning("No frame available for annotation.")
            return

        # Create a copy of the frame to annotate
        annotated_image = frame.copy()

        for track in tracked_objects_with_velocities:

            xtl, ytl, xbr, ybr, class_id, track_id, velocity_x, velocity_y = track

             #Ensure bounding box is within frame dimensions
            x1 = max(0, int(xtl))
            y1 = max(0, int(ytl))
            x2 = min(frame.shape[1] - 1, int(xbr))
            y2 = min(frame.shape[0] - 1, int(ybr))

            xc= (x1+x2)//2 
            yc= (y1+y2)//2
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Overlay text for track_id and velocities
            label = f"ID: {track_id} Vx: {velocity_x:.2f} Vy: {velocity_y:.2f}"
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            scale = 1  # Adjust arrow length scaling factor
            arrow_x = int(xc + scale * velocity_x)
            arrow_y = int(yc + scale * velocity_y)
    
            if abs(velocity_x) > 0.1 or abs(velocity_y) > 0.1:  # Draw only if moving
                cv2.arrowedLine(annotated_image, (xc, yc), (arrow_x, arrow_y), (0, 0, 255), 2, tipLength=0.3)

        # Convert the annotated image back to ROS Image and publish
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            self.annotated_image_pub.publish(annotated_msg)
            self.get_logger().debug("Annotated image published.")
        except Exception as e:
            self.get_logger().error(f"Failed to convert annotated image to ROS message: {e}")



    
    
    def calculate_and_publish_velocities(self, tracked_objects, input_header):
        """
        Calculate velocities of tracked objects and publish the data.
        """
        tracked_objects_with_velocities = []  # List to hold Object messages for annotation and publishing
    
        for track in tracked_objects:
            if len(track) < 6:
                self.get_logger().error(f"Invalid track data: {track}")
                continue
            
            xtl, ytl, xbr, ybr, class_id, track_id = track
            
            #convert (xtl, ytl, xbr, ybr to xc, yc, w, h)
            xc = (xtl + xbr) / 2
            yc = (ytl + ybr) / 2
            w = xbr - xtl
            h = ybr - ytl

            velocity_x, velocity_y = 0.0, 0.0
    
            # Calculate velocity if previous position exists
            if track_id in self.previous_positions:
                prev_x, prev_y = self.previous_positions[track_id]
                velocity_x = (xc - prev_x) / self.time_interval
                velocity_y = (yc - prev_y) / self.time_interval 
    
            # Update previous position
            self.previous_positions[track_id] = (xc, yc)
    

            tracked_objects_with_velocities.append(
                (xtl, ytl, xbr, ybr, class_id, track_id, velocity_x, velocity_y)
            )
            
            # Prepare Object message
            obj_msg = Object()
            obj_msg.header = input_header
            obj_msg.id = int(track_id)
            obj_msg.detection_level = Object.OBJECT_TRACKED
            obj_msg.object_classified = True
    
            # Pose
            obj_msg.pose.position.x = float(xc)
            obj_msg.pose.position.y = float(yc)
            obj_msg.pose.position.z = 0.0  # Assuming 2D tracking
            obj_msg.pose.orientation.w = 1.0  # Default orientation (no rotation)
    
            # Twist (velocities)
            obj_msg.twist.linear.x = float(velocity_x)
            obj_msg.twist.linear.y = float(velocity_y)
            obj_msg.twist.linear.z = 0.0
    
            # Acceleration (optional, assuming 0 for now)
            obj_msg.accel.linear.x = 0.0
            obj_msg.accel.linear.y = 0.0
            obj_msg.accel.linear.z = 0.0
    
            # Shape (bounding box dimensions)
            obj_msg.shape.type = SolidPrimitive.BOX
            obj_msg.shape.dimensions = [float(w), float(h), 1.0]  # Assuming a 2D object with unit depth
    
            # Classification
            obj_msg.classification = int(class_id)
            obj_msg.classification_certainty = 255  # Max certainty
            obj_msg.classification_age = 1  # Assuming first detection (increment this if tracking persists)


            if self.enable_save_detections:
                frame_number = input_header.frame_id
                self.save_detections_to_file (
                    frame= frame_number,
                    track_id= track_id,
                    bb_left=xtl + 1,
                    bb_top=ytl + 1,
                    bb_w=w,
                    bb_h=h,
                    score= -1,
                    x_world= -1,
                    y_world= -1,
                    z_world= -1
                )
        
    
            # Publish the object message
            self.tracker_pub.publish(obj_msg)
            self.get_logger().debug(f"Published object: ID={obj_msg.id}, Class={obj_msg.classification}, "
                                   f"Velocities=({obj_msg.twist.linear.x}, {obj_msg.twist.linear.y})")
    
        # Annotate and publish the image with bounding boxes and IDs
        if self.draw_image is not None:
            self.annotate_and_publish_image(tracked_objects_with_velocities, self.draw_image)
        else:
            self.get_logger().warning("No image available for annotation.")


    def save_detections_to_file(self, frame, track_id, bb_left, bb_top, bb_w, bb_h, score, x_world, y_world, z_world):
        """
        Save a single detection to the output file in TrackEval format.
        """
        line = f"{frame},{track_id},{bb_left:.2f},{bb_top:.2f},{bb_w:.2f},{bb_h:.2f},{score:.2f},{x_world:.2f},{y_world},{z_world:.2f}\n"
        self.output_handle.write(line)
        self.output_handle.flush()  # Ensure the data is written to the file
        self.get_logger().info(f"Saved detection: {line.strip()}")

    def destroy_node(self):
        if self.enable_save_detections:
            self.output_handle.close()
        super().destroy_node()
       
             
            

def main(args=None):
    rclpy.init(args=args)
    node = DeepSortTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
