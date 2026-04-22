import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class MOT17ImagePublisher(Node):
    def __init__(self):
        super().__init__('mot17_image_publisher')

        # Declare parameters
        self.declare_parameter('dataset_path', '/home/ganesh/llm-bench/.venv/data')
        self.declare_parameter('sequence_name', 'img10')

        # Get parameters
        dataset_path = self.get_parameter('dataset_path').get_parameter_value().string_value
        sequence_name = self.get_parameter('sequence_name').get_parameter_value().string_value

        # Set up paths
        self.img_dir = os.path.join(dataset_path, sequence_name, 'img10')
        if not os.path.exists(self.img_dir):
            self.get_logger().error(f"Image directory does not exist: {self.img_dir}")
            raise FileNotFoundError(f"Image directory does not exist: {self.img_dir}")

        # Load image files
        self.image_files = sorted(os.listdir(self.img_dir))
        if not self.image_files: 
            self.get_logger().error(f"No images found in directory: {self.img_dir}")
            raise ValueError(f"No images found in directory: {self.img_dir}")

        self.current_frame = 0
        self.sequence_number = 0
        self.bridge = CvBridge()

        # Publisher
        self.image_pub = self.create_publisher(Image, 'image_raw', 10)

        # Timer to publish images at 30 FPS
        self.timer = self.create_timer(1.0 / 10.0, self.publish_image)

        self.get_logger().info(f"Publishing images from {self.img_dir} on 'image_raw' topic")

    def publish_image(self):
        if self.current_frame >= len(self.image_files):
            self.get_logger().info("Finished publishing all images.")
            self.destroy_node()
            return

        # Read the current image
        img_path = os.path.join(self.img_dir, self.image_files[self.current_frame])
        img = cv2.imread(img_path)
        if img is None:
            self.get_logger().error(f"Failed to read image: {img_path}")
            self.current_frame += 1
            return

        # Convert and publish the image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')

            # Add header information
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = f"{self.current_frame + 1}"  # Adding sequence name and frame number

            # Publish the message
            self.image_pub.publish(img_msg)
            self.get_logger().info(f"Published frame {self.current_frame + 1}/{len(self.image_files)}: {self.image_files[self.current_frame]}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert or publish image: {e}")

        # Increment the frame counter
        self.current_frame += 1

def main(args=None):
    rclpy.init(args=args)
    node = MOT17ImagePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
