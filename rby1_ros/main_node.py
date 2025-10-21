import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32, Bool, Int32MultiArray, Float32MultiArray

class MainNode(Node):
    def __init__(self):
        super().__init__('main_node')
        
        self.error_sub = self.create_subscription(
            String,
            '/system/error',
            self.error_callback,
            10)
        
        self.switch_sub = self.create_subscription(
            Int32MultiArray,
            '/control/switch',
            self.switch_callback,
            10)

        self.meta_sub = self.create_subscription(
            Float32MultiArray,
            '/control/meta',
            self.meta_callback,
            10)

        self.action_sub = self.create_subscription(
            String,
            '/control/action',
            self.action_callback,
            10)
        
        self.rby1_sub = self.create_subscription(
            Float32MultiArray,
            '/rby1/data',
            self.rby1_callback,
            10)
        
        self.control_pub = self.create_publisher(
            Float32MultiArray,
            '/control/command',
            10)

    def switch_callback(self, msg):
        self.get_logger().info(f'Received switch data: {msg.data}')

    def meta_callback(self, msg):
        self.get_logger().info(f'Received meta data: {msg.data}')

    def action_callback(self, msg):
        self.get_logger().info(f'Received action data: {msg.data}')

    def error_callback(self, msg):
        self.get_logger().info(f'Received error data: {msg.data}')

    def rby1_callback(self, msg):
        self.get_logger().info(f'Received RBY1 data: {msg.data}')
