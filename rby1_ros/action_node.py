import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32, Float32, Bool, Int32MultiArray, Float32MultiArray
from rby1_interfaces.msg import EEpos, FTsensor, State, Command, Action
from rby1_interfaces.srv import MetaInitialReq, MetaDataReq

from rby1_ros.qos_profiles import qos_state_latest, qos_cmd, qos_ctrl_latched, qos_image_stream
from rby1_ros.main_status import MainStatus as MainState

import cv2
from cv_bridge import CvBridge


class ActionNode(Node):
    def __init__(self):
        super().__init__('command_node')

        self.main_state = MainState()
    
        self.rby1_sub = self.create_subscription(
            State,
            '/rby1/state',
            self.state_callback,
            qos_state_latest
        )

        self.action_pub = self.create_publisher(
            Action,
            '/control/action',
            qos_cmd
        )

        self.done_sub = self.create_subscription(
            Bool,
            '/control/done',
            self.done_callback,
            qos_cmd
        )

        self.wrist_img_sub = self.create_subscription(
            Image,
            '/camera/right_wrist/image_raw',
            self.wrist_image_callback,
            qos_image_stream
        )

        self.external_img_sub = self.create_subscription(
            Image,
            '/camera/external/image_raw',
            self.external_image_callback,
            qos_image_stream
        )

        self.bridge = CvBridge()

        self.mode_list: list[str] = [
            "image",
            "left_pos",
            "right_pos",
            "both_pos",
            "left_rot_local",
            "right_rot_local",
            "both_rot_local",
            "left_rot_global",
            "right_rot_global",
            "both_rot_global",
            "left_gripper",
            "right_gripper"
        ]

    # ----- /rby1/state 콜백 -----
    def state_callback(self, msg: State):
        self.latest_state_msg = msg
        self.main_state.is_robot_connected = True

        self.main_state.is_robot_initialized = msg.is_initialized
        self.main_state.is_robot_stopped = msg.is_stopped
        
        self.main_state.current_joint_positions = np.array(msg.joint_positions.data)
        self.main_state.current_left_arm_angle = self.main_state.current_joint_positions[15:22]
        self.main_state.current_right_arm_angle = self.main_state.current_joint_positions[8:15]
        
        self.main_state.current_torso_position = np.array(msg.torso_ee_pos.position.data)
        self.main_state.current_torso_quaternion = np.array(msg.torso_ee_pos.quaternion.data)
        self.main_state.current_right_arm_position = np.array(msg.right_ee_pos.position.data)
        self.main_state.current_right_arm_quaternion = np.array(msg.right_ee_pos.quaternion.data)
        self.main_state.current_left_arm_position = np.array(msg.left_ee_pos.position.data)
        self.main_state.current_left_arm_quaternion = np.array(msg.left_ee_pos.quaternion.data)

        self.main_state.current_right_gripper_position = msg.right_gripper_pos
        self.main_state.current_left_gripper_position = msg.left_gripper_pos

    def wrist_image_callback(self, msg):
        # 이미지 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # 이미지 처리 로직 추가 가능
        self.get_logger().info('Received wrist image of size: {}x{}'.format(cv_image.shape[1], cv_image.shape[0]))

    def external_image_callback(self, msg):
        # 이미지 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # 이미지 처리 로직 추가 가능
        self.get_logger().info('Received external image of size: {}x{}'.format(cv_image.shape[1], cv_image.shape[0]))

    def done_callback(self, msg):
        if msg.data:
            self.get_logger().info('Received done signal from action executor.')

    def publish_action(self, mode, arm=None, param_name=None, param=None):
        if mode not in self.mode_list:
            self.get_logger().error('Invalid mode: {}'.format(mode))
            return
        action_msg = Action()
        action_msg.mode = mode
        if arm is not None:
            action_msg.arm = arm
        if param_name is not None and param is not None:
            action_msg.params[param_name] = param

    def request_image(self):
        self.publish_action(mode="image")

    def move_ee_delta_pos(self, arm, delta_pos):
        mode = f"{arm}_pos" if arm in ["left", "right"] else "both_pos"
        self.publish_action(mode=mode, arm=arm, param_name="delta_pos", param=delta_pos)

    def move_ee_delta_rot(self, arm, delta_rot, axis, type="global"):
        mode = f"{arm}_rot_{type}" if arm in ["left", "right"] else f"both_rot_{type}"
        rot_vec = np.zeros(3)
        if axis == 'x':
            rot_vec[0] = delta_rot
        elif axis == 'y':
            rot_vec[1] = delta_rot
        elif axis == 'z':
            rot_vec[2] = delta_rot
        else:
            self.get_logger().error('Invalid axis: {}'.format(axis))
            return
        self.publish_action(mode=mode, arm=arm, param_name="delta_rot", param=rot_vec)

    def right_gripper_open(self):
        self.publish_action(mode="right_gripper", param_name="command", param=True)

    def left_gripper_open(self):
        self.publish_action(mode="left_gripper", param_name="command", param=True)

    def right_gripper_close(self):
        self.publish_action(mode="right_gripper", param_name="command", param=False)

    def left_gripper_close(self):
        self.publish_action(mode="left_gripper", param_name="command", param=False)