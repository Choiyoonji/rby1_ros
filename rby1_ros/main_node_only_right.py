import numpy as np
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32, Bool, Int32MultiArray, Float32MultiArray
from rby1_interfaces.msg import EEpos, FTsensor, State, Command
from rby1_interfaces.srv import MetaInitialReq, MetaDataReq
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
)

# 1) 제어/설정: 마지막 값이 반드시 전달되어야 함 (라치드)
qos_ctrl_latched = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # 퍼블리셔가 캐시
)

# 2) 주기 신호(명령/행동): 잠깐의 지터 흡수, 최신 위주
qos_cmd = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,  # 5~20 사이 권장 (너비 큰 I/O는 줄이기)
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# 3) 상태: 최신값만 필요 → 깊이 1
qos_state_latest = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

import cv2
from rby1_ros.main_status import MainStatus as MainState
from rby1_ros.rby1_dyn import RBY1Dyn
from rby1_ros.utils import *

class MainNode(Node):
    def __init__(self):
        super().__init__('main_node_only_right')

        self.ready: bool = False
        self.move: bool = False
        self.stop: bool = False
        self.record: bool = False

        self.main_state = MainState()

        # /record : 제어 토글 → 라치드 + Reliable (마지막 값 보장)
        self.record_pub = self.create_publisher(
            Bool,
            '/record',
            qos_ctrl_latched
        )

        # /control/action : 외부에서 들어오는 행동 벡터 → 주기 신호
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            '/control/action',
            self.action_callback,
            qos_cmd
        )

        # /rby1/state : 로봇 상태 → 최신값만
        self.rby1_sub = self.create_subscription(
            State,
            '/rby1/state',
            self.rby1_callback,
            qos_state_latest
        )

        # /control/command : 컨트롤 명령 주기 발행
        self.command_pub = self.create_publisher(
            Command,
            '/control/command',
            qos_cmd
        )

        self.main_timer = self.create_timer(1/100.0, self.main_loop)
        self.command_timer = self.create_timer(1/20.0, self.publish_command)
        
        self.command_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
        cv2.namedWindow("control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("control", 480, 640)

    def action_callback(self, msg):
        self.get_logger().info(f'Received action data: {msg.data}')
        data = msg.data
        self.main_state.is_controller_initialized = True
        self.main_state.is_controller_connected = data[0]
        self.main_state.desired_joint_positions = np.array(data[1:8])
        self.main_state.desired_right_gripper_position = data[8]

    def rby1_callback(self, msg):
        # self.get_logger().info(f'Received RBY1 data')
        self.main_state.is_robot_connected = True

        self.main_state.is_robot_initialized = msg.is_initialized
        self.main_state.is_robot_stopped = msg.is_stopped
        
        self.main_state.current_joint_positions = np.array(msg.joint_positions.data)
        
        self.main_state.current_torso_position = np.array(msg.torso_ee_pos.position.data)
        self.main_state.current_torso_quaternion = np.array(msg.torso_ee_pos.quaternion.data)
        self.main_state.current_right_arm_position = np.array(msg.right_ee_pos.position.data)
        self.main_state.current_right_arm_quaternion = np.array(msg.right_ee_pos.quaternion.data)

        self.main_state.current_right_gripper_position = msg.right_gripper_pos

    def publish_command(self):
        if self.main_state.is_robot_connected:
            command_msg = Command()

            command_msg.is_active = self.main_state.is_controller_connected
            command_msg.control_mode = "joint_position"

            command_msg.desired_joint_positions = Float32MultiArray(data=self.main_state.desired_joint_positions.tolist())

            command_msg.desired_right_gripper_pos = self.main_state.desired_right_gripper_position

            command_msg.estop = False

            command_msg.ready = self.ready
            command_msg.move = self.move
            command_msg.stop = self.stop
            
            if command_msg.ready:

                self.stop = False
                self.ready = False

                self.command_pub.publish(command_msg)

    def reset_state(self):
        self.main_state.desired_head_position = np.array([])
        self.main_state.desired_head_quaternion = np.array([])
        self.main_state.desired_torso_position = np.array([])
        self.main_state.desired_torso_quaternion = np.array([])
        self.main_state.desired_right_arm_position = np.array([])
        self.main_state.desired_right_arm_quaternion = np.array([])
        self.main_state.desired_right_gripper_position = 0.0

    def main_loop(self):
        if self.main_state.is_robot_stopped:
            self.reset_state()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.get_logger().info('Received ready command')
            if self.main_state.is_robot_initialized:
                self.get_logger().warning('You must uninitialize the robot before setting ready pose offsets')
            else:
                self.ready = True
                self.main_state.is_robot_in_ready_pose = True
        elif key == ord('f'):
            self.get_logger().info('Received move command')
            self.move = not self.move
        elif key == ord('g'):
            self.get_logger().info('Received stop command')
            self.stop = True
        elif key == ord('q'):
            self.get_logger().info('Received quit command')
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()
        elif key == ord('h'):
            self.get_logger().info('Received record command')
            self.record = not self.record
            self.record_pub.publish(Bool(data=self.record))

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "'r' : Ready Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "'m' : Move Toggle", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "'s' : Stop", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "'c' : Record Toggle", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "'q' : Quit", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        status_text = f"Robot Connected: {self.main_state.is_robot_connected} | Initialized: {self.main_state.is_robot_initialized} | In Ready Pose: {self.main_state.is_robot_in_ready_pose}"
        cv2.putText(img, status_text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("control", img)

def main():
    rclpy.init()
    main_node = MainNode()
    rclpy.spin(main_node)
    main_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()