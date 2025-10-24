import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32, Bool, Int32MultiArray, Float32MultiArray
from rby1_interfaces.msg import EEpos, FTsensor, State, Command
from rby1_interfaces.srv import MetaInitialReq, MetaDataReq

import cv2
from rby1_ros.main_status import MainStatus as MainState

class MainNode(Node):
    def __init__(self):
        super().__init__('main_node')

        self.ready: bool = False
        self.move: bool = False
        self.stop: bool = False
        self.record: bool = False

        self.main_state = MainState()

        self.record_pub = self.create_publisher(
            Bool,
            '/record',
            10)

        self.action_sub = self.create_subscription(
            Float32MultiArray,
            '/control/action',
            self.action_callback,
            10)
        
        self.rby1_sub = self.create_subscription(
            State,
            '/rby1/state',
            self.rby1_callback,
            10)
        
        self.command_pub = self.create_publisher(
            Command,
            '/control/command',
            10)
        
        self.meta_initialize_client = self.create_client(
            MetaInitialReq,
            '/meta/set_offset')
        while not self.meta_initialize_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Meta Initial Offset service not available, waiting...')
        
        self.meta_data_client = self.create_client(
            MetaDataReq,
            '/meta/get_data')
        while not self.meta_data_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Meta Data service not available, waiting...')

        self.init_req = MetaInitialReq.Request()
        self.data_req = MetaDataReq.Request()

        self.main_timer = self.create_timer(1/100.0, self.main_loop)
        self.meta_timer = self.create_timer(1/20.0, self.meta_loop)
        self.command_timer = self.create_timer(1/20.0, self.publish_command)
        
        self.command_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
        cv2.namedWindow("control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("control", 480, 640)

    def send_meta_initial_offset(self, check_pose: bool):
        self.init_req.initialize = True
        self.init_req.check_pose = check_pose

        self.init_req.left_ready_pos = EEpos()
        self.init_req.right_ready_pos = EEpos()
        self.init_req.head_ready_pos = EEpos()

        self.init_req.left_ready_pos.position = Float32MultiArray(data=self.main_state.current_left_arm_position.tolist())
        self.init_req.left_ready_pos.quaternion = Float32MultiArray(data=self.main_state.current_left_arm_quaternion.tolist())

        self.init_req.right_ready_pos.position = Float32MultiArray(data=self.main_state.current_right_arm_position.tolist())
        self.init_req.right_ready_pos.quaternion = Float32MultiArray(data=self.main_state.current_right_arm_quaternion.tolist())

        self.init_req.head_ready_pos.position = Float32MultiArray(data=self.main_state.current_torso_position.tolist())
        self.init_req.head_ready_pos.quaternion = Float32MultiArray(data=self.main_state.current_torso_quaternion.tolist())

        return self.meta_initialize_client.call_async(self.init_req)

    def send_meta_get_data(self):
        self.data_req.request = True
        return self.meta_data_client.call_async(self.data_req)

    def action_callback(self, msg):
        self.get_logger().info(f'Received action data: {msg.data}')
        pass

    def rby1_callback(self, msg):
        self.get_logger().info(f'Received RBY1 data')
        self.main_state.is_robot_connected = True

        self.main_state.is_robot_initialized = msg.is_initialized
        self.main_state.is_robot_stopped = msg.is_stopped
        
        self.main_state.current_torso_position = np.array(msg.torso_ee_pos.position.data)
        self.main_state.current_torso_quaternion = np.array(msg.torso_ee_pos.quaternion.data)
        self.main_state.current_right_arm_position = np.array(msg.right_ee_pos.position.data)
        self.main_state.current_right_arm_quaternion = np.array(msg.right_ee_pos.quaternion.data)
        self.main_state.current_left_arm_position = np.array(msg.left_ee_pos.position.data)
        self.main_state.current_left_arm_quaternion = np.array(msg.left_ee_pos.quaternion.data)

    def publish_command(self):
        if self.main_state.is_robot_connected:
            command_msg = Command()

            command_msg.is_active = self.main_state.is_meta_ready
            command_msg.control_mode = "component"

            command_msg.desired_head_ee_pos = EEpos()
            command_msg.desired_head_ee_pos.position = Float32MultiArray(data=self.main_state.desired_head_position.tolist())
            command_msg.desired_head_ee_pos.quaternion = Float32MultiArray(data=self.main_state.desired_head_quaternion.tolist())

            command_msg.desired_right_ee_pos = EEpos()
            command_msg.desired_right_ee_pos.position = Float32MultiArray(data=self.main_state.desired_right_arm_position.tolist())
            command_msg.desired_right_ee_pos.quaternion = Float32MultiArray(data=self.main_state.desired_right_arm_quaternion.tolist())

            command_msg.desired_left_ee_pos = EEpos()
            command_msg.desired_left_ee_pos.position = Float32MultiArray(data=self.main_state.desired_left_arm_position.tolist())
            command_msg.desired_left_ee_pos.quaternion = Float32MultiArray(data=self.main_state.desired_left_arm_quaternion.tolist())

            command_msg.estop = False

            command_msg.ready = self.ready
            command_msg.move = self.move
            command_msg.stop = self.stop

            self.stop = False
            self.ready = False

            self.command_pub.publish(command_msg)

    def meta_loop(self):
        if not self.move:
            self.main_state.is_meta_initialized = False
            return

        if not self.main_state.is_robot_connected:
            self.get_logger().warning('Robot is not connected')
            return
        
        if not self.main_state.is_robot_initialized:
            self.get_logger().warning('Robot is not initialized')
            return
        
        if not self.main_state.is_meta_initialized:
            initial_future: rclpy.Future = self.send_meta_initial_offset(check_pose=self.main_state.is_robot_in_ready_pose)
            rclpy.spin_until_future_complete(self, initial_future)

            response = initial_future.result()

            if response is None:
                self.get_logger().error('Failed to call Meta initial offset service')
                self.main_state.is_meta_initialized = False
                return
            
            if not response.success:
                self.get_logger().error(f'Failed to initialize Meta offsets {response.check1, response.check2, response.check3}')
                self.main_state.is_meta_initialized = False
                return
            
            self.get_logger().info('Meta initialized successfully')
            self.main_state.is_meta_initialized = True
            self.main_state.is_robot_in_ready_pose = False

        data_future: rclpy.Future = self.send_meta_get_data()
        rclpy.spin_until_future_complete(self, data_future)

        response = data_future.result()
        if response is None:
            self.get_logger().error('Failed to call Meta data service')
            return

        if len(response.error_msg) > 0:
            self.get_logger().error(f'Meta data error: {response.error_msg}')
            return

        self.get_logger().info('Meta data received successfully')
        self.main_state.is_meta_ready = True
        self.main_state.desired_head_position = np.array(response.head_ee_pos.position.data)
        self.main_state.desired_head_quaternion = np.array(response.head_ee_pos.quaternion.data)
        self.main_state.desired_right_arm_position = np.array(response.right_ee_pos.position.data)
        self.main_state.desired_right_arm_quaternion = np.array(response.right_ee_pos.quaternion.data)
        self.main_state.desired_left_arm_position = np.array(response.left_ee_pos.position.data)
        self.main_state.desired_left_arm_quaternion = np.array(response.left_ee_pos.quaternion.data)

    def reset_state(self):
        self.main_state.is_meta_initialized = False
        self.main_state.is_meta_ready = False

        self.main_state.desired_head_position = np.array([])
        self.main_state.desired_head_quaternion = np.array([])
        self.main_state.desired_right_arm_position = np.array([])
        self.main_state.desired_right_arm_quaternion = np.array([])
        self.main_state.desired_left_arm_position = np.array([])
        self.main_state.desired_left_arm_quaternion = np.array([])

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
        elif key == ord('m'):
            self.get_logger().info('Received move command')
            self.move = not self.move
        elif key == ord('s'):
            self.get_logger().info('Received stop command')
            self.stop = True
        elif key == ord('q'):
            self.get_logger().info('Received quit command')
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()
        elif key == ord('c'):
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