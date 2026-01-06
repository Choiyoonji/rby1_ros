import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32, Float32, Bool, Int32MultiArray, Float32MultiArray
from rby1_interfaces.msg import EEpos, FTsensor, StateRBY1, Action
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
            StateRBY1,
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

        self.wrist_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.external_image = np.zeros((480, 640, 3), dtype=np.uint8)

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
            "right_gripper",
            "left_hand",
            "right_hand"
        ]

        self.loop_timer = self.create_timer(0.02, self.loop_callback)
        self.is_action_done: bool = True
        self.action_queue: list = []  # Add this line 
        
        self.target_arm = ["left", "right"]
        self.target_idx = 1
        
        self.target_positions = {
            "red_cylinder": [ 0.53371574, -0.24851188,0.82],
            "white_box": [0.51530604, -0.18308753, 0.83],
            "black_box": [0.5289317, -0.10273802, 0.86183036],
            "up_black_box": [0.53691385, -0.59051222, 1.02815218]
        }
    # ----- /rby1/state 콜백 -----
    def state_callback(self, msg: StateRBY1):
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
        
    def move_ee_absolute_pos(self, arm, target_pos: list[float]):
        
        if arm == "right":
            current_pos = self.main_state.current_right_arm_position
        elif arm == "left":
            current_pos = self.main_state.current_left_arm_position
        else:
            self.get_logger().error('Invalid arm: {}. Use "left" or "right"'.format(arm))
            return
        
        delta_pos = np.array(target_pos) - current_pos
        
        self.get_logger().info(f'Moving {arm} arm from {current_pos} to {target_pos} (delta: {delta_pos})')
        
        # 1단계: x, y만 이동 (z는 현재 위치 유지)
        delta_xy = [delta_pos[0], delta_pos[1], 0.0]
        self.get_logger().info(f'Step 1: Moving x,y by {delta_xy}')
        
        mode = f"{arm}_pos"
        
        # 첫 번째 액션 즉시 실행
        self.publish_action(mode=mode, param_name="dpos", param=delta_xy)
        
        # 2단계: z축만 이동 (큐에 추가)
        delta_z = [0.0, 0.0, delta_pos[2]]
        self.get_logger().info(f'Step 2: Will move z by {delta_z} after step 1 completes')
        self.action_queue.append({
            "mode": mode,
            "param_name": "dpos",
            "param": delta_z
        })
        
    def wrist_image_callback(self, msg):
        # 이미지 메시지를 OpenCV 이미지로 변환
        self.wrist_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # 이미지 처리 로직 추가 가능
        self.get_logger().info('Received wrist image of size: {}x{}'.format(self.wrist_image.shape[1], self.wrist_image.shape[0]))
        self.is_action_done = True

    def external_image_callback(self, msg):
        # 이미지 메시지를 OpenCV 이미지로 변환
        self.external_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # 이미지 처리 로직 추가 가능
        self.get_logger().info('Received external image of size: {}x{}'.format(self.external_image.shape[1], self.external_image.shape[0]))
        self.is_action_done = True

    def done_callback(self, msg):
        if msg.data:
            self.get_logger().info('Received done signal from action executor.')
            self.is_action_done = True
            
            # Process next action in queue
            if self.action_queue:
                next_action = self.action_queue.pop(0)
                self.get_logger().info(f'Executing queued action: {next_action["mode"]}')
                self.publish_action(
                    mode=next_action["mode"],
                    param_name=next_action["param_name"],
                    param=next_action["param"]
                )

    def publish_action(self, mode, param_name=None, param=None):
        if mode not in self.mode_list:
            self.get_logger().error('Invalid mode: {}'.format(mode))
            return
        action_msg = Action()
        action_msg.mode = mode
        if param_name is not None and param is not None:
            setattr(action_msg, param_name, Float32MultiArray(data=param) if isinstance(param, list) else param)
            
        self.action_pub.publish(action_msg)
        self.is_action_done = False
    
    def publish_hand_action(self, mode, param_name=None, param=0.0):
        if mode not in self.mode_list:
            self.get_logger().error('Invalid mode: {}'.format(mode))
            return
        action_msg = Action()
        action_msg.mode = mode
        
        if mode.startswith('left'):
            setattr(action_msg.left_hand, param_name, Float32MultiArray(data=param) if isinstance(param, list) else param)
        elif mode.startswith('right'):
            setattr(action_msg.right_hand, param_name, Float32MultiArray(data=param) if isinstance(param, list) else param)
            
        self.action_pub.publish(action_msg)
        self.is_action_done = False

    def request_image(self):
        self.publish_action(mode="image")

    def move_ee_delta_pos(self, arm, delta_pos:list[float]):
        mode = f"{arm}_pos" if arm in ["left", "right"] else "both_pos"
        self.publish_action(mode=mode, param_name="dpos", param=delta_pos)

    def move_ee_delta_rot(self, arm, delta_rot, axis='z', type="global"):
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
        self.publish_action(mode=mode, param_name="drot", param=list(rot_vec))

    def right_gripper_open(self):
        self.publish_action(mode="right_gripper", param_name="right_gripper_pos", param=1.0)

    def left_gripper_open(self):
        self.publish_action(mode="left_gripper", param_name="left_gripper_pos", param=1.0)

    def right_gripper_close(self):
        self.publish_action(mode="right_gripper", param_name="right_gripper_pos", param=0.0)

    def left_gripper_close(self):
        self.publish_action(mode="left_gripper", param_name="left_gripper_pos", param=0.0)

    def right_gripper_set_position(self, position):
        self.publish_action(mode="right_gripper", param_name="right_gripper_pos", param=position)

    def left_gripper_set_position(self, position):
        self.publish_action(mode="left_gripper", param_name="left_gripper_pos", param=position)

    def right_hand_set_position(self, position):
        self.publish_hand_action(mode="right_hand", param_name="opening", param=position)

    def left_hand_set_position(self, position):
        self.publish_hand_action(mode="left_hand", param_name="opening", param=position)

    def draw_ui(self, image):
        h, w = image.shape[:2]
        
        # 상태 표시 (Ready / Busy)
        if self.is_action_done:
            state_text = "[READY]"
            color = (0, 255, 0)
        else:
            state_text = "[BUSY...]"
            color = (0, 0, 255)
            
        cv2.putText(image, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 마지막 상태 메시지
        state_text = "Status: " + ("executing" if not self.is_action_done else "idle")
        arm_text = f"Target Arm: {self.target_arm[self.target_idx]}"
        cv2.putText(image, arm_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, state_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 가이드
        guide = "'o':Open, 'c':Close, 'm':Move, 'q':Quit"
        cv2.putText(image, guide, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def loop_callback(self):
        display_image = cv2.hconcat([self.wrist_image.copy(), self.external_image.copy()])
        self.draw_ui(display_image)
        cv2.imshow("RBY1 Camera Views (Wrist | External)", display_image)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.get_logger().info("Quit signal received.")
            # ROS 종료 예외 발생시켜 Spin을 멈춤
            raise SystemExit
        
        elif key == ord('i'):
            self.request_image()
            self.get_logger().info("Requesting Images from Cameras")
        
        elif key == ord('o') and self.is_action_done:
            self.right_gripper_open()
            self.get_logger().info("Opening Gripper")
        
        elif key == ord('c') and self.is_action_done:
            self.right_gripper_close()
            self.get_logger().info("Closing Gripper")

        elif key == ord('w') and self.is_action_done:
            self.move_ee_delta_pos(self.target_arm[self.target_idx], [0.10, 0.0, 0.0])

        elif key == ord('s') and self.is_action_done:
            self.move_ee_delta_pos(self.target_arm[self.target_idx], [-0.10, 0.0, 0.0])

        elif key == ord('a') and self.is_action_done:
            self.move_ee_delta_pos(self.target_arm[self.target_idx], [0.0, 0.10, 0.0])
            
        elif key == ord('d') and self.is_action_done:
            self.move_ee_delta_pos(self.target_arm[self.target_idx], [0.0, -0.10, 0.0])

        elif key == ord('u') and self.is_action_done:
            self.move_ee_delta_pos(self.target_arm[self.target_idx], [0.0, 0.0, 0.10])

        elif key == ord('b') and self.is_action_done:
            self.move_ee_delta_pos(self.target_arm[self.target_idx], [0.0, 0.0, -0.10])

        elif key == ord('l') and self.is_action_done:
            self.move_ee_delta_rot(self.target_arm[self.target_idx], 30.0, axis='x', type="global")

        elif key == ord('j') and self.is_action_done:
            self.move_ee_delta_rot(self.target_arm[self.target_idx], -30.0, axis='x', type="global")
            
        elif key in [ord('1'), ord('2'), ord('3'),ord('4')] and self.is_action_done:
            # 키에 따른 타겟 매핑
            target_map = {
                ord('1'): "red_cylinder",
                ord('2'): "white_box",
                ord('3'): "black_box",
                ord('4'): "up_black_box"
            }
            
            target_name = target_map[key]
            target_pos = self.target_positions[target_name]
            self.move_ee_absolute_pos(self.target_arm[self.target_idx], target_pos)
            self.get_logger().info(f"Moving to {target_name} at position: {target_pos}")
        elif key == ord('n') and self.is_action_done:
            self.left_hand_set_position(0.0)
        elif key == ord('m') and self.is_action_done:
            self.left_hand_set_position(1.0)
        elif key == ord('z') and self.is_action_done:
            self.right_hand_set_position(0.0)
        elif key == ord('x') and self.is_action_done:
            self.right_hand_set_position(1.0)
        elif key == ord('7'):
            self.target_idx = (self.target_idx + 1) % len(self.target_arm)
            self.get_logger().info(f"Target arm set to: {self.target_arm[self.target_idx]}")
            
def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()

    try:
        rclpy.spin(node)
    except SystemExit:
        print("Closing application...")
    except KeyboardInterrupt:
        print("Keyboard Interrupt.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()