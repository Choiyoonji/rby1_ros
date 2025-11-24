import numpy as np
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32, Bool, Int32MultiArray, Float32MultiArray
from rby1_interfaces.msg import EEpos, FTsensor, State, Command
from rby1_interfaces.srv import MetaInitialReq, MetaDataReq

from rby1_ros.qos_profiles import qos_state_latest, qos_cmd, qos_ctrl_latched

import cv2
import collections
import pickle
import struct
from rby1_ros.main_status import MainStatus as MainState
from rby1_ros.rby1_dyn import RBY1Dyn
from rby1_ros.utils import *
from rby1_ros.tcp_utils import *

from .cam_utils.shm_util import NamedSharedNDArray
from .cam_utils.realsense_mp_full_ts import Realsense_MP
from .cam_utils.zed_mp import ZED_MP
from .cam_utils.digit_mp import DIGIT_MP

EXO_CAM_SHM_NAME = "right_wrist_D405"
WRI1_CAM_SHM_NAME = "external_D435I"
# CAM3_SHM_NAME = "left_wrist_D405"  # 필요시 사용

MODE = "joint_position"  # joint_position / cartesian_position
ARM = "right"  # right / left / both


class MainNode(Node):
    def __init__(self):
        super().__init__('main_node_inference')

        # ----- 파라미터 (IP/PORT/TASK) -----
        self.declare_parameter("server_ip", "127.0.0.1")
        self.declare_parameter("server_port", 12345)
        self.declare_parameter("task", "Pick up the eclipse case")  # 필요하면 바꾸기

        self.server_ip = self.get_parameter("server_ip").get_parameter_value().string_value
        self.server_port = self.get_parameter("server_port").get_parameter_value().integer_value
        self.prompt = self.get_parameter("task").get_parameter_value().string_value
        
        self.ready: bool = False
        self.move: bool = False
        self.stop: bool = False

        self.main_state = MainState()
        
        # /rby1/state : 로봇 상태 → 최신값만
        self.rby1_sub = self.create_subscription(
            State,
            '/rby1/state',
            self.state_callback,
            qos_state_latest
        )

        # /control/command : 컨트롤 명령 주기 발행
        self.command_pub = self.create_publisher(
            Command,
            '/control/command',
            qos_cmd
        )

        # ----- 내부 상태 -----
        self.sock = None
        self.connected = False

        # action plan (여러 step을 한번에 받아서 쓰고 싶으면 사용)
        self.action_plan = collections.deque()
        self.action_len = 5  # plan 길이 (step 수)

        self.get_logger().info(
            f"[init] RBY1InferenceNode started. "
            f"Server: {self.server_ip}:{self.server_port}, task='{self.prompt}'"
        )
        
        self.command_data = {
            "mode": "joint_position", # joint_position / cartesian_position
            "arm": "right", # right / left / both
            "right_arm_angle": np.zeros(7),
            "left_arm_angle": np.zeros(7),
            "right_arm_pos": np.zeros(7), # (xyz + quaternion)
            "left_arm_pos": np.zeros(7), # (xyz + quaternion)
            "right_gripper": 1.0,
            "left_gripper": 1.0,
        }
        
        # ----- camera process open -----
        self.cam1 = self.create_cam_process("D405", EXO_CAM_SHM_NAME)
        self.cam2 = self.create_cam_process("D435I", WRI1_CAM_SHM_NAME)

        self.cam1.start()
        self.cam2.start()

        time.sleep(3.0) # Wait for camera to initialize
        self.get_logger().info("Camera processes started.")
        
        self.shm1, self.shm2 = self.open_all_shm()
        
        self.connect()
        
        if self.connected:
            self.get_logger().info("Connected to policy server.")
        else:
            self.get_logger().warn("Failed to connect to policy server.")
        
        self.step_hz = 30.0
        self.step_timer = self.create_timer(1/self.step_hz, self.inference_step)

    # ----- TCP 연결 -----
    def connect(self):
        if self.connected:
            return
        self.sock, self.connected = connect_tcp(self.server_ip, self.server_port)
        if self.connected:
            self.get_logger().info("Connected to policy server.")
        else:
            self.get_logger().warn("Failed to connect to policy server.")
    
    def disconnect(self):
        if self.connected:
            disconnect_tcp(self.sock)
            self.sock = None
            self.connected = False
            self.get_logger().info("Disconnected from policy server.")
    
    # ----- /rby1/state 콜백 -----
    def state_callback(self, msg: State):
        self.latest_state_msg = msg
        self.main_state.is_robot_connected = True

        self.main_state.is_robot_initialized = msg.is_initialized
        self.main_state.is_robot_stopped = msg.is_stopped
        
        self.main_state.current_joint_positions = np.array(msg.joint_positions.data)
        
        self.main_state.current_torso_position = np.array(msg.torso_ee_pos.position.data)
        self.main_state.current_torso_quaternion = np.array(msg.torso_ee_pos.quaternion.data)
        self.main_state.current_right_arm_position = np.array(msg.right_ee_pos.position.data)
        self.main_state.current_right_arm_quaternion = np.array(msg.right_ee_pos.quaternion.data)
        self.main_state.current_left_arm_position = np.array(msg.left_ee_pos.position.data)
        self.main_state.current_left_arm_quaternion = np.array(msg.left_ee_pos.quaternion.data)

        self.main_state.current_right_gripper_position = msg.right_gripper_pos
        self.main_state.current_left_gripper_position = msg.left_gripper_pos
        
    def create_cam_process(self, cam_type, shm_name, serial_number=None):
        if cam_type == "D405" or cam_type == "D435I":
            return Realsense_MP(
                shm_name=shm_name,
                device_name=cam_type,
                width=640,
                height=480,
                fps=30,
                serial_number=serial_number,
            )
        elif cam_type == "ZED":
            return ZED_MP(
                shm_left=shm_name+"_left",
                shm_right=shm_name+"_right",
                shm_meta=shm_name+"_meta",
                width=1280, height=720, fps=30,
                serial_number=None,
                camera_id=0,
                verbose=True,
            )
        elif cam_type == "DIGIT":
            return DIGIT_MP(shm_name, name=shm_name)
        else:
            self.get_logger().error(f"Unknown camera type: {cam_type}")
            return None
        
    def open_all_shm(self, timeout_sec: float = 5.0):
        t0 = time.time()
        shm1 = shm2 = None
        while time.time() - t0 < timeout_sec:
            try:
                shm1 = shm1 or NamedSharedNDArray.open(EXO_CAM_SHM_NAME)
                shm2 = shm2 or NamedSharedNDArray.open(WRI1_CAM_SHM_NAME)
                if shm1 is not None and shm2 is not None:
                    break
            except FileNotFoundError:
                pass
            time.sleep(0.05)
        return shm1, shm2
    
    def ensure_bgr(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def preprocess_image(self, image):
        if image is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (256, 256))
    
    def get_action_from_server(self, state_dict):
        if not self.connected:
            return None
        try:
            action = send_packet(self.sock, state_dict)  # 상태 전송 및 응답 받기
            return action[:self.action_len]
        except Exception as e:
            self.get_logger().error(f"Error during TCP communication: {e}")
            self.disconnect()
            return None
    
    def publish_command(self, command_data):
        cmd_msg = Command()
        if command_data["mode"] == "joint_position":
            cmd_msg.control_mode = "joint_position"
            
            if command_data["arm"] == "right":
                cmd_msg.desired_joint_positions = Float32MultiArray(
                    data=command_data["right_arm_angle"].tolist()
                )
                cmd_msg.right_btn = True
            elif command_data["arm"] == "left":
                cmd_msg.desired_joint_positions = Float32MultiArray(
                    data=command_data["left_arm_angle"].tolist()
                )
                cmd_msg.left_btn = True
            elif command_data["arm"] == "both":
                cmd_msg.desired_joint_positions = Float32MultiArray(
                    data=np.concatenate([
                        command_data["right_arm_angle"],
                        command_data["left_arm_angle"]
                    ]).tolist()
                )
                cmd_msg.right_btn = True
                cmd_msg.left_btn = True
                
        elif command_data["mode"] == "cartesian_position":
            cmd_msg.control_mode = "component"
            
            if command_data["arm"] == "right" or command_data["arm"] == "both":
                cmd_msg.desired_right_ee_pos = EEpos()
                cmd_msg.desired_right_ee_pos.position = Float32MultiArray(
                    data=command_data["right_arm_pos"][:3].tolist()
                )
                cmd_msg.desired_right_ee_pos.quaternion = Float32MultiArray(
                data=command_data["right_arm_pos"][3:].tolist()
                )
            if command_data["arm"] == "left" or command_data["arm"] == "both":
                cmd_msg.desired_left_ee_pos = EEpos()
                cmd_msg.desired_left_ee_pos.position = Float32MultiArray(
                    data=command_data["left_arm_pos"][:3].tolist()
                )
                cmd_msg.desired_left_ee_pos.quaternion = Float32MultiArray(
                data=command_data["left_arm_pos"][3:].tolist()
            )
        cmd_msg.desired_right_gripper_pos = command_data["right_gripper"]
        cmd_msg.desired_left_gripper_pos = command_data["left_gripper"]
        cmd_msg.is_active = True
        cmd_msg.ready = self.ready
        cmd_msg.move = self.move
        cmd_msg.stop = self.stop
            
        self.stop = False
        self.ready = False
        
        self.command_pub.publish(cmd_msg)

    def reset_state(self):
        self.main_state.desired_head_position = np.array([])
        self.main_state.desired_head_quaternion = np.array([])
        self.main_state.desired_torso_position = np.array([])
        self.main_state.desired_torso_quaternion = np.array([])
        self.main_state.desired_right_arm_position = np.array([])
        self.main_state.desired_right_arm_quaternion = np.array([])
        self.main_state.desired_right_gripper_position = 0.0
        self.main_state.desired_left_arm_position = np.array([])
        self.main_state.desired_left_arm_quaternion = np.array([])
        self.main_state.desired_left_gripper_position = 0.0
        
    def inference_step(self):
        if self.main_state.is_robot_stopped:
            self.reset_state()
            
        try:
            color1 = self.ensure_bgr(self.shm1.as_array().copy())
            color2 = self.ensure_bgr(self.shm2.as_array().copy())
        except Exception as e:
            print("⚠️ Failed to read camera frames from SHM:", e)
            time.sleep(0.05)
            return

        if color1 is None or color2 is None:
            print("⚠️ Invalid camera frames. Skipping.")
            time.sleep(0.05)
            return
        
        concat_img = np.hstack([color1.copy(), color2.copy()])
        cv2.imshow("CAM", concat_img)
        
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
        
        # state가 아직 없음
        if self.main_state.is_robot_connected is False:
            return
        
        if self.ready or self.stop:
            self.publish_command({
                "mode": "joint_position",
                "arm": "right",
                "right_arm_angle": self.main_state.current_joint_positions[8:15],
                "left_arm_angle": self.main_state.current_joint_positions[15:22],
                "right_gripper": self.main_state.current_right_gripper_position,
                "left_gripper": self.main_state.current_left_gripper_position,
            })
            return
        
        if self.move is False:
            return
        
        if np.all(np.abs(self.main_state.current_joint_positions) < 1e-3):
            print("⚠️ Robot joints nearly zero; likely initializing. Retrying shortly.")
            time.sleep(0.05)
            return
        
        try:
            robot_state = np.zeros(8)
            if MODE == "joint_position" and ARM == "right":
                robot_state = self.main_state.current_joint_positions.tolist()[8:15] + [float(self.main_state.current_right_gripper_position)]
                
            img_exo = self.preprocess_image(color1)
            img_wri1 = self.preprocess_image(color2)
            
            if not self.action_plan:
                input_payload = {
                    "image": img_exo,
                    "wrist_image": img_wri1,
                    "state": robot_state,
                    "prompt": self.prompt,
                }
                
                action_seq = self.get_action_from_server(input_payload)
                if action_seq is None:
                    print("⚠️ No action received from server.")
                    return
                
                try:
                    self.action_plan.extend(action_seq)
                except Exception as e:
                    self.action_plan.clear()
                    print("⚠️ Failed to extend action plan:", e)
                    return
                
            action = self.action_plan.popleft()
            print("Action:", action)
            
            if not isinstance(action, (list, tuple, np.ndarray)) or len(action) != 8:
                print("⚠️ Invalid action format received. Skipping this step.")
                return
            
            if MODE == "joint_position" and ARM == "right":
                command_data = {
                    "mode": "joint_position",
                    "arm": "right",
                    "right_arm_angle": np.array(action[:7]),
                    "left_arm_angle": self.main_state.current_joint_positions[15:22],
                    "right_gripper": float(1.0),
                    "left_gripper": self.main_state.current_left_gripper_position,
                }
                self.publish_command(command_data)
                
        except KeyboardInterrupt:
            print("🛑 Interrupted (Ctrl+C)")
        except Exception as e:
            self.get_logger().error(f"Error in inference step: {e}")
            self.disconnect()
            self.cam1.stop()
            self.cam2.stop()
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()
            return
        

def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()
        