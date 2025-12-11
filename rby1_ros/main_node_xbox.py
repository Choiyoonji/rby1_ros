import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32, Float32, Bool, Int32MultiArray, Float32MultiArray
from rby1_interfaces.msg import EEpos, FTsensor, StateRBY1, CommandRBY1, CommandHand, Action
from rby1_interfaces.srv import MetaInitialReq, MetaDataReq

from rby1_ros.qos_profiles import qos_state_latest, qos_cmd, qos_ctrl_latched, qos_image_stream

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge
import collections
from pathlib import Path
from rby1_ros.main_status import MainStatus as MainState
from rby1_ros.utils import *

from .cam_utils.shm_util import NamedSharedNDArray
from .cam_utils.realsense_mp_full_ts import Realsense_MP
from .cam_utils.zed_mp import ZED_MP
from .cam_utils.digit_mp import DIGIT_MP
from .command_filter import LinearCommandFilter, AngularCommandFilter, PoseCommandFilter, CommandFilter

EXO_CAM_SHM_NAME = "right_wrist_D405"
WRI1_CAM_SHM_NAME = "external_D435I"
# CAM3_SHM_NAME = "left_wrist_D405"  # 필요시 사용

class MainNode(Node):
    def __init__(self):
        super().__init__('main_node_command')
        
        self.ready: bool = False
        self.move: bool = False
        self.stop: bool = False

        self.main_state = MainState()
        
        # /rby1/state : 로봇 상태 → 최신값만
        self.rby1_sub = self.create_subscription(
            StateRBY1,
            '/rby1/state',
            self.state_callback,
            qos_state_latest
        )

        # /control/command : 컨트롤 명령 주기 발행
        self.command_pub_rby1 = self.create_publisher(
            CommandRBY1,
            '/control/command/rby1',
            qos_cmd
        )
        self.command_pub_hand = self.create_publisher(
            CommandHand,
            '/control/command/hand',
            qos_cmd
        )
        
        self.action_sub = self.create_subscription(
            Action,
            '/control/action',
            self.action_callback,
            qos_cmd
        )

        self.done_pub = self.create_publisher(
            Bool,
            '/control/done',
            qos_cmd
        )

        self.wrist_img_pub = self.create_publisher(
            Image,
            '/camera/right_wrist/image_raw',
            qos_image_stream
        )
        
        self.external_img_pub = self.create_publisher(
            Image,
            '/camera/external/image_raw',
            qos_image_stream
        )
        
        self.bridge = CvBridge()

        # ----- 내부 상태 -----
        self.sock = None
        self.connected = False

        self.action_plan = collections.deque()
        self.last_planned_state = {}
        self.is_published = True

        self.get_logger().info(
            f"[init] RBY1CommandNode started."
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
        
        self.step_hz = 30
        self.step_timer = self.create_timer(1/self.step_hz, self.step)
        
        # Command Filters for teleoperation (replacing trajectory planners)
        dt = 1.0 / self.step_hz
        
        # Position and Orientation Filters (omega=10.0 for responsive teleoperation)
        self.right_pos_filter = LinearCommandFilter(dt=dt, omega=5.0, zeta=1.5, 
                                                    x_bounds=(0.2434, 0.4292), 
                                                    y_bounds=(-0.3385, -0.1776), 
                                                    z_bounds=(0.8286, 1.1150))
        self.right_rot_filter = AngularCommandFilter(dt=dt, omega=5.0, zeta=1.5)
        self.left_pos_filter = LinearCommandFilter(dt=dt, omega=10.0, zeta=1.0)
        self.left_rot_filter = AngularCommandFilter(dt=dt, omega=10.0, zeta=1.0)
        
        # Joint Filters for joint-based control
        self.right_joint_filter = CommandFilter(num_joints=7, dt=dt, omega=10.0, zeta=1.0)
        self.left_joint_filter = CommandFilter(num_joints=7, dt=dt, omega=10.0, zeta=1.0)
        
        self.action_history = []
        self.state_history = []
        
        # Simplified action_map for teleoperation (filters handle smoothing internally)
        self.action_map = {
            "image": [self.publish_images],
            "left_joint": [],
            "right_joint": [],
            "both_joint": [],
            "left_pos": [],
            "right_pos": [],
            "both_pos": [],
            "left_rot_local": [],
            "right_rot_local": [],
            "both_rot_local": [],
            "left_rot_global": [],
            "right_rot_global": [],
            "both_rot_global": [],
            "left_gripper": [],
            "right_gripper": [],
            "left_hand": [],
            "right_hand": []
        }

        self.action_list = list(self.action_map.keys())
    
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
    
    def action_callback(self, msg: Action):
        """
        For teleoperation with Xbox controller: Apply delta inputs to command filters.
        Each call applies one delta step to the filter, which smooths the output.
        """
        self.get_logger().info(f'Received action data: {msg.mode}')
        if msg.mode not in self.action_map.keys():
            self.get_logger().error(f'Unknown action mode: {msg.mode}')
            return
        
        # Cancel last action clears the action plan
        if msg.cancel_last_action:
            self.action_plan.clear()
            self.get_logger().info("Cleared action plan")
            return
        
        # Image publishing (no filter needed)
        if msg.mode == "image":
            self.publish_images()
            return
        
        self.is_published = False
        
        # Gripper control (immediate command)
        if msg.mode in ["left_gripper", "right_gripper"]:
            arm = msg.mode.split("_")[0]
            opening = msg.left_gripper_pos if arm == "left" else msg.right_gripper_pos
            if isinstance(opening, list):
                opening = opening[0] if len(opening) > 0 else 1.0
            self.set_gripper_position(arm, opening)
            return

        if msg.mode in ["left_hand", "right_hand"]:
            hand = msg.mode.split("_")[0]
            position = msg.left_hand_pos.data if hand == "left" else msg.right_hand_pos.data
            if isinstance(position, list):
                position = position[0] if len(position) > 0 else 0.0
            self.set_hand_position(hand, position, duration=0.5)
            return

        # Parse action mode to determine control type and arm(s)
        parts = msg.mode.split('_')
        arm_spec = parts[0]  # 'left', 'right', or 'both'
        control_type = '_'.join(parts[1:])  # 'pos', 'joint', 'rot_local', 'rot_global'
        
        # Extract delta inputs from message
        if 'pos' in control_type:
            delta = np.array(msg.dpos.data, dtype=float)
        elif 'joint' in control_type:
            delta = np.array(msg.dtheta.data, dtype=float)
        elif 'rot' in control_type:
            delta = np.array(msg.drot.data, dtype=float)
        else:
            self.get_logger().error(f"Unknown control type in mode: {msg.mode}")
            return
        
        # Determine if rotation is in local or global frame
        rot_frame = 'global' if 'global' in control_type else 'local'
        
        # Apply filter updates based on control type and arm
        try:
            if 'pos' in control_type:
                self._handle_position_delta(arm_spec, delta)
            elif 'joint' in control_type:
                self._handle_joint_delta(arm_spec, delta)
            elif 'rot' in control_type:
                self._handle_rotation_delta(arm_spec, delta, rot_frame)
        except Exception as e:
            self.get_logger().error(f"Error in action callback: {e}")

    def _handle_position_delta(self, arm_spec: str, dpos: np.ndarray):
        """Apply position delta through filters"""
        if arm_spec == 'right':
            target_pos = self.main_state.current_right_arm_position + dpos
            filtered_pos, lin_vel, lin_acc = self.right_pos_filter.update(target_pos)
            print(f"Right Pos Target: {target_pos}, Filtered: {filtered_pos}")
            self.main_state.desired_right_arm_position = filtered_pos
        elif arm_spec == 'left':
            target_pos = self.main_state.current_left_arm_position + dpos
            filtered_pos, lin_vel, lin_acc = self.left_pos_filter.update(target_pos)
            print(f"Left Pos Target: {target_pos}, Filtered: {filtered_pos}")
            self.main_state.desired_left_arm_position = filtered_pos
        elif arm_spec == 'both':
            # Split delta for both arms
            dpos_right = dpos[:3]
            dpos_left = dpos[3:6] if len(dpos) >= 6 else dpos_right
            
            target_pos_right = self.main_state.current_right_arm_position + dpos_right
            target_pos_left = self.main_state.current_left_arm_position + dpos_left
            
            filtered_pos_right, _, _ = self.right_pos_filter.update(target_pos_right)
            filtered_pos_left, _, _ = self.left_pos_filter.update(target_pos_left)
            
            print(f"Both Pos Target Right: {target_pos_right}, Filtered: {filtered_pos_right}")
            print(f"Both Pos Target Left: {target_pos_left}, Filtered: {filtered_pos_left}")
            
            self.main_state.desired_right_arm_position = filtered_pos_right
            self.main_state.desired_left_arm_position = filtered_pos_left

    def _handle_joint_delta(self, arm_spec: str, dtheta: np.ndarray):
        """Apply joint delta through filters"""
        if arm_spec == 'right':
            target_angle = self.main_state.current_right_arm_angle + dtheta
            filtered_angle, _, _ = self.right_joint_filter.update(target_angle)
            self.main_state.desired_right_arm_angle = filtered_angle
        elif arm_spec == 'left':
            target_angle = self.main_state.current_left_arm_angle + dtheta
            filtered_angle, _, _ = self.left_joint_filter.update(target_angle)
            self.main_state.desired_left_arm_angle = filtered_angle
        elif arm_spec == 'both':
            # Split delta for both arms
            dtheta_right = dtheta[:7]
            dtheta_left = dtheta[7:14] if len(dtheta) >= 14 else dtheta_right
            
            target_angle_right = self.main_state.current_right_arm_angle + dtheta_right
            target_angle_left = self.main_state.current_left_arm_angle + dtheta_left
            
            filtered_angle_right, _, _ = self.right_joint_filter.update(target_angle_right)
            filtered_angle_left, _, _ = self.left_joint_filter.update(target_angle_left)
            
            self.main_state.desired_right_arm_angle = filtered_angle_right
            self.main_state.desired_left_arm_angle = filtered_angle_left

    def _handle_rotation_delta(self, arm_spec: str, drot: np.ndarray, frame: str = 'local'):
        """Apply rotation delta through filters"""
        # Convert rotation vector to delta quaternion for incremental rotation
        delta_quat = rotvec_to_quat(drot)
        
        if arm_spec == 'right':
            # Apply incremental rotation to current quaternion
            target_quat = mul_quat(self.main_state.current_right_arm_quaternion, delta_quat)
            filtered_quat, ang_vel, ang_acc = self.right_rot_filter.update(target_quat, frame=frame)
            print(f"Right Rot Target: {target_quat}, Filtered: {filtered_quat}")
            self.main_state.desired_right_arm_quaternion = filtered_quat
        elif arm_spec == 'left':
            target_quat = mul_quat(self.main_state.current_left_arm_quaternion, delta_quat)
            filtered_quat, ang_vel, ang_acc = self.left_rot_filter.update(target_quat, frame=frame)
            print(f"Left Rot Target: {target_quat}, Filtered: {filtered_quat}")
            self.main_state.desired_left_arm_quaternion = filtered_quat
        elif arm_spec == 'both':
            # Split delta for both arms
            drot_right = drot[:3]
            drot_left = drot[3:6] if len(drot) >= 6 else drot_right
            
            delta_quat_right = rotvec_to_quat(drot_right)
            delta_quat_left = rotvec_to_quat(drot_left)
            
            target_quat_right = mul_quat(self.main_state.current_right_arm_quaternion, delta_quat_right)
            target_quat_left = mul_quat(self.main_state.current_left_arm_quaternion, delta_quat_left)
            
            filtered_quat_right, _, _ = self.right_rot_filter.update(target_quat_right, frame=frame)
            filtered_quat_left, _, _ = self.left_rot_filter.update(target_quat_left, frame=frame)
            
            print(f"Both Rot Target Right: {target_quat_right}, Filtered: {filtered_quat_right}")
            print(f"Both Rot Target Left: {target_quat_left}, Filtered: {filtered_quat_left}")
            
            self.main_state.desired_right_arm_quaternion = filtered_quat_right
            self.main_state.desired_left_arm_quaternion = filtered_quat_left


    def set_gripper_position(self, arm: str, opening: float, duration=0.5):
        if arm == "right":
            self.main_state.desired_right_gripper_position = opening
            print(f"Set right gripper to {opening}")
        elif arm == "left":
            self.main_state.desired_left_gripper_position = opening
            print(f"Set left gripper to {opening}")
        else:
            self.get_logger().error(f"Unknown arm for gripper: {arm}")

    def set_hand_position(self, hand: str, position: float, duration=0.5):
        """Set hand position directly and publish command"""
        self.publish_command_hand(hand, position)

    
    def publish_images(self):
        try:
            color1 = self.ensure_bgr(self.shm1.as_array().copy())
            color2 = self.ensure_bgr(self.shm2.as_array().copy())
            # 현재 시간 고정 (스냅샷 시간)
            now = self.get_clock().now().to_msg()
            
            # 첫 번째 이미지 메시지 생성
            msg1 = self.bridge.cv2_to_imgmsg(self.preprocess_image(color1), encoding="bgr8")
            msg1.header.stamp = now
            msg1.header.frame_id = "camera_main"
            
            # 두 번째 이미지 메시지 생성
            msg2 = self.bridge.cv2_to_imgmsg(self.preprocess_image(color2), encoding="bgr8")
            msg2.header.stamp = now
            msg2.header.frame_id = "camera_wrist"
            
            # 전송
            self.external_img_pub.publish(msg1)
            self.wrist_img_pub.publish(msg2)

        except Exception as e:
            self.get_logger().error(f"Error publishing: {e}")
        
    def publish_command_rby1(self, command_data):
        # print(f"Publishing RBY1 command: mode={command_data['mode']}, arm={command_data.get('arm', 'N/A')}")
        cmd_msg = CommandRBY1()
        
        if command_data["mode"] == "signal":
            cmd_msg.is_active = True
            cmd_msg.ready = self.ready
            cmd_msg.move = self.move
            cmd_msg.stop = self.stop
            self.stop = False
            self.ready = False
            self.command_pub_rby1.publish(cmd_msg)
            return
            
        elif command_data["mode"] == "joint":
            cmd_msg.control_mode = "joint_position"
            action = np.array(command_data["action"], dtype=float)
            
            if command_data["arm"] == "right":
                self.main_state.desired_right_arm_angle = action
                cmd_msg.right_btn = True
                cmd_msg.left_btn = False
            elif command_data["arm"] == "left":
                self.main_state.desired_left_arm_angle = action
                cmd_msg.right_btn = False
                cmd_msg.left_btn = True
            elif command_data["arm"] == "both":
                self.main_state.desired_right_arm_angle = action[:7]
                self.main_state.desired_left_arm_angle = action[7:]
                cmd_msg.right_btn = True
                cmd_msg.left_btn = True
                
            self.main_state.desired_joint_positions = np.concatenate([
                self.main_state.desired_right_arm_angle,
                self.main_state.desired_left_arm_angle,
            ])
            cmd_msg.desired_joint_positions = Float32MultiArray(
                data=self.main_state.desired_joint_positions.tolist()
            )
                
        elif command_data["mode"] == "pos":
            cmd_msg.control_mode = "ee_position"
            action = np.array(command_data["action"], dtype=float)
            
            if command_data["arm"] == "right":
                self.main_state.desired_right_arm_position = action[:3]
                self.main_state.desired_right_arm_quaternion = action[3:7]
                cmd_msg.right_btn = True
                cmd_msg.left_btn = False
            elif command_data["arm"] == "left":
                self.main_state.desired_left_arm_position = action[:3]
                self.main_state.desired_left_arm_quaternion = action[3:7]
                cmd_msg.right_btn = False
                cmd_msg.left_btn = True
            elif command_data["arm"] == "both":
                # Format: [right_pos(3), right_quat(4), left_pos(3), left_quat(4)]
                self.main_state.desired_right_arm_position = action[:3]
                self.main_state.desired_right_arm_quaternion = action[3:7]
                self.main_state.desired_left_arm_position = action[7:10]
                self.main_state.desired_left_arm_quaternion = action[10:14]
                cmd_msg.right_btn = True
                cmd_msg.left_btn = True
                
        elif command_data["mode"] == "rot":
            cmd_msg.control_mode = "ee_position"
            action = np.array(command_data["action"], dtype=float)
            
            if command_data["arm"] == "right":
                self.main_state.desired_right_arm_quaternion = action
                cmd_msg.right_btn = True
                cmd_msg.left_btn = False
            elif command_data["arm"] == "left":
                self.main_state.desired_left_arm_quaternion = action
                cmd_msg.right_btn = False
                cmd_msg.left_btn = True
            elif command_data["arm"] == "both":
                self.main_state.desired_right_arm_quaternion = action[:4]
                self.main_state.desired_left_arm_quaternion = action[4:]
                cmd_msg.right_btn = True
                cmd_msg.left_btn = True

        elif command_data["mode"] == "gripper":
            cmd_msg.control_mode = "ee_position"
            cmd_msg.right_btn = False
            cmd_msg.left_btn = False
                
        if command_data["mode"] in ["pos", "rot"]:
            cmd_msg.desired_right_ee_pos = EEpos()
            cmd_msg.desired_right_ee_pos.position = Float32MultiArray(data=self.main_state.desired_right_arm_position.tolist())
            cmd_msg.desired_right_ee_pos.quaternion = Float32MultiArray(data=self.main_state.desired_right_arm_quaternion.tolist())
            cmd_msg.desired_left_ee_pos = EEpos()
            cmd_msg.desired_left_ee_pos.position = Float32MultiArray(data=self.main_state.desired_left_arm_position.tolist())
            cmd_msg.desired_left_ee_pos.quaternion = Float32MultiArray(data=self.main_state.desired_left_arm_quaternion.tolist())
                
        cmd_msg.desired_right_gripper_pos = self.main_state.desired_right_gripper_position
        cmd_msg.desired_left_gripper_pos = self.main_state.desired_left_gripper_position
        cmd_msg.is_active = True
        cmd_msg.ready = self.ready
        cmd_msg.move = self.move
        cmd_msg.stop = self.stop
            
        self.stop = False
        self.ready = False
        
        self.command_pub_rby1.publish(cmd_msg)

    def publish_command_hand(self, hand, opening):
        print(f"Publishing Hand command: {hand} opening to {opening}")
        cmd_msg = CommandHand()
        if hand == "right":
            cmd_msg.hand = "right"
        elif hand == "left":
            cmd_msg.hand = "left"
        else:
            self.get_logger().error(f"Unknown hand for command: {hand}")
            return
        cmd_msg.opening = opening
        
        self.command_pub_hand.publish(cmd_msg)

    def reset_state(self):
        self.main_state.desired_head_position = np.array([])
        self.main_state.desired_head_quaternion = np.array([])
        self.main_state.desired_torso_position = self.main_state.current_torso_position.copy()
        self.main_state.desired_torso_quaternion = self.main_state.current_torso_quaternion.copy()
        self.main_state.desired_right_arm_position = self.main_state.current_right_arm_position.copy()
        self.main_state.desired_right_arm_quaternion = self.main_state.current_right_arm_quaternion.copy()
        self.main_state.desired_right_gripper_position = self.main_state.current_right_gripper_position
        self.main_state.desired_left_arm_position = self.main_state.current_left_arm_position.copy()
        self.main_state.desired_left_arm_quaternion = self.main_state.current_left_arm_quaternion.copy()
        self.main_state.desired_left_gripper_position = self.main_state.current_left_gripper_position
        self.main_state.desired_joint_positions = self.main_state.current_joint_positions.copy()
        self.main_state.desired_left_arm_angle = self.main_state.current_left_arm_angle.copy()
        self.main_state.desired_right_arm_angle = self.main_state.current_right_arm_angle.copy()
        self.last_planned_state = {}
        self.action_plan.clear()
        # self.action_history = []
        # self.state_history = []
        
    def save_history_plot_png(self):
        if len(self.action_history) == 0 or len(self.state_history) == 0:
            return
        action_array = np.array(self.action_history)
        state_array = np.array(self.state_history)
        
        plt.figure(figsize=(16, 3 * action_array.shape[1]))
        
        for i in range(action_array.shape[1]):
            plt.subplot(action_array.shape[1], 1, i+1)
            plt.plot(action_array[:, i], label=f'Action {i}')
            plt.plot(state_array[:, i], label=f'State {i}', alpha=0.5)
            plt.legend()
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.grid()
        
        plt.tight_layout()
        timestamp = int(time.time())
        home_dir = str(Path.home())
        filename = f'{home_dir}/action_state_history_{timestamp}.png'
        plt.savefig(filename)
        plt.close()
        self.get_logger().info(f'Saved action/state history plot to {filename}')
        
        self.action_history = []
        self.state_history = []
        
    def step(self):
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
            self.reset_state()
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
            self.publish_command_rby1({
                "mode": "signal",
            })
            return
        
        if self.move is False:
            print("⚠️ Move mode is off. Waiting for 'f' key to start moving.")
            if len(self.action_history) > 0 and len(self.state_history) > 0:
                print("Saving action/state history plot...")
                self.save_history_plot_png()
            return
        
        if np.all(np.abs(self.main_state.current_joint_positions) < 1e-3):
            print("⚠️ Robot joints nearly zero; likely initializing. Retrying shortly.")
            time.sleep(0.05)
            return
        
        try:
            if self.is_published is False:
                self.get_logger().info("Publishing updated command based on latest action.")
                self.is_published = True
                # Publish current filtered state directly
                # The action_callback applies filter updates, and step publishes the results
                self.publish_command_rby1({
                    "mode": "pos",
                    "arm": "right",
                    "action": np.concatenate([
                        self.main_state.desired_right_arm_position,
                        self.main_state.desired_right_arm_quaternion,
                        # self.main_state.desired_left_arm_position,
                        # self.main_state.desired_left_arm_quaternion
                    ])
                })
            
            self.action_history.append(
                self.main_state.desired_right_arm_position.tolist() + self.main_state.desired_right_arm_quaternion.tolist()
            )
            self.state_history.append(
                self.main_state.current_right_arm_position.tolist() + self.main_state.current_right_arm_quaternion.tolist()
            )
                
        except KeyboardInterrupt:
            print("🛑 Interrupted (Ctrl+C)")
        except Exception as e:
            self.get_logger().error(f"Error in step: {e}")
            if self.cam1 is not None:
                self.cam1.stop()
            if self.cam2 is not None:
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
        