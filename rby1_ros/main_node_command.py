import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32, Float32, Bool, Int32MultiArray, Float32MultiArray
from rby1_interfaces.msg import EEpos, FTsensor, State, CommandRBY1, CommandHand, Action
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
from .ee_move_class import Move_ee, Rotate_ee

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
            State,
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
        
        self.pos_traj = Move_ee(Hz=self.step_hz, duration=0.3)
        self.rot_traj = Rotate_ee(Hz=self.step_hz, duration=0.3)
        
        self.action_history = []
        self.state_history = []
        self.action_traj = None
        
        self.action_map = {
            "image": [self.publish_images],
            "left_joint": [self.calc_joint_traj, "dtheta", ["left_arm_angle"]],
            "right_joint": [self.calc_joint_traj, "dtheta", ["right_arm_angle"]],
            "both_joint": [self.calc_joint_traj, "dtheta", ["right_arm_angle", "left_arm_angle"]],
            "left_pos": [self.calc_ee_pos_traj, "dpos", ["left_arm_position"]],
            "right_pos": [self.calc_ee_pos_traj, "dpos", ["right_arm_position"]],
            "both_pos": [self.calc_ee_pos_traj, "dpos", ["right_arm_position", "left_arm_position"]],
            "left_rot_local": [self.calc_ee_rot_traj, "drot", ["left_arm_quaternion"]],
            "right_rot_local": [self.calc_ee_rot_traj, "drot", ["right_arm_quaternion"]],
            "both_rot_local": [self.calc_ee_rot_traj, "drot", ["right_arm_quaternion", "left_arm_quaternion"]],
            "left_rot_global": [self.calc_ee_rot_traj, "drot", ["left_arm_quaternion"]],
            "right_rot_global": [self.calc_ee_rot_traj, "drot", ["right_arm_quaternion"]],
            "both_rot_global": [self.calc_ee_rot_traj, "drot", ["right_arm_quaternion", "left_arm_quaternion"]],
            "left_gripper": [self.set_gripper_position, "left_gripper_pos"],
            "right_gripper": [self.set_gripper_position, "right_gripper_pos"],
            "left_hand": [self.set_hand_position, "left_hand_pos"],
            "right_hand": [self.set_hand_position, "right_hand_pos"]
        }
    
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
        self.get_logger().info(f'Received action data: {msg.mode}')
        if msg.mode not in self.action_map.keys():
            self.get_logger().error(f'Unknown action mode: {msg.mode}')
            return
        
        # 대기 중인 action이 없다면, 실제 로봇 현재 상태부터 시작하도록 캐시 초기화
        # if len(self.action_plan) == 0:
        #     self.last_planned_state.clear()

        action_info = self.action_map[msg.mode]
        
        if msg.mode == "image":
            action_info[0]()
            return
        
        # 그리퍼 제어는 궤적(Trajectory)이 아닌 즉시 명령이므로 예외 처리
        if msg.mode in ["left_gripper", "right_gripper"]:
            arm = msg.mode.split("_")[0]
            opening = getattr(msg, action_info[1]).data
            self.set_gripper_position(arm, opening)
            # 그리퍼 동작은 action_plan에 넣지 않고 즉시 반영하거나, 필요하다면 별도 로직 추가
            return

        if msg.mode in ["left_hand", "right_hand"]:
            hand = msg.mode.split("_")[0]
            position = getattr(msg, action_info[1]).data
            self.set_hand_position(hand, position, duration=0.5)
            return

        action_func = action_info[0]
        action_arg1_name = action_info[1] if len(action_info) > 1 else None
        action_arg2_names = action_info[2] if len(action_info) > 2 else None
        
        action_arg1 = getattr(msg, action_arg1_name).data if action_arg1_name else None
        action_arg2 = None
        
        if action_arg2_names:
            action_arg2 = []
            for name in action_arg2_names:
                if len(self.action_plan) == 0 or msg.cancel_last_action:
                    # 계획된 궤적이 없거나, 마지막 계획 취소 요청 시 현재 상태 사용
                    self.action_plan.clear()  # 계획 취소
                    val = getattr(self.main_state, "current_" + name)
                else:
                    name = "desired_" + name
                    # [수정] 계획된 마지막 상태가 있으면 그것을 사용, 없으면 현재 센서 값 사용
                    if name in self.last_planned_state:
                        val = self.last_planned_state[name]
                    else:
                        val = getattr(self.main_state, name)
                action_arg2.extend(val if isinstance(val, list) else val.tolist() if hasattr(val, 'tolist') else [val])
            
            # numpy array로 변환 (필요시)
            action_arg2 = np.array(action_arg2)

        # 궤적 계산
        if msg.mode in ["left_rot_global", "right_rot_global", "both_rot_global"]:
            _traj = action_func(action_arg2, action_arg1, type='global')
        elif msg.mode in ["left_rot_local", "right_rot_local", "both_rot_local"]:
            _traj = action_func(action_arg2, action_arg1, type='local')
        elif 'pos' in msg.mode:
            _traj = action_func(action_arg2, action_arg1, msg.mode.split('_')[0])  # 'left', 'right', 'both' 전달
        
        if _traj is None or len(_traj) == 0:
            return

        traj = []
        for t in _traj:
            traj.append([msg.mode, t])
            
        # print(f"Planned {len(traj)} steps for {msg.mode}")
        
        self.action_plan.extend(traj)

        # [수정] 계산된 궤적의 마지막 값을 다음 시작점으로 저장 (Update last planned state)
        last_point = _traj[-1]
        
        # "both" 모드일 경우 데이터를 쪼개서 저장해야 함
        if len(action_arg2_names) == 2:  # 예: ["current_right_...", "current_left_..."]
            name_right = "desired_" + action_arg2_names[0]
            name_left = "desired_" + action_arg2_names[1]
            
            # 데이터 길이에 따라 분할 (Joint: 7, Pos: 3, Rot(Quat): 4)
            mid_idx = len(last_point) // 2
            self.last_planned_state[name_right] = last_point[:mid_idx]
            self.last_planned_state[name_left] = last_point[mid_idx:]
            
        elif len(action_arg2_names) == 1: # 단일 팔 제어
            name = "desired_" + action_arg2_names[0]
            self.last_planned_state[name] = last_point
        
        print(f"Updated last planned state: {self.last_planned_state}")

    def set_gripper_position(self, arm: str, opening: float, duration=0.5):
        if arm == "right":
            self.main_state.desired_right_gripper_position = opening
            for _ in range(int(self.step_hz * duration)):
                self.action_plan.append([f"right_gripper", opening])
        elif arm == "left":
            self.main_state.desired_left_gripper_position = opening
            for _ in range(int(self.step_hz * duration)):
                self.action_plan.append([f"left_gripper", opening])
        else:
            self.get_logger().error(f"Unknown arm for gripper: {arm}")

    def set_hand_position(self, hand: str, position: float, duration=0.5):
        action_len = int(self.step_hz * duration)
        for _ in range(action_len):
            self.action_plan.append([f"{hand}_hand", position])
        
    def calc_joint_traj(self, current_angle, dtheta):
        pass
        
    def calc_ee_pos_traj(self, last_pos, dpos, arm='right'):
        if len(dpos) > 3:
            dpos_right = dpos[:3]
            dpos_left = dpos[3:]
            traj_right = self.pos_traj.plan_move_ee('right', last_pos[:3], dpos_right, True)
            traj_left = self.pos_traj.plan_move_ee('left', last_pos[3:], dpos_left, True)
            traj = []
            for t_right, t_left in zip(traj_right, traj_left):
                traj.append(np.concatenate([t_right, t_left]))
            return traj

        traj = self.pos_traj.plan_move_ee(arm,last_pos, dpos, True)
        return traj
        
    def calc_ee_rot_traj(self, last_quat, drot, type='local'):
        if len(drot) > 3:
            drot_right = drot[:3]
            drot_left = drot[3:]
            
            axis_right = np.argmax(np.abs(drot_right))
            axis_left = np.argmax(np.abs(drot_left))
            
            drot_right = drot_right[axis_right]
            drot_left = drot_left[axis_left]
            
            axis_right = ['x', 'y', 'z'][axis_right]
            axis_left = ['x', 'y', 'z'][axis_left]

            traj_right = self.rot_traj.plan_rotate_ee(last_quat[:4], axis_right, drot_right, type=type, calc_duration=True)
            traj_left = self.rot_traj.plan_rotate_ee(last_quat[4:], axis_left, drot_left, type=type, calc_duration=True)
            traj = []
            for t_right, t_left in zip(traj_right, traj_left):
                traj.append(np.concatenate([t_right, t_left]))
            return traj
        
        axis = np.argmax(np.abs(drot))
        drot = drot[axis]
        axis = ['x', 'y', 'z'][axis]
        print(f"Rotating around axis: {axis} by {drot} degrees")

        traj = self.rot_traj.plan_rotate_ee(last_quat, axis, drot, type=type, calc_duration=True)
        return traj
    
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
        print(f"Publishing RBY1 command: {command_data}")
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
            
            if command_data["arm"] == "right":
                self.main_state.desired_right_arm_angle = command_data["action"]
                cmd_msg.right_btn = True
                cmd_msg.left_btn = False
            elif command_data["arm"] == "left":
                self.main_state.desired_left_arm_angle = command_data["action"]
                cmd_msg.right_btn = False
                cmd_msg.left_btn = True
            elif command_data["arm"] == "both":
                self.main_state.desired_right_arm_angle = command_data["action"][:7]
                self.main_state.desired_left_arm_angle = command_data["action"][7:]
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
            
            if command_data["arm"] == "right":
                self.main_state.desired_right_arm_position = command_data["action"]
                cmd_msg.right_btn = True
                cmd_msg.left_btn = False
            elif command_data["arm"] == "left":
                self.main_state.desired_left_arm_position = command_data["action"]
                cmd_msg.right_btn = False
                cmd_msg.left_btn = True
            elif command_data["arm"] == "both":
                self.main_state.desired_right_arm_position = command_data["action"][:3]
                self.main_state.desired_left_arm_position = command_data["action"][3:]
                cmd_msg.right_btn = True
                cmd_msg.left_btn = True
                
        elif command_data["mode"] == "rot":
            cmd_msg.control_mode = "ee_position"
            
            if command_data["arm"] == "right":
                self.main_state.desired_right_arm_quaternion = command_data["action"]
                cmd_msg.right_btn = True
                cmd_msg.left_btn = False
            elif command_data["arm"] == "left":
                self.main_state.desired_left_arm_quaternion = command_data["action"]
                cmd_msg.right_btn = False
                cmd_msg.left_btn = True
            elif command_data["arm"] == "both":
                self.main_state.desired_right_arm_quaternion = command_data["action"][:4]
                self.main_state.desired_left_arm_quaternion = command_data["action"][4:]
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
            
            if self.action_plan:
                action = self.action_plan.popleft()
                action_mode:str = action[0]

                if "hand" in action_mode:
                    hand = action_mode.split('_')[0]
                    opening = action[1]
                    self.publish_command_hand(hand, opening)
                    
                else:
                    arm = action_mode.split('_')[0]
                    mode = action_mode.split('_')[1]
                    self.publish_command_rby1({
                        "mode": mode,
                        "arm": arm,
                        "action": action[1]
                    })

                # 마지막 행동이면 done 퍼블리시
                if len(self.action_plan) == 0:
                    print("✅ Completed action plan.")
                    done_msg = Bool()
                    done_msg.data = True
                    self.done_pub.publish(done_msg)
                
            self.action_history.append(
                self.main_state.desired_right_arm_position.tolist() + self.main_state.desired_right_arm_quaternion.tolist()
            )
            self.state_history.append(
                self.main_state.current_right_arm_position.tolist() + self.main_state.current_right_arm_quaternion.tolist()
            )
            
                
        except KeyboardInterrupt:
            print("🛑 Interrupted (Ctrl+C)")
        except Exception as e:
            self.get_logger().error(f"Error in inference step: {e}")
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
        