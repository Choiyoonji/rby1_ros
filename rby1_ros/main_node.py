import numpy as np
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32, Bool, Int32MultiArray, Float32MultiArray
from rby1_interfaces.msg import EEpos, FTsensor, State, Command
from rby1_interfaces.srv import MetaInitialReq, MetaDataReq

import cv2
from rby1_ros.main_status import MainStatus as MainState
from rby1_ros.rby1_dyn import RBY1Dyn
from rby1_ros.utils import *

class MainNode(Node):
    def __init__(self):
        super().__init__('main_node')

        self.ready: bool = False
        self.move: bool = False
        self.stop: bool = False
        self.record: bool = False

        self._awaiting_meta = False
        self._meta_future = None
        self._last_meta_req_ts = 0.0
        self._meta_timeout = 5.0      # 서비스 응답 타임아웃(초)
        self._cooldown = 1.0          # 실패/타임아웃 후 최소 대기(초)

        self.torso_pos_weight = 0.2

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
        
        self.head_pub = self.create_publisher(
            Float32MultiArray,
            '/control/head_command',
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
        self.meta_timer = self.create_timer(1/50.0, self.meta_loop)
        self.command_timer = self.create_timer(1/20.0, self.publish_command)
        self.head_command_timer = self.create_timer(1/50.0, self.head_command)

        # 초기화 서비스
        self._awaiting_meta_init = False
        self._meta_init_future = None
        self._meta_init_last_ts = 0.0
        self._meta_init_timeout = 5.0      # 초기화 응답 타임아웃 (초)
        self._meta_init_cooldown = 1.0     # 초기화 실패/타임아웃 후 재시도 쿨다운 (초)

        # 데이터 서비스
        self._awaiting_meta_data = False
        self._meta_data_future = None
        self._meta_data_last_ts = 0.0
        self._meta_data_timeout = 2.0      # 데이터 응답 타임아웃 (초)
        self._meta_data_cooldown = 0.0     # 데이터는 주기 폴링 성격 → 0으로 두고 single-flight만 보장

        
        self.command_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
        cv2.namedWindow("control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("control", 480, 640)

    def send_meta_initial_offset(self, check_pose: bool):
        self.init_req.initialize = True
        # self.init_req.check_pose = check_pose
        self.init_req.check_pose =  False

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
    
    def head_command(self):
        if not self.main_state.is_robot_connected:
            return
        if not self.main_state.is_meta_ready:
            return
        if not self.move:
            return
        head_cmd_msg = Float32MultiArray()

        T_base2head = pos_to_se3(self.main_state.desired_head_position, self.main_state.desired_head_quaternion)
        T_base2torso = pos_to_se3(self.main_state.current_torso_position, self.main_state.current_torso_quaternion)

        T_torso2head = np.linalg.pinv(T_base2torso) @ T_base2head
        
        head_pos_in_torso, head_quat_in_torso = se3_to_pos_quat(T_torso2head)
        head_cmd_msg.data = head_pos_in_torso.tolist() + head_quat_in_torso.tolist()
        self.head_pub.publish(head_cmd_msg)

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
        self.main_state.desired_torso_quaternion = np.array(msg.torso_ee_pos.quaternion.data) # torso의 orientation은 그대로 유지
        self.main_state.current_right_arm_position = np.array(msg.right_ee_pos.position.data)
        self.main_state.current_right_arm_quaternion = np.array(msg.right_ee_pos.quaternion.data)
        self.main_state.current_left_arm_position = np.array(msg.left_ee_pos.position.data)
        self.main_state.current_left_arm_quaternion = np.array(msg.left_ee_pos.quaternion.data)

    def publish_command(self):
        if self.main_state.is_robot_connected:
            command_msg = Command()

            command_msg.is_active = self.main_state.is_meta_ready
            command_msg.control_mode = "joint_position"

            command_msg.desired_torso_ee_pos = EEpos()

            command_msg.desired_torso_ee_pos.position = Float32MultiArray(data=self.main_state.desired_torso_position.tolist())
            command_msg.desired_torso_ee_pos.quaternion = Float32MultiArray(data=self.main_state.desired_torso_quaternion.tolist())

            command_msg.desired_right_ee_pos = EEpos()
            command_msg.desired_right_ee_pos.position = Float32MultiArray(data=self.main_state.desired_right_arm_position.tolist())
            command_msg.desired_right_ee_pos.quaternion = Float32MultiArray(data=self.main_state.desired_right_arm_quaternion.tolist())

            command_msg.desired_left_ee_pos = EEpos()
            command_msg.desired_left_ee_pos.position = Float32MultiArray(data=self.main_state.desired_left_arm_position.tolist())
            command_msg.desired_left_ee_pos.quaternion = Float32MultiArray(data=self.main_state.desired_left_arm_quaternion.tolist())

            if command_msg.control_mode == "joint_position":
                current_joint_positions = np.array(self.main_state.current_joint_positions)
                target_T = {
                    'link_torso_5': pos_to_se3(self.main_state.desired_torso_position, self.main_state.desired_torso_quaternion),
                    'link_right_6': pos_to_se3(self.main_state.desired_right_arm_position, self.main_state.desired_right_arm_quaternion),
                    'link_left_6': pos_to_se3(self.main_state.desired_left_arm_position, self.main_state.desired_left_arm_quaternion)
                }
                joint_positions = RBY1Dyn().get_ik(target_T, current_joint_positions)
                self.main_state.desired_joint_positions = joint_positions  # 기본값 설정
                command_msg.desired_joint_positions = Float32MultiArray(data=self.main_state.desired_joint_positions.tolist())

            command_msg.estop = False

            command_msg.ready = self.ready
            command_msg.move = self.move
            command_msg.stop = self.stop

            self.stop = False
            self.ready = False

            self.command_pub.publish(command_msg)

    def meta_loop(self):
        # 동작 조건 미충족 시 빠르게 리턴 (타이머 콜백은 짧게!)
        if not self.move:
            self.main_state.is_meta_initialized = False
            self.main_state.is_meta_ready = False
            return

        self.get_logger().info('Meta loop executing')

        if not self.main_state.is_robot_connected:
            self.get_logger().warning('Robot is not connected')
            return

        if not self.main_state.is_robot_initialized:
            self.get_logger().warning('Robot is not initialized')
            return

        now = time.time()

        # -------------------------------
        # (A) 초기화 상태머신
        # -------------------------------
        if not self.main_state.is_meta_initialized:
            # 요청 중이 아니고 쿨다운 지났으면 → 1회 발사
            if (not self._awaiting_meta_init) and (now - self._meta_init_last_ts >= self._meta_init_cooldown):
                self.get_logger().info('Meta initialization requested.')
                try:
                    self._meta_init_future = self.send_meta_initial_offset(
                        check_pose=self.main_state.is_robot_in_ready_pose
                    )
                except Exception as e:
                    self.get_logger().error(f'call_async(set_offset) failed: {repr(e)}')
                    self._meta_init_last_ts = now
                    return
                self._awaiting_meta_init = True
                self._meta_init_last_ts = now
                return  # 즉시 리턴(블로킹 금지)

            # 요청 중이면 완료/타임아웃만 확인
            if self._awaiting_meta_init:
                # 타임아웃
                if (now - self._meta_init_last_ts) > self._meta_init_timeout:
                    self.get_logger().error('Meta initialization timed out.')
                    try:
                        self._meta_init_future.cancel()
                    except Exception:
                        pass
                    self._awaiting_meta_init = False
                    self._meta_init_future = None
                    self._meta_init_last_ts = now  # 쿨다운 시작
                    return

                # 완료
                if self._meta_init_future.done():
                    try:
                        resp = self._meta_init_future.result()
                    except Exception as e:
                        self.get_logger().error(f'Meta init future exception: {repr(e)}')
                        self._awaiting_meta_init = False
                        self._meta_init_future = None
                        self._meta_init_last_ts = now
                        return

                    ok = bool(getattr(resp, 'success', False))
                    c1 = bool(getattr(resp, 'check1', False))
                    c2 = bool(getattr(resp, 'check2', False))
                    c3 = bool(getattr(resp, 'check3', False))
                    self.get_logger().info(
                        f'Meta initialization response received. success={ok}, checks=({c1},{c2},{c3})'
                    )

                    if ok:
                        self.get_logger().info('Meta initialized successfully')
                        self.main_state.is_meta_initialized = True
                        # ready pose 소비
                        self.main_state.is_robot_in_ready_pose = False
                    else:
                        self.get_logger().warning('Meta initialization returned success=False')

                    self._awaiting_meta_init = False
                    self._meta_init_future = None
                    self._meta_init_last_ts = now  # 성공/실패 모두 쿨다운 시작
                    return

            # 아직 초기화 안 끝났으면 데이터 요청 안 함
            return

        # -------------------------------
        # (B) 데이터 상태머신 (초기화 후에만)
        # -------------------------------
        self.get_logger().info('Meta data requested.')

        # 요청 중이 아니면 1회 발사
        if (not self._awaiting_meta_data) and (now - self._meta_data_last_ts >= self._meta_data_cooldown):
            try:
                self._meta_data_future = self.send_meta_get_data()
            except Exception as e:
                self.get_logger().error(f'call_async(get_data) failed: {repr(e)}')
                self._meta_data_last_ts = now
                return
            self._awaiting_meta_data = True
            self._meta_data_last_ts = now
            return  # 즉시 리턴

        if self._awaiting_meta_data:
            # 타임아웃
            if (now - self._meta_data_last_ts) > self._meta_data_timeout:
                self.get_logger().error('Meta data timed out.')
                try:
                    self._meta_data_future.cancel()
                except Exception:
                    pass
                self._awaiting_meta_data = False
                self._meta_data_future = None
                self._meta_data_last_ts = now
                self.main_state.is_meta_ready = False
                return

            # 완료
            if self._meta_data_future.done():
                try:
                    response = self._meta_data_future.result()
                except Exception as e:
                    self.get_logger().error(f'Meta data future exception: {repr(e)}')
                    self._awaiting_meta_data = False
                    self._meta_data_future = None
                    self._meta_data_last_ts = now
                    self.main_state.is_meta_ready = False
                    return

                if response is None:
                    self.get_logger().error('Failed to call Meta data service (None)')
                    self._awaiting_meta_data = False
                    self._meta_data_future = None
                    self._meta_data_last_ts = now
                    self.main_state.is_meta_ready = False
                    return

                if len(response.error_msg) > 0:
                    self.get_logger().error(f'Meta data error: {response.error_msg}')
                    self._awaiting_meta_data = False
                    self._meta_data_future = None
                    self._meta_data_last_ts = now
                    self.main_state.is_meta_ready = False
                    return

                # 정상 수신 → 상태 업데이트
                self.get_logger().info('Meta data received successfully')
                self.main_state.is_meta_ready = True
                self.main_state.desired_head_position = np.array(response.head_ee_pos.position.data)
                self.main_state.desired_torso_position = self.torso_pos_weight * np.array(response.head_ee_pos.position.data) # head pos의 일부만 torso로 맞춤
                self.main_state.desired_head_quaternion = np.array(response.head_ee_pos.quaternion.data)
                self.main_state.desired_right_arm_position = np.array(response.right_ee_pos.position.data)
                self.main_state.desired_right_arm_quaternion = np.array(response.right_ee_pos.quaternion.data)
                self.main_state.desired_left_arm_position = np.array(response.left_ee_pos.position.data)
                self.main_state.desired_left_arm_quaternion = np.array(response.left_ee_pos.quaternion.data)

                self._awaiting_meta_data = False
                self._meta_data_future = None
                self._meta_data_last_ts = now
                return

    def reset_state(self):
        # 초기화/데이터 상태머신도 함께 리셋
        self._awaiting_meta_init = False
        self._meta_init_future = None
        self._awaiting_meta_data = False
        self._meta_data_future = None

        self.main_state.is_meta_initialized = False
        self.main_state.is_meta_ready = False

        self.main_state.desired_head_position = np.array([])
        self.main_state.desired_head_quaternion = np.array([])
        self.main_state.desired_torso_position = np.array([])
        self.main_state.desired_torso_quaternion = np.array([])
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