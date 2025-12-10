import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# 메시지 임포트
from rby1_interfaces.msg import Action
from std_msgs.msg import Float32MultiArray

import pygame
import time
import numpy as np

# ==================== 조이스틱 설정 ====================
# 응답 특성 (조이스틱 값 -1.0 ~ 1.0에 곱해질 계수)
SCALE_POS = 0.005     # 위치 이동 스케일 (미터 단위)
SCALE_ROT = 0.03      # 회전 스케일 (라디안 단위)
DEADZONE = 0.1        # 조이스틱 데드존 (노이즈 방지, 더 엄격함)
DEADZONE_ROT = 0.15   # 회전 입력용 별도 데드존

# 버튼 매핑
BUTTON_GRIPPER_CLOSE = 0  # A 버튼
BUTTON_GRIPPER_OPEN = 1   # B 버튼
BUTTON_ARM_SWITCH = 6     # Back/Select 버튼
BUTTON_CANCEL = 7         # Start 버튼

# 축 매핑 (Xbox Controller 표준)
AXIS_LS_X = 0      # Left Stick X (좌우)
AXIS_LS_Y = 1      # Left Stick Y (상하)
AXIS_RS_X = 3      # Right Stick X (좌우)
AXIS_RS_Y = 4      # Right Stick Y (상하)

# 제어 모드
MODE_POS = "pos"
MODE_ROT_LOCAL = "rot_local"
MODE_ROT_GLOBAL = "rot_global"
MODE_GRIPPER = "gripper"

# ==================== 컨트롤러 상태 ====================
class ControllerState:
    """Xbox 컨트롤러 입력 상태 추적"""
    def __init__(self):
        self.prev_buttons = {}  # 이전 프레임 버튼 상태
        self.button_pressed = {}  # 이번 프레임에 새로 눌린 버튼
        
    def update(self, controller):
        """컨트롤러 상태 업데이트"""
        self.button_pressed = {}
        for btn_idx in range(11):  # Xbox 컨트롤러는 약 11개 버튼
            try:
                current = controller.get_button(btn_idx)
                prev = self.prev_buttons.get(btn_idx, False)
                self.button_pressed[btn_idx] = current and not prev  # 새로 눌린 경우
                self.prev_buttons[btn_idx] = current
            except:
                pass
    
    def is_pressed(self, button_id):
        """버튼이 이 프레임에 눌렸는가?"""
        return self.button_pressed.get(button_id, False)
    
    def is_held(self, button_id, controller):
        """버튼이 계속 눌려있는가?"""
        try:
            return controller.get_button(button_id)
        except:
            return False

class XboxControllerNode(Node):
    def __init__(self):
        super().__init__('xbox_controller_node')

        # QoS 설정 (받는 쪽이 Reliable QoS를 사용하므로 일치시킴)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.publisher_ = self.create_publisher(Action, '/control/action', qos_profile)
        
        # Pygame 및 조이스틱 초기화
        pygame.init()
        pygame.joystick.init()
        
        self.controller = None
        if pygame.joystick.get_count() > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            self.get_logger().info(f"✓ Xbox Controller Connected: {self.controller.get_name()}")
        else:
            self.get_logger().error("✗ No Xbox controller found! Please connect one.")
            return

        # 컨트롤러 상태 추적
        self.ctrl_state = ControllerState()

        # 현재 제어 중인 팔 (기본값: right)
        self.target_arm = "right"
        self.last_arm_switch_time = 0
        self.arm_switch_cooldown = 0.5  # 팔 전환 쿨다운 (초)

        # 회전 모드 (기본값: local)
        self.rotation_mode = MODE_ROT_LOCAL
        self.last_rot_mode_switch_time = 0
        self.rot_mode_cooldown = 0.3

        # 입력 이력 (부드러운 입력용)
        self.prev_input = {
            'pos': np.zeros(3),
            'rot': np.zeros(3)
        }
        self.input_smoothing_alpha = 0.3  # Exponential moving average factor

        # 제어 빈도 (30 Hz)
        self.control_rate_hz = 30
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.timer_callback)
        
        self.get_logger().info(f"Xbox Controller Node Initialized (Rate: {self.control_rate_hz} Hz)")

    def apply_deadzone(self, value, threshold=DEADZONE):
        """데드존 적용 및 선형화"""
        if abs(value) < threshold:
            return 0.0
        # 데드존 이상의 값에 대해 선형 스케일링
        sign = np.sign(value)
        normalized = (abs(value) - threshold) / (1.0 - threshold)
        return sign * normalized

    def apply_exponential_curve(self, value, power=1.5):
        """조이스틱 입력에 곡선 적용 (정밀도 향상)"""
        if abs(value) < 0.1:
            return value
        sign = np.sign(value)
        return sign * (abs(value) ** power)

    def smooth_input(self, current, previous, alpha=None):
        """지수 이동 평균을 사용한 입력 평활화"""
        if alpha is None:
            alpha = self.input_smoothing_alpha
        return alpha * current + (1.0 - alpha) * previous

    def timer_callback(self):
        if not self.controller:
            return

        # Pygame 이벤트 펌프 (입력 갱신)
        pygame.event.pump()

        # 컨트롤러 상태 업데이트 (버튼 눌림 감지)
        self.ctrl_state.update(self.controller)

        msg = Action()
        msg.cancel_last_action = False
        
        # --- 1. 버튼 입력 처리 (그리퍼 및 모드 변경) ---
        
        # A 버튼: 그리퍼 닫기
        if self.ctrl_state.is_pressed(BUTTON_GRIPPER_CLOSE):
            msg.mode = f"{self.target_arm}_gripper"
            msg.right_gripper_pos = Float32MultiArray(data=[0.0])  # Fully closed
            msg.left_gripper_pos = Float32MultiArray(data=[0.0])
            self.publisher_.publish(msg)
            self.get_logger().info(f"🤚 Gripper: CLOSE ({self.target_arm})")
            return

        # B 버튼: 그리퍼 열기
        if self.ctrl_state.is_pressed(BUTTON_GRIPPER_OPEN):
            msg.mode = f"{self.target_arm}_gripper"
            msg.right_gripper_pos = Float32MultiArray(data=[1.0])  # Fully open
            msg.left_gripper_pos = Float32MultiArray(data=[1.0])
            self.publisher_.publish(msg)
            self.get_logger().info(f"🤚 Gripper: OPEN ({self.target_arm})")
            return

        # Start 버튼: 취소 마지막 액션
        if self.ctrl_state.is_pressed(BUTTON_CANCEL):
            msg.cancel_last_action = True
            msg.mode = "cancel"
            self.publisher_.publish(msg)
            self.get_logger().info("⏹️  Cancel: Last action cancelled")
            return

        # Back/Select 버튼: 팔 전환 (Left ↔ Right)
        if self.ctrl_state.is_pressed(BUTTON_ARM_SWITCH):
            if time.time() - self.last_arm_switch_time > self.arm_switch_cooldown:
                self.target_arm = "left" if self.target_arm == "right" else "right"
                self.get_logger().info(f"🔄 Arm Switched: {self.target_arm.upper()}")
                self.last_arm_switch_time = time.time()

        # --- 2. 아날로그 스틱 입력 처리 (Move & Rotate) ---
        
        # 원본 스틱 값 읽기
        raw_ls_x = self.controller.get_axis(AXIS_LS_X)
        raw_ls_y = self.controller.get_axis(AXIS_LS_Y)
        raw_rs_x = self.controller.get_axis(AXIS_RS_X)
        raw_rs_y = self.controller.get_axis(AXIS_RS_Y)

        # 데드존 처리 (선형화)
        lx = self.apply_deadzone(raw_ls_x, DEADZONE)
        ly = self.apply_deadzone(raw_ls_y, DEADZONE)
        rx = self.apply_deadzone(raw_rs_x, DEADZONE_ROT)
        ry = self.apply_deadzone(raw_rs_y, DEADZONE)

        # 선택적: 정밀도 향상을 위한 곡선 적용 (커멘트 처리)
        # lx = self.apply_exponential_curve(lx, power=1.3)
        # ly = self.apply_exponential_curve(ly, power=1.3)
        # rx = self.apply_exponential_curve(rx, power=1.2)
        # ry = self.apply_exponential_curve(ry, power=1.3)

        # 위치 이동 계산 [x, y, z]
        dx = -ly * SCALE_POS  # Left Stick Y(상하) -> X축(전진/후진)
        dy = -lx * SCALE_POS  # Left Stick X(좌우) -> Y축(좌우이동)
        dz = 0.0              # Z축은 수동으로 처리하거나 다른 컨트롤러 축 사용

        # 회전 이동 계산 [rx, ry, rz]
        drx = 0.0               # Right Stick Y -> X축 회전 (옵션)
        dry = 0.0               # Right Stick X -> Y축 회전 (옵션)
        drz = -rx * SCALE_ROT   # Right Stick X -> Z축 회전 (Roll)

        # 입력이 있는지 확인 (데드존보다 큼)
        has_translation = np.linalg.norm([dx, dy, dz]) > 1e-6
        has_rotation = np.linalg.norm([drx, dry, drz]) > 1e-6

        # 입력 평활화 (선택적)
        # current_pos = np.array([dx, dy, dz])
        # current_rot = np.array([drx, dry, drz])
        # smoothed_pos = self.smooth_input(current_pos, self.prev_input['pos'])
        # smoothed_rot = self.smooth_input(current_rot, self.prev_input['rot'])
        # self.prev_input['pos'] = smoothed_pos
        # self.prev_input['rot'] = smoothed_rot

        # --- 3. 메시지 작성 및 전송 ---
        
        if has_translation:
            # 위치 제어 모드
            msg.mode = f"{self.target_arm}_{MODE_POS}"
            
            # dpos 채우기
            dpos_array = Float32MultiArray()
            dpos_array.data = [float(dx), float(dy), float(dz)]
            msg.dpos = dpos_array
            
            # drot는 0으로
            drot_array = Float32MultiArray()
            drot_array.data = [0.0, 0.0, 0.0]
            msg.drot = drot_array
            
            self.publisher_.publish(msg)
            # self.get_logger().info(f"➡️  Pos: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

        elif has_rotation:
            # 회전 제어 모드 (local이 기본)
            msg.mode = f"{self.target_arm}_{self.rotation_mode}"
            
            # drot 채우기
            drot_array = Float32MultiArray()
            drot_array.data = [float(drx), float(dry), float(drz)]
            msg.drot = drot_array
            
            # dpos는 0으로
            dpos_array = Float32MultiArray()
            dpos_array.data = [0.0, 0.0, 0.0]
            msg.dpos = dpos_array
            
            self.publisher_.publish(msg)
            # self.get_logger().info(f"🔄 Rot ({self.rotation_mode}): drx={drx:.4f}, dry={dry:.4f}, drz={drz:.4f}")

    def get_controller_info(self):
        """컨트롤러 정보 출력"""
        if not self.controller:
            return None
        
        info = {
            'name': self.controller.get_name(),
            'num_buttons': self.controller.get_numbuttons(),
            'num_axes': self.controller.get_numaxes(),
            'num_hats': self.controller.get_numhats(),
            'target_arm': self.target_arm,
            'rotation_mode': self.rotation_mode,
        }
        return info
    
    def print_controller_status(self):
        """컨트롤러 상태 출력"""
        info = self.get_controller_info()
        if info:
            self.get_logger().info(
                f"Controller: {info['name']} | "
                f"Buttons: {info['num_buttons']} | "
                f"Axes: {info['num_axes']} | "
                f"Target Arm: {info['target_arm'].upper()} | "
                f"Rotation Mode: {info['rotation_mode'].upper()}"
            )
    
    def get_axis_info(self):
        """현재 축 값 정보 반환 (디버깅용)"""
        if not self.controller:
            return None
        
        try:
            return {
                'axis_0_ls_x': self.controller.get_axis(AXIS_LS_X),
                'axis_1_ls_y': self.controller.get_axis(AXIS_LS_Y),
                'axis_3_rs_x': self.controller.get_axis(AXIS_RS_X),
                'axis_4_rs_y': self.controller.get_axis(AXIS_RS_Y),
            }
        except:
            return None


def main(args=None):
    rclpy.init(args=args)
    node = XboxControllerNode()
    
    # 컨트롤러 정보 출력
    node.print_controller_status()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()


if __name__ == '__main__':
    main()