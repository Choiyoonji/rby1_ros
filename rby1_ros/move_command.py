#!/usr/bin/python3
# -*- coding: utf-8 -*-
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
SCALE_POS = 0.03     # 위치 이동 스케일
SCALE_ROT = 0.12      # 회전 스케일
DEADZONE = 0.15       # 조이스틱 데드존
DEADZONE_ROT = 0.15   # 회전 입력용 데드존

# 그리퍼 설정
GRIPPER_SPEED = 0.01  # 트리거를 눌렀을 때 그리퍼가 움직이는 속도 (0.01 ~ 0.1)

# 버튼 매핑
BUTTON_LB = 4
BUTTON_RB = 5
BUTTON_ARM_SWITCH = 6     
BUTTON_CANCEL = 7         

# 축 매핑
AXIS_LS_X = 0      
AXIS_LS_Y = 1      
AXIS_LT = 2        # Left Trigger (열기)
AXIS_RS_X = 3      
AXIS_RS_Y = 4      
AXIS_RT = 5        # Right Trigger (닫기)

# 제어 모드
MODE_POS = "pos"
MODE_ROT_LOCAL = "rot_local"
MODE_GRIPPER = "gripper"

class ControllerState:
    def __init__(self):
        self.prev_buttons = {}
        self.button_pressed = {}
        
    def update(self, controller):
        self.button_pressed = {}
        for btn_idx in range(controller.get_numbuttons()):
            try:
                current = controller.get_button(btn_idx)
                prev = self.prev_buttons.get(btn_idx, False)
                self.button_pressed[btn_idx] = current and not prev
                self.prev_buttons[btn_idx] = current
            except:
                pass
    
    def is_pressed(self, button_id):
        return self.button_pressed.get(button_id, False)

class XboxControllerNode(Node):
    def __init__(self):
        super().__init__('xbox_controller_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.publisher_ = self.create_publisher(Action, '/control/action', qos_profile)
        
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

        self.ctrl_state = ControllerState()
        self.target_arm = "right"
        self.last_arm_switch_time = 0
        self.arm_switch_cooldown = 0.5
        self.rotation_mode = MODE_ROT_LOCAL

        # 그리퍼 현재 위치 기억 (1.0: 열림, 0.0: 닫힘)
        # 초기값은 안전하게 열림(1.0)으로 시작하거나, 로봇 상태를 알 수 없으니 중간값 혹은 열림으로 설정
        self.current_gripper_val = 1.0 

        self.control_rate_hz = 30
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.timer_callback)
        
        self.get_logger().info(f"Xbox Controller Node Initialized (Rate: {self.control_rate_hz} Hz)")

    def apply_deadzone(self, value, threshold=DEADZONE):
        if abs(value) < threshold:
            return 0.0
        sign = np.sign(value)
        normalized = (abs(value) - threshold) / (1.0 - threshold)
        return sign * normalized

    def normalize_trigger(self, value):
        """트리거 값(-1.0 ~ 1.0)을 0.0 ~ 1.0으로 변환"""
        val = (value + 1.0) / 2.0
        # 아주 미세한 눌림은 무시 (노이즈 방지)
        if val < 0.05:
            return 0.0
        return val

    def timer_callback(self):
        if not self.controller:
            return

        pygame.event.pump()
        self.ctrl_state.update(self.controller)

        msg = Action()
        msg.cancel_last_action = False
        
        # --- 1. 기능 버튼 ---
        if self.ctrl_state.is_pressed(BUTTON_CANCEL):
            msg.cancel_last_action = True
            msg.mode = "cancel"
            self.publisher_.publish(msg)
            self.get_logger().info("⏹️  Cancel Action")
            return

        if self.ctrl_state.is_pressed(BUTTON_ARM_SWITCH):
            if time.time() - self.last_arm_switch_time > self.arm_switch_cooldown:
                self.target_arm = "left" if self.target_arm == "right" else "right"
                self.get_logger().info(f"🔄 Arm Switched: {self.target_arm.upper()}")
                self.last_arm_switch_time = time.time()

        # --- 2. 그리퍼 제어 (증분 방식) ---
        raw_lt = self.controller.get_axis(AXIS_LT)
        raw_rt = self.controller.get_axis(AXIS_RT)
        
        lt_val = self.normalize_trigger(raw_lt) # 열기 강도 (0.0 ~ 1.0)
        rt_val = self.normalize_trigger(raw_rt) # 닫기 강도 (0.0 ~ 1.0)
        
        gripper_changed = False

        # RT(닫기)가 눌리면 값을 뺌
        if rt_val > 0:
            self.current_gripper_val -= rt_val * GRIPPER_SPEED
            gripper_changed = True
            
        # LT(열기)가 눌리면 값을 더함
        if lt_val > 0:
            self.current_gripper_val += lt_val * GRIPPER_SPEED
            gripper_changed = True

        # 값의 범위를 0.0 ~ 1.0 사이로 제한 (Clamp)
        self.current_gripper_val = max(0.0, min(1.0, self.current_gripper_val))

        # 그리퍼 입력이 있을 때만 명령 전송
        if gripper_changed:
            msg.mode = f"{self.target_arm}_gripper"
            msg.right_gripper_pos = float(self.current_gripper_val)
            # msg.left_gripper_pos = float(self.current_gripper_val)
            self.publisher_.publish(msg)
            # self.get_logger().info(f"🤚 Gripper: {self.current_gripper_val:.2f}")
            return # 그리퍼 조작 중에는 팔 이동 차단

        # --- 3. 이동 및 회전 제어 ---
        lx = self.apply_deadzone(self.controller.get_axis(AXIS_LS_X), DEADZONE)
        ly = self.apply_deadzone(self.controller.get_axis(AXIS_LS_Y), DEADZONE)
        rx = self.apply_deadzone(self.controller.get_axis(AXIS_RS_X), DEADZONE_ROT)
        ry = self.apply_deadzone(self.controller.get_axis(AXIS_RS_Y), DEADZONE)

        dx = -ly * SCALE_POS
        dy = lx * SCALE_POS
        dz = -ry * SCALE_POS

        drx = 0.0
        dry = 0.0
        drz = 0.0
        
        if self.controller.get_button(BUTTON_LB):
            drx = rx * SCALE_ROT # X축 회전 (Roll)
        elif self.controller.get_button(BUTTON_RB):
            dry = rx * SCALE_ROT # Y축 회전 (Pitch)
        else:
            drz = -rx * SCALE_ROT # Z축 회전 (Yaw)

        has_translation = np.linalg.norm([dx, dy, dz]) > 1e-5
        has_rotation = np.linalg.norm([drx, dry, drz]) > 1e-6

        if has_translation:
            msg.mode = f"{self.target_arm}_{MODE_POS}"
            dpos_array = Float32MultiArray()
            dpos_array.data = [float(dx), float(dy), float(dz)]
            msg.dpos = dpos_array
            
            drot_array = Float32MultiArray()
            drot_array.data = [0.0, 0.0, 0.0]
            msg.drot = drot_array
            self.publisher_.publish(msg)

        elif has_rotation:
            msg.mode = f"{self.target_arm}_{self.rotation_mode}"
            drot_array = Float32MultiArray()
            drot_array.data = [float(drx), float(dry), float(drz)]
            msg.drot = drot_array
            
            dpos_array = Float32MultiArray()
            dpos_array.data = [0.0, 0.0, 0.0]
            msg.dpos = dpos_array
            self.publisher_.publish(msg)

    def print_controller_status(self):
        if self.controller:
            self.get_logger().info(
                f"Controller: {self.controller.get_name()} | "
                f"Buttons: {self.controller.get_numbuttons()} | "
                f"Target: {self.target_arm.upper()}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = XboxControllerNode()
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