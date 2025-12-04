import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# 사용자 인터페이스 메시지 임포트
from rby1_interfaces.msg import EEpos, FTsensor, State, Command, Action
from std_msgs.msg import Float32MultiArray

import pygame
import time

# 민감도 설정 (조이스틱 값 -1.0 ~ 1.0에 곱해질 계수)
SCALE_POS = 0.005  # 위치 이동 스케일 (미터 단위 추정)
SCALE_ROT = 0.005   # 회전 속도 스케일 (라디안 단위 추정)
DEADZONE = 0.5    # 조이스틱 데드존 (노이즈 방지)

class XboxControllerNode(Node):
    def __init__(self):
        super().__init__('xbox_controller_node')

        # QoS 설정 (받는 쪽이 qos_cmd를 사용하므로 Reliable로 설정)
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
            self.get_logger().info(f"Connected to controller: {self.controller.get_name()}")
        else:
            self.get_logger().error("No Xbox controller found! Please connect one.")

        # 현재 제어 중인 팔 (기본값: right)
        self.target_arm = "right"
        self.last_arm_switch_time = 0

        # 주기적으로 입력을 체크하고 퍼블리시 (30Hz)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)

    def apply_deadzone(self, value):
        if abs(value) < DEADZONE:
            return 0.0
        return value

    def timer_callback(self):
        if not self.controller:
            return

        # Pygame 이벤트 펌프 (입력 갱신)
        pygame.event.pump()

        msg = Action()
        msg.cancel_last_action = False
        
        # --- 1. 버튼 입력 처리 (그리퍼 및 모드 변경) ---
        
        # A 버튼: 그리퍼 닫기 (0번 버튼 가정, 매핑 확인 필요)
        if self.controller.get_button(0): 
            msg.mode = f"{self.target_arm}_gripper_close"
            self.publisher_.publish(msg)
            self.get_logger().info(f"Action: {msg.mode}")
            return # 동시 입력 방지를 위해 리턴

        # B 버튼: 그리퍼 열기 (1번 버튼 가정)
        if self.controller.get_button(1):
            msg.mode = f"{self.target_arm}_gripper_open"
            self.publisher_.publish(msg)
            self.get_logger().info(f"Action: {msg.mode}")
            return

        # Start 버튼: 취소 (7번 버튼 가정)
        if self.controller.get_button(7):
            msg.cancel_last_action = True
            msg.mode = "cancel" # 모드는 더미로 넣음
            self.publisher_.publish(msg)
            self.get_logger().info("Action: Cancel Last Action")
            time.sleep(0.5) # 디바운싱
            return

        # Select/Back 버튼: 팔 전환 (6번 버튼 가정)
        if self.controller.get_button(6):
            if time.time() - self.last_arm_switch_time > 0.5:
                self.target_arm = "left" if self.target_arm == "right" else "right"
                self.get_logger().info(f"Switched target arm to: {self.target_arm}")
                self.last_arm_switch_time = time.time()

        # --- 2. 아날로그 스틱 입력 처리 (Move & Rotate) ---
        
        # 축 매핑 (Xbox Controller 표준)
        # Left Stick X: Axis 0 (좌우) -> dy (로봇 기준 횡이동)
        # Left Stick Y: Axis 1 (상하) -> dx (로봇 기준 전진/후진) * Y축은 위가 -1이므로 반전 필요
        # Right Stick Y: Axis 3 (상하) -> dz (로봇 기준 높이) * 반전 필요
        # Right Stick X: Axis 2 (좌우) -> drz (Z축 회전)

        raw_ls_x = self.controller.get_axis(0)
        raw_ls_y = self.controller.get_axis(1)
        raw_rs_x = self.controller.get_axis(3)
        raw_rs_y = self.controller.get_axis(4) # 윈도우/리눅스 드라이버에 따라 Axis 번호가 3 또는 4일 수 있음

        # 데드존 처리
        lx = self.apply_deadzone(raw_ls_x)
        ly = self.apply_deadzone(raw_ls_y)
        rx = self.apply_deadzone(raw_rs_x)
        ry = self.apply_deadzone(raw_rs_y)

        # 값 변환
        dx = -ly * SCALE_POS  # Stick Y Up -> Robot Forward (+x)
        dy = -lx * SCALE_POS  # Stick X Left -> Robot Left (+y) (좌표계에 따라 부호 조정 필요)
        dz = -ry * SCALE_POS  # Stick R Y Up -> Robot Up (+z)
        drz = -rx * SCALE_ROT # Stick R X Left -> Rotate CCW (+z rot)

        # 입력이 있는지 확인
        has_translation = (abs(dx) > 0 or abs(dy) > 0 or abs(dz) > 0)
        has_rotation = (abs(drz) > 0)

        # 받는 쪽 노드(main_node_command.py)는 mode에 따라 분기하므로
        # 위치 이동과 회전을 동시에 보내기보다 우선순위를 두어 보냅니다.
        # (여기서는 위치 이동을 우선함)

        if has_translation:
            msg.mode = f"{self.target_arm}_pos"
            
            # dpos 데이터 채우기 [x, y, z]
            dpos_array = Float32MultiArray()
            dpos_array.data = [float(dx), float(dy), float(dz)]
            msg.dpos = dpos_array
            
            # drot는 비워두거나 0으로 초기화
            drot_dummy = Float32MultiArray()
            drot_dummy.data = [0.0, 0.0, 0.0]
            msg.drot = drot_dummy
            
            self.publisher_.publish(msg)
            # self.get_logger().info(f"Pub Pos: {dx:.3f}, {dy:.3f}, {dz:.3f}")

        elif has_rotation:
            msg.mode = f"{self.target_arm}_rot"
            
            # drot 데이터 채우기 [rx, ry, rz]
            # 받는 쪽 코드에서 axis = np.argmax(np.abs(drot)) 로 축을 결정하므로
            # Z축 회전을 하려면 Z성분(인덱스 2)에 가장 큰 값을 넣어야 함
            drot_array = Float32MultiArray()
            drot_array.data = [0.0, 0.0, float(drz)] 
            msg.drot = drot_array
            
            # dpos는 0으로
            dpos_dummy = Float32MultiArray()
            dpos_dummy.data = [0.0, 0.0, 0.0]
            msg.dpos = dpos_dummy
            
            self.publisher_.publish(msg)
            # self.get_logger().info(f"Pub Rot: {drz:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = XboxControllerNode()
    
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