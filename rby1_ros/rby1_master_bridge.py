# Author: acewnd

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import rby1_sdk as rby

READY_POS = [0.0, -15.0, 0.0, -120.0, 0.0, 30.0, -15.0]

class MasterArmBridge(Node):
    def __init__(self):
        super().__init__('master_arm_bridge')
        self.declare_parameter('loop_hz', 100.0)
        self.declare_parameter('model_path', '/home/acewnd/rby1_ros-main/rby1_ros/models/master_arm/model.urdf')
        self.declare_parameter('device_name', 'MasterArmDevice')  # rby.upc.MasterArmDeviceName 사용 가능
        self.declare_parameter('publish_velocity', False)

        hz = float(self.get_parameter('loop_hz').value)
        self.dt = 1.0 / hz

        # pubs
        self.pub_states = self.create_publisher(Float32MultiArray, '/control/action', 10)

        # init master arm
        dev = self.get_parameter('device_name').value
        rby.upc.initialize_device(dev)
        self.master_arm = rby.upc.MasterArm(dev)
        model_path = self.get_parameter('model_path').value
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), '..', model_path)
        self.master_arm.set_model_path(model_path)
        self.master_arm.set_control_period(self.dt)

        active_ids = self.master_arm.initialize(verbose=False)
        if len(active_ids) != rby.upc.MasterArm.DeviceCount:
            self.get_logger().error(f'Active devices mismatch: {active_ids}')
            raise RuntimeError('MasterArm device mismatch')

        # 제어 입력: 중력보상/점성/토크 제한 등 (원본과 동일한 상수)
        self.ma_q_limit_barrier = 0.5
        self.ma_min_q = np.deg2rad([-360,-30, 0,-135,-90,35,-360, -360,-360,10,-90,-135,-90,35,-360])[:14]
        self.ma_max_q = np.deg2rad([ 360,-10,90, -60, 90,80, 360,  360, 360,30,  0, -60,  90,80, 360])[:14]
        self.ma_torque_limit = np.array([3.5,3.5,3.5,1.5,1.5,1.5,1.5]*2)
        self.ma_viscous_gain = np.array([0.02,0.02,0.02,0.02,0.01,0.01,0.002]*2)

        # 초기 위치로 이동
        self.master_arm.start_control(self.master_arm_start_control_callback)
        
        self.prev_q = None
        self.timer = self.create_timer(self.dt, self.loop)

        # 마스터암 내부 제어 루프 실행 (원본은 콜백 기반이었으나, 여기선 polling)
        # 필요 시 self.master_arm.start_control(...) 로 내부 안정화 가능
    
    def master_arm_start_control_callback(self):
        right_q = np.deg2rad(READY_POS)
        left_q = np.deg2rad(READY_POS)
        ma_input = rby.upc.MasterArm.ControlInput()
        ma_input.target_operating_mode[0:7].fill(
            rby.DynamixelBus.CurrentBasedPositionControlMode
        )
        ma_input.target_position[0:7] = right_q
        ma_input.target_torque[0:7] = self.ma_torque_limit[0:7]
        ma_input.target_operating_mode[7:14].fill(
            rby.DynamixelBus.CurrentBasedPositionControlMode
        )
        ma_input.target_position[7:14] = left_q
        ma_input.target_torque[7:14] = self.ma_torque_limit[7:14]
        self.master_arm.send_control_input(ma_input)
        self.prev_q = np.concatenate([right_q, left_q])

    def master_arm_control_callback(self, state: rby.upc.MasterArm.State):
        ma_input = rby.upc.MasterArm.ControlInput()
        
        torque = (
            state.gravity_term
            + self.ma_q_limit_barrier
            * (
                np.maximum(self.ma_min_q - state.q_joint, 0)
                + np.minimum(self.ma_max_q - state.q_joint, 0)
            )
            + self.ma_viscous_gain * state.qvel_joint
        )
        torque = np.clip(torque, -self.ma_torque_limit, self.ma_torque_limit)
        if state.button_right.button == 1:
            ma_input.target_operating_mode[0:7].fill(
                rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[0:7] = torque[0:7] * 0.6
            right_q = state.q_joint[0:7]
        else:
            ma_input.target_operating_mode[0:7].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[0:7] = self.ma_torque_limit[0:7]
            ma_input.target_position[0:7] = right_q

        if state.button_left.button == 1:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[7:14] = torque[7:14] * 0.6
            left_q = state.q_joint[7:14]
        else:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[7:14] = self.ma_torque_limit[7:14]
            ma_input.target_position[7:14] = left_q
        
        # q = robot_q.copy() # rby1의 현재 상태를 받아야됨.
        # q[model.right_arm_idx] = right_q
        # q[model.left_arm_idx] = left_q
        # dyn_state.set_q(q)
        # dyn_model.compute_forward_kinematics(dyn_state)
        # is_collision = (
        #     dyn_model.detect_collisions_or_nearest_links(dyn_state, 1)[0].distance
        #     < 0.02
        # )
        
        return ma_input
        
    def loop(self):
        state = self.master_arm.read_state()  # 가정: SDK에 동기 상태 읽기 API가 있음 (없다면 start_control 콜백 패턴 사용)
        if state is None:
            return
        
        self.master_arm.start_control(self.master_arm_control_callback)
        
        # 버튼/트리거 0~1 정규화
        right_joint = float(state.q_joint[:7])
        left_joint = float(state.q_joint[7:])
        right_btn = float(state.button_right.button)
        left_btn  = float(state.button_left.button)
        right_trg = float(state.button_right.trigger) / 1000.0
        left_trg  = float(state.button_left.trigger) / 1000.0

        right_actions = [right_btn, right_joint, right_trg]
        left_actions  = [left_btn,  left_joint,  left_trg]

        master_actions = Float32MultiArray()
        master_actions.data = right_actions
        self.pub_states.publish(master_actions)

def main():
    rclpy.init()
    node = MasterArmBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()