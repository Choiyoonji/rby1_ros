# Author: acewnd

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import rby1_sdk as rby
from .trajectory import Trajectory
from pynput import keyboard

READY_POS_R = np.deg2rad([0.0, -15.0, 0.0, -120.0, 0.0, 30.0, 75.0])
READY_POS_L = np.deg2rad([0.0, 15.0, 0.0, -120.0, 0.0, 30.0, -75.0])

class MasterArmBridge(Node):
    def __init__(self):
        super().__init__('master_arm_bridge')
        self.declare_parameter('loop_hz', 100.0)
        self.declare_parameter('model_path', '/home/nvidia/rby1-sdk/models/master_arm/model.urdf')
        self.declare_parameter('publish_velocity', False)

        hz = float(self.get_parameter('loop_hz').value)
        self.dt = 1.0 / hz

        # pubs
        self.pub_states = self.create_publisher(Float32MultiArray, '/control/action', 10)

        # init master arm
        rby.upc.initialize_device(rby.upc.MasterArmDeviceName)
        self.master_arm = rby.upc.MasterArm(rby.upc.MasterArmDeviceName)
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
        self.init_cnt = 300
        self.loop_cnt = 0

        self.init_traj_R = Trajectory(0, self.init_cnt * self.dt)
        self.init_traj_L = Trajectory(0, self.init_cnt * self.dt)
        
        self.prev_q = None 
        self.right_q = READY_POS_R.copy()
        self.left_q = READY_POS_L.copy()

        self.timer = self.create_timer(self.dt, self.loop)
        # 마스터암 내부 제어 루프 실행 (원본은 콜백 기반이었으나, 여기선 polling)
        # 필요 시 self.master_arm.start_control(...) 로 내부 안정화 가능
    

    def master_arm_start_input_callback(self, state: rby.upc.MasterArm.State):
        self.init_traj_R.get_coeff_qpos(state.q_joint[:7], np.zeros_like(READY_POS_R), READY_POS_R)
        self.init_traj_L.get_coeff_qpos(state.q_joint[7:14], np.zeros_like(READY_POS_L), READY_POS_L)

        q_R, qv_R, qa_R = self.init_traj_R.calculate_pva_qpos(self.loop_cnt * self.dt)
        ma_input = rby.upc.MasterArm.ControlInput()
        ma_input.target_operating_mode[0:7].fill(
            rby.DynamixelBus.CurrentBasedPositionControlMode
        )
        ma_input.target_position[0:7] = q_R
        ma_input.target_torque[0:7] = self.ma_torque_limit[0:7] * 4.0

        q_L, qv_L, qa_L = self.init_traj_L.calculate_pva_qpos(self.loop_cnt * self.dt)
        ma_input.target_operating_mode[7:14].fill(
            rby.DynamixelBus.CurrentBasedPositionControlMode
        )
        ma_input.target_position[7:14] = q_L
        ma_input.target_torque[7:14] = self.ma_torque_limit[7:14] * 4.0
        return ma_input
    
    def master_arm_loop_input_callback(self, state: rby.upc.MasterArm.State):
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
            self.right_q = state.q_joint[0:7]
        else:
            ma_input.target_operating_mode[0:7].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[0:7] = self.ma_torque_limit[0:7]
            ma_input.target_position[0:7] = self.right_q

        if state.button_left.button == 1:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[7:14] = torque[7:14] * 0.6
            self.left_q = state.q_joint[7:14]
        else:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[7:14] = self.ma_torque_limit[7:14]
            ma_input.target_position[7:14] = self.left_q
        
        # q = robot_q.copy() # rby1의 현재 상태를 받아야됨.
        # q[model.right_arm_idx] = right_q
        # q[model.left_arm_idx] = left_q
        # dyn_state.set_q(q)
        # dyn_model.compute_forward_kinematics(dyn_state)
        # is_collision = (
        #     dyn_model.detect_collisions_or_nearest_links(dyn_state, 1)[0].distance
        #     < 0.02
        # )
        
        # 버튼/트리거 0~1 정규화
        right_joint = self.right_q.astype(np.float32)
        left_joint = self.left_q.astype(np.float32)
        right_btn = float(state.button_right.button)
        left_btn  = float(state.button_left.button)
        right_trg = float(state.button_right.trigger) / 1000.0
        left_trg  = float(state.button_left.trigger) / 1000.0

        right_actions = [right_btn] + right_joint.tolist() + [right_trg]
        left_actions  = [left_btn] + left_joint.tolist() + [left_trg]

        master_actions = Float32MultiArray()
        # master_actions.data = right_actions
        master_actions.data = right_actions + left_actions
        self.pub_states.publish(master_actions)

        return ma_input

    def master_arm_control_callback(self, state: rby.upc.MasterArm.State):
        if self.loop_cnt < self.init_cnt:
            ma_input = self.master_arm_start_input_callback(state)
        else:
            ma_input = self.master_arm_loop_input_callback(state)
        self.loop_cnt += 1
        
        return ma_input
        
    def loop(self):
        self.master_arm.start_control(self.master_arm_control_callback)


def main():
    rclpy.init()
    node = MasterArmBridge()
    rclpy.spin(node)
    node.master_arm.stop_control()
    node.destroy_node()
    rclpy.shutdown()
    exit(1)

if __name__ == '__main__':
    main()