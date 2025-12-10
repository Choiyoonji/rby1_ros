import os
import sys
import time
import numpy as np
import h5py
import rclpy
from rclpy.node import Node

from rby1_interfaces.msg import CommandHand, StateHand
from std_msgs.msg import Float32MultiArray
from rby1_ros.qos_profiles import qos_cmd, qos_state_latest
from bind import inspire_np
from inspire.inspire_kinematics import RH56F1_Kinematic, spring_damper_ik, mapping_meta2inspire_l, mapping_meta2inspire_r, FINGER_DIST_INSPIRE, FINGET_DIST_META
from inspire.functions_np import *

class HandMainNode(Node):
    def __init__(self):
        super().__init__('inspire_hand_ctrl_and_record')
        # EtherCAT master 설정
        self.controller_l = inspire_np.EtherCATController(0)
        self.controller_r = inspire_np.EtherCATController(1)
        
        # Mode 설정 (2: impedance)
        self.controller_l.initialize(2) 
        self.controller_r.initialize(2)
        
        # 제어 파라미터
        self.target_force = [1000] * 6
        self.target_speed = [2000] * 6
        
        # 각도 설정 (Init / Open / Close)
        self.init_angle = [1721, 1721, 1721, 1721, 1350, 1700]
        self.open_angle = [1721, 1721, 1721, 1721, 1350, 600]
        self.close_angle = [1000, 1000, 1000, 1000, 1200, 600]
        
        # Kinematics
        self.hand_kin = RH56F1_Kinematic()
        
        # cmd msg 합친 리스트
        self.meta_l = list()
        self.meta_r = list()
        
        # 현재 목표 각도 (Action에 사용)
        self.current_target_l = np.zeros(6)
        self.current_target_r = np.zeros(6)
        
        # 현재 실제 센서값 (Recording에 사용)
        self.act_angle_l = [0.0] * 6
        self.act_force_l = [0.0] * 6
        self.act_temp_l = [0.0] * 6
        self.act_cur_l = [0.0] * 6
        self.act_norm_force_l = [0.0] * 8
        self.act_tang_force_l = [0.0] * 8
        
        self.act_angle_r = [0.0] * 6
        self.act_force_r = [0.0] * 6
        self.act_temp_r = [0.0] * 6
        self.act_cur_r = [0.0] * 6
        self.act_norm_force_r = [0.0] * 8
        self.act_tang_force_r = [0.0] * 8
        
        self.initialize_hands()

        self.hand_cmd_sub = self.create_subscription(
            CommandHand, '/control/hand_command', self.hand_command_callback, qos_cmd
        )
        self.hand_state_pub = self.create_publisher(
            StateHand, '/hand/state', qos_state_latest
        )

        # Main Loop Timer (500Hz)
        self.step_hz = 500
        self.step_timer = self.create_timer(1/self.step_hz, self.loop_step)
        self.state_pub_timer = self.create_timer(1/100.0, self.publish_hand_state) 
        self.get_logger().info("Inspire Hand Control & Record Node Started.")

    def initialize_hands(self):
        for i in range(100):
            self.controller_l.cyclic_PDO_init()
            self.controller_r.cyclic_PDO_init()

            self.controller_l.write_pdo_enable(1)
            self.controller_r.write_pdo_enable(1)

            self.controller_l.write_pdo_angle_set6(self.init_angle)
            self.controller_l.write_pdo_force_set6(self.target_force)
            self.controller_l.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.controller_l.cyclic_PDO_exit()

            self.controller_r.write_pdo_angle_set6(self.init_angle)
            self.controller_r.write_pdo_force_set6(self.target_force)
            self.controller_r.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.controller_r.cyclic_PDO_exit()
    
    def compute_target_angle(self, type: str, meta_data: list, current_angle: np.ndarray) -> list:
        if not type in ['left', 'right']: raise ValueError("type must be 'left' or 'right'")
        self.hand_kin.type = type
        _, _, P_lnk, R_lnk = self.hand_kin.DH_forward(current_angle)
        
        if type == 'right':
            p_EE_meta, _ = mapping_meta2inspire_r(meta_data, FINGET_DIST_META, P_lnk, R_lnk, FINGER_DIST_INSPIRE)
        else:
            p_EE_meta, _ = mapping_meta2inspire_l(meta_data, FINGET_DIST_META, P_lnk, R_lnk, FINGER_DIST_INSPIRE)
        
        q_d = spring_damper_ik(
            q_init=current_angle,
            target_pos=p_EE_meta,
            type=type,
            base_pos=np.array([0,0,0]),
            base_rot=np.eye(3),
            K=50.0,
            D=2.0,
            max_iterations=10,
            step_clip=0.1,
            tol=1e-3
        )
        return q_d

    def hand_command_callback(self, msg: CommandHand):
        if msg.p_EE_l.data and msg.p_EE_r.data is None:
            return
        self.meta_l.append(np.array(msg.p_EE_l.data))
        self.meta_l.append(np.array(msg.p_lnk_l.data))
        self.meta_l.append(np.array(msg.r_lnk_l.data))
        self.meta_r.append(np.array(msg.p_EE_r.data))
        self.meta_r.append(np.array(msg.p_lnk_r.data))
        self.meta_r.append(np.array(msg.r_lnk_r.data))
        
        self.current_target_l = self.compute_target_angle("left", self.meta_l, self.act_angle_l)
        self.current_target_r = self.compute_target_angle("right", self.meta_r, self.act_angle_r)
            
        self.get_logger().info(f"Command received :left -> {self.current_target_l}, right -> {self.current_target_r}")
    
    def publish_hand_state(self):
        state_msg = StateHand()
        state_msg.timestamp = self.get_clock().now().to_msg()
        
        state_msg.act_angle_l = Float32MultiArray(data=self.act_angle_l.tolist())
        state_msg.act_force_l = Float32MultiArray(data=self.act_force_l)
        state_msg.act_temp_l = Float32MultiArray(data=self.act_temp_l)
        state_msg.act_cur_l = Float32MultiArray(data=self.act_cur_l)
        state_msg.act_norm_force_l = Float32MultiArray(data=self.act_norm_force_l)
        state_msg.act_tang_force_l = Float32MultiArray(data=self.act_tang_force_l)
        
        state_msg.act_angle_r = Float32MultiArray(data=self.act_angle_r.tolist())
        state_msg.act_force_r = Float32MultiArray(data=self.act_force_r)
        state_msg.act_temp_r = Float32MultiArray(data=self.act_temp_r)
        state_msg.act_cur_r = Float32MultiArray(data=self.act_cur_r)
        state_msg.act_norm_force_r = Float32MultiArray(data=self.act_norm_force_r)
        state_msg.act_tang_force_r = Float32MultiArray(data=self.act_tang_force_r)
        
        self.hand_state_pub.publish(state_msg)
    
    def loop_step(self):
        """
        1. Write: 목표 각도 전송
        2. Read: 모든 센서값 읽어서 변수에 업데이트
        """
        try:
            # --- Left Hand ---
            self.controller_l.cyclic_PDO_init()
            self.controller_l.write_pdo_enable(1)
            self.controller_l.write_pdo_convert_angle_set6(self.current_target_l)
            self.controller_l.write_pdo_force_set6(self.target_force)
            self.controller_l.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.act_angle_l = self.controller_l.read_pdo_convert_angle_get6()
            self.act_force_l = self.controller_l.read_pdo_force_get6()
            self.act_temp_l = self.controller_l.read_pdo_temp_get6()
            self.act_cur_l = self.controller_l.read_pdo_cur_get6()
            self.act_norm_force_l = self.controller_l.read_pdo_normal_force_act8()
            self.act_tang_force_l = self.controller_l.read_pdo_tangent_force_act8()      
            # time.sleep(0.0001)
            self.controller_l.cyclic_PDO_exit()

            # --- Right Hand ---
            self.controller_r.cyclic_PDO_init()
            self.controller_r.write_pdo_enable(1)
            self.controller_r.write_pdo_convert_angle_set6(self.current_target_r)
            self.controller_r.write_pdo_force_set6(self.target_force)
            self.controller_r.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.act_angle_r = self.controller_r.read_pdo_convert_angle_get6()
            self.act_force_r = self.controller_r.read_pdo_force_get6()
            self.act_temp_r = self.controller_r.read_pdo_temp_get6()
            self.act_cur_r = self.controller_r.read_pdo_cur_get6()
            self.act_norm_force_r = self.controller_r.read_pdo_normal_force_act8()
            self.act_tang_force_r = self.controller_r.read_pdo_tangent_force_act8()      
            # time.sleep(0.0001)
            self.controller_r.cyclic_PDO_exit()
            
        except Exception as e:
            pass
        
    def destroy_node(self):
        if self.recording:
            self._stop_and_dump()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HandMainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()