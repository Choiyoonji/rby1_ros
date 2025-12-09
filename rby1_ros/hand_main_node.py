import os
import sys
import time
import numpy as np
import h5py
import rclpy
from rclpy.node import Node

from bind import inspire_np
from std_msgs.msg import String, Bool, UInt64
from rby1_interfaces.msg import CommandHand
from rby1_ros.qos_profiles import qos_cmd, qos_ctrl_latched, qos_tick

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
        
        # 현재 목표 각도 (Action에 사용)
        self.current_target_l = list(self.init_angle)
        self.current_target_r = list(self.init_angle)
        
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
        
        # Recording 관련 파라미터 및 변수
        self.recording = False
        self.dataset_path = None
        self.h5_path = None
        
        # 버퍼 초기화
        self._init_buffers()

        self.hand_cmd_sub = self.create_subscription(
            CommandHand, '/control/action/hand', self.hand_command_callback, qos_cmd
        )

        self.sub_record = self.create_subscription(Bool, "/record", self._on_record, qos_ctrl_latched)
        self.sub_path = self.create_subscription(String, "/dataset_path", self._on_data_path, qos_ctrl_latched)
        self.sub_tick = self.create_subscription(UInt64, "/tick", self._on_tick, qos_tick)

        self.cmd_type = ["left", "right"]
        self.cmd_mode = [1, 0]

        # Main Loop Timer (500Hz)
        self.step_hz = 500
        self.step_timer = self.create_timer(1/self.step_hz, self.loop_step)
        
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
            
    def _init_buffers(self):
        self.buf_now_mono_ns = []
        self.buf_tick = []
        
        self.buf_l_angle = []
        self.buf_l_force = [] 
        self.buf_l_norm_force = []
        self.buf_l_tang_force = []
        self.buf_l_temp = []
        self.buf_l_cur = []
        
        self.buf_r_angle = []
        self.buf_r_force = []
        self.buf_r_norm_force = []
        self.buf_r_tang_force = []
        self.buf_r_temp = []
        self.buf_r_cur = []

    # TODO: Meta 연동
    def hand_command_callback(self, msg: CommandHand):
        if msg.hand not in self.cmd_type:
            return
        
        target_angle = self.open_angle if msg.opening == 1 else self.close_angle
        action_str = "OPEN" if msg.opening == 1 else "CLOSE"
        
        if msg.hand == "left":
            self.current_target_l = target_angle
        elif msg.hand == "right":
            self.current_target_r = target_angle
            
        self.get_logger().info(f"Command received: {msg.hand} -> {action_str}")

    def _on_record(self, msg: Bool):
        if msg.data and not self.recording:
            if self.dataset_path:
                self._start_recording()
            else:
                self.recording = True
        elif (not msg.data) and self.recording:
            self._stop_and_dump()

    def _on_data_path(self, msg: String):
        self.dataset_path = msg.data
        self.h5_path = os.path.join(self.dataset_path, "inspire_hand_data.h5")
        if self.recording:
            self._start_recording()

    def _start_recording(self):
        self.recording = True
        self._init_buffers()
        self.get_logger().info("[Record] START")

    def _on_tick(self, msg: UInt64):
        """
        Tick이 오면 loop_step에서 최신 업데이트한 self.act_... 값을 저장
        """
        if not self.recording or self.dataset_path is None:
            return

        # Time & Tick
        self.buf_now_mono_ns.append(time.monotonic_ns())
        self.buf_tick.append(msg.data)
        
        # Sensor Data (Control Loop에서 갱신된 최신값 복사)
        self.buf_l_angle.append(list(self.act_angle_l))
        self.buf_l_force.append(list(self.act_force_l))
        self.buf_l_norm_force.append(list(self.act_norm_force_l))
        self.buf_l_tang_force.append(list(self.act_tang_force_l))
        self.buf_l_temp.append(list(self.act_temp_l))
        self.buf_l_cur.append(list(self.act_cur_l))

        self.buf_r_angle.append(list(self.act_angle_r))
        self.buf_r_force.append(list(self.act_force_r))
        self.buf_r_norm_force.append(list(self.act_norm_force_r))
        self.buf_r_tang_force.append(list(self.act_tang_force_r))
        self.buf_r_temp.append(list(self.act_temp_r))
        self.buf_r_cur.append(list(self.act_cur_r))

    def _stop_and_dump(self):
        self.recording = False
        self.get_logger().info(f"[Record] STOP. Saving {len(self.buf_tick)} samples...")
        self._write_h5()

    def _write_h5(self):
        if not self.h5_path: return
        
        try:
            os.makedirs(os.path.dirname(self.h5_path), exist_ok=True)
            with h5py.File(self.h5_path, "w") as f:
                f.attrs["created_wall_time_ns"] = int(time.time_ns())
                
                def create_dset(name, data):
                    if not data: 
                        return
                    f.create_dataset(name, data=np.array(data), compression="gzip")
                
                create_dset("now_mono_ns", self.buf_now_mono_ns)
                create_dset("tick", self.buf_tick)
                
                # Left Hand Data
                create_dset("left/angle", self.buf_l_angle)
                create_dset("left/force", self.buf_l_force)
                create_dset("left/norm_force", self.buf_l_norm_force) 
                create_dset("left/tang_force", self.buf_l_tang_force) 
                create_dset("left/temp", self.buf_l_temp)             
                create_dset("left/current", self.buf_l_cur)           

                # Right Hand Data
                create_dset("right/angle", self.buf_r_angle)
                create_dset("right/force", self.buf_r_force)
                create_dset("right/norm_force", self.buf_r_norm_force)
                create_dset("right/tang_force", self.buf_r_tang_force)
                create_dset("right/temp", self.buf_r_temp)            
                create_dset("right/current", self.buf_r_cur)          

            self.get_logger().info(f"Saved to {self.h5_path}")
            self._init_buffers()
            
        except Exception as e:
            self.get_logger().error(f"Failed to write HDF5: {e}")
            
    def loop_step(self):
        """
        1. Write: 목표 각도 전송
        2. Read: 모든 센서값 읽어서 변수에 업데이트
        """
        try:
            # --- Left Hand ---
            self.controller_l.cyclic_PDO_init()
            self.controller_l.write_pdo_enable(1)
            self.controller_l.write_pdo_angle_set6(self.current_target_l)
            self.controller_l.write_pdo_force_set6(self.target_force)
            self.controller_l.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.act_angle_l = self.controller_l.read_pdo_angle_get6()
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
            self.controller_r.write_pdo_angle_set6(self.current_target_r)
            self.controller_r.write_pdo_force_set6(self.target_force)
            self.controller_r.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.act_angle_r = self.controller_r.read_pdo_angle_get6()
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