import rclpy
from rclpy.node import Node
from bind import inspire_np, my_loop
from std_msgs.msg import String, Float32
from rby1_interfaces.msg import CommandHand
from rby1_ros.qos_profiles import qos_cmd
import time

class HandNodeCommand(Node):
    def __init__(self):
        super().__init__('hand_node_command')
        # master 설정
        self.controller_l = inspire_np.EtherCATController(0)
        self.controller_r = inspire_np.EtherCATController(1)
        
        # mode 설정 (0: position, 1: force, 2: impedance)
        self.controller_l.initialize(2)
        self.controller_r.initialize(2)
        
        # force 및 speed 값 설정
        self.target_force = [1000] * 6 # 0 ~ 12000
        self.target_speed = [2000] * 6 # 0 ~ 6000
        
        # 초기 각도 설정
        self.init_angle = [1721, 1721, 1721, 1721, 1350, 1700]
        
        # open/close 각도 설정 (joint range.txt 참고)
        self.open_angle = [1721, 1721, 1721, 1721, 1350, 600]
        self.close_angle = [1000, 1000, 1000, 1000, 1200, 600]
        
        self.initialize_hands()
        
        self.hand_cmd_sub = self.create_subscription(
            CommandHand,
            '/control/action/hand',
            self.hand_command_callback,
            qos_cmd
        )
        
        self.cmd_type = ["left", "right"]
        self.cmd_mode = [1, 0]  # 1: open, 0: close
        
    def initialize_hands(self):
        for i in range(100):
            self.controller_l.cyclic_PDO_init()
            self.controller_r.cyclic_PDO_init()
            
            self.controller_l.write_pdo_enable(1)
            self.controller_r.write_pdo_enable(1)
        
            self.controller_l.write_pdo_angle_set6(self.init_angle)
            self.controller_l.write_pdo_force_set6(self.target_force) 
            self.controller_l.write_pdo_speed_set6(self.target_speed)
            self.controller_l.cyclic_PDO_exit()

            self.controller_r.write_pdo_angle_set6(self.init_angle)
            self.controller_r.write_pdo_force_set6(self.target_force) 
            self.controller_r.write_pdo_speed_set6(self.target_speed)
            self.controller_r.cyclic_PDO_exit()
    
    def open_hand(self, hand: str):
        if hand == "left":
            self.controller_l.cyclic_PDO_init()
            self.controller_l.write_pdo_enable(1)
            self.controller_l.write_pdo_angle_set6(self.open_angle)
            self.controller_l.write_pdo_force_set6(self.target_force) 
            self.controller_l.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.controller_l.cyclic_PDO_exit()
            self.get_logger().info('Left hand opened')
        elif hand == "right":
            self.controller_r.cyclic_PDO_init()
            self.controller_r.write_pdo_enable(1)
            self.controller_r.write_pdo_angle_set6(self.open_angle)
            self.controller_r.write_pdo_force_set6(self.target_force) 
            self.controller_r.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.controller_r.cyclic_PDO_exit()
            self.get_logger().info('Right hand opened')
    
    def close_hand(self, hand: str):
        if hand == "left":
            self.controller_l.cyclic_PDO_init()
            self.controller_l.write_pdo_enable(1)
            self.controller_l.write_pdo_angle_set6(self.close_angle)
            self.controller_l.write_pdo_force_set6(self.target_force) 
            self.controller_l.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.controller_l.cyclic_PDO_exit()
            self.get_logger().info('Left hand closed')
        elif hand == "right":
            self.controller_r.cyclic_PDO_init()
            self.controller_r.write_pdo_enable(1)
            self.controller_r.write_pdo_angle_set6(self.close_angle)
            self.controller_r.write_pdo_force_set6(self.target_force) 
            self.controller_r.write_pdo_speed_set6(self.target_speed)
            # time.sleep(0.0001)
            self.controller_r.cyclic_PDO_exit()
            self.get_logger().info('Right hand closed')
    
    def hand_command_callback(self, msg: CommandHand):
        self.get_logger().info('Received hand command')
        if msg.hand not in self.cmd_type:
            self.get_logger().error(f'Unknown hand type: {msg.hand}, it must be "left" or "right"')
            return
        if msg.opening not in self.cmd_mode:
            self.get_logger().error(f'Unknown hand cmd: {msg.opening}, it must be "1" or "0"')
            return
        
        if msg.opening == 1:
            self.open_hand(msg.hand)
        elif msg.opening == 0:
            self.close_hand(msg.hand)