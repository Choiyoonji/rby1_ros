import numpy as np
import rclpy
from rclpy.node import Node
from MetaQuest_HandTracking.XRHandReceiver import XRHandReceiver
from MetaQuest_HandTracking.StereoStream.StereoStreamer import UdpImageSender

class MetaNode(Node):
    def __init__(self):
        super().__init__('meta_node')
        
        
        self.init_offset = {
            'head': np.array([]),
            'right': np.array([]),
            'left': np.array([])
        }

        ip = "192.168.0.99"
        # ip = "192.168.0.146"
        self.receiver = XRHandReceiver(server_ip=ip)
        self.receiver.connect()
        self.sender = UdpImageSender(ip=ip, port=9003,
                                      width=1280, height=480,
                                      max_payload=1024*1024, jpeg_quality=50)
        self.sender.open()
        self.sender.connect()

    def get_data(self):
        parsed = self.receiver.parse(self.receiver.get())

        pos_H, rotmat_H = None, None
        pos_L, rotmat_L = None, None
        pos_R, rotmat_R = None, None
        hand_L, hand_R = None, None

        if parsed:
            pos_H = parsed["head_robot"]["pos"]
            rotmat_H = parsed["head_robot"]["rotmat"]

            hand_L = parsed['left_raw'] # shape (182,)
            hand_R = parsed['right_raw']

            pos_L = parsed["left_robot"]["pos"]
            rotmat_L = parsed["left_robot"]["rotmat"]

            pos_R = parsed["right_robot"]["pos"]
            rotmat_R = parsed["right_robot"]["rotmat"]

        return {
            "head": {"pos": pos_H, "rotmat": rotmat_H},
            "left": {"pos": pos_L, "rotmat": rotmat_L, "hand": hand_L},
            "right": {"pos": pos_R, "rotmat": rotmat_R, "hand": hand_R}
        }
    
    def set_init_offset(self):
        pass

    def meta_status_callback(self, msg):
        self.get_logger().info(f'Received meta status data: {msg.data}')