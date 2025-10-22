import numpy as np
import rclpy
from rclpy.node import Node
from rby1_ros.srv import MetaDataReq, MetaInitialReq
from scipy.spatial.transform import Rotation as R
from MetaQuest_HandTracking.XRHandReceiver import XRHandReceiver
from MetaQuest_HandTracking.StereoStream.StereoStreamer import UdpImageSender
from rby1_ros.utils import *
from rby1_ros.meta_status import MetaStatus as MetaState

class MetaNode(Node):
    def __init__(self):
        super().__init__('meta_node')

        self.meta_state = MetaState()

        ip = "192.168.0.106"
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
    
    def is_available_area(self, data):
        # check if the meta data is in available area to initialize offset
        # 1. -0.05 < (Left_x - Right_x), (Left_z - Right_z) < 0.05
        # 2. ready_pos_L-ready_pos_R - 0.02 < Left_y - Right_y < ready_pos_L-ready_pos_R + 0.02
        # 3. -5 < ready_deg - deg_R, L, H < 5 for all axes
        pos_R = data["right"]["pos"]
        pos_L = data["left"]["pos"]
        pos_H = data["head"]["pos"]

        deg_R = R.from_matrix(data["right"]["rotmat"]).as_euler('xyz', degrees=True)
        deg_L = R.from_matrix(data["left"]["rotmat"]).as_euler('xyz', degrees=True)
        deg_H = R.from_matrix(data["head"]["rotmat"]).as_euler('xyz', degrees=True)

        _, ready_pos, ready_deg = get_ready_pos()

        check_1 = (-0.05 < (pos_L[0] - pos_R[0]) < 0.05) and (-0.05 < (pos_L[2] - pos_R[2]) < 0.05)
        check_2 = (ready_pos['left'][1] - ready_pos['right'][1] - 0.02 < (pos_L[1] - pos_R[1]) < 
                    ready_pos['left'][1] - ready_pos['right'][1] + 0.02)
        check_3 = all(-5 < (ready_deg['right'][i] - deg_R[i]) < 5 for i in range(3)) and \
                        all(-5 < (ready_deg['left'][i] - deg_L[i]) < 5 for i in range(3)) and \
                        all(-5 < (ready_deg['torso'][i] - deg_H[i]) < 5 for i in range(3))

        return check_1 and check_2 and check_3, [check_1, check_2, check_3]

    def set_init_offset(self, request, response):
        if request.initialize:
            attempt = 0
            max_attempts = 5
            self.get_logger().info('Starting Meta offset initialization...')
            while True:
                data = self.get_data()
                available, checks = self.is_available_area(data)
                if available:
                    self.init_offset['head'] = np.array(data['head']['pos'])
                    self.init_offset['right'] = np.array(data['right']['pos'])
                    self.init_offset['left'] = np.array(data['left']['pos'])
                    self.get_logger().info('Meta offset initialized successfully.')
                    response.success = True
                    response.check1 = True
                    response.check2 = True
                    response.check3 = True
                    return response
                else:
                    self.get_logger().warning(f'Meta offset initialization failed. Checks: {checks}')
                    if checks[0] == False:
                        self.get_logger().warning('Check 1 failed: Hand X and Z positions are not aligned.')
                    if checks[1] == False:
                        self.get_logger().warning('Check 2 failed: Hand Y positions are not within the required range.')
                    if checks[2] == False:
                        self.get_logger().warning('Check 3 failed: Euler angles are not within the required range.')
                    attempt += 1
                    if attempt >= max_attempts:
                        self.get_logger().error('Max attempts reached. Meta offset initialization failed.')
                        response.success = False
                        response.check1 = checks[0]
                        response.check2 = checks[1]
                        response.check3 = checks[2]
                        return response
                

    def meta_status_callback(self, msg):
        self.get_logger().info(f'Received meta status data: {msg.data}')