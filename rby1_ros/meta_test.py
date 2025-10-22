import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from MetaQuest_HandTracking.XRHandReceiver import XRHandReceiver
# from MetaQuest_HandTracking.StereoStream.StereoStreamer import UdpImageSender
from utils import *

T_conv = np.array([
    [0, -1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

if __name__ == "__main__":
    receiver = XRHandReceiver(server_ip="192.168.0.106")
    receiver.connect()
    # sender = UdpImageSender(ip="192.168.0.99", port=9003,
    #                                         width=1280, height=480,
    #                                         max_payload=1024*1024, jpeg_quality=50)
    # sender.open()
    # sender.connect()
    T, pos, quat, deg = get_ready_pos()
    while True:
        data = receiver.get()
        parsed = receiver.parse(data)
        if parsed:
            # print("Right Hand Position:", parsed["right_robot"]["pos"])
            # print("Ready Right Position:", pos["right"])
            fixed_rotmat_R = np.array(parsed["right_robot"]["rotmat"]) @ R.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix() # @ R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
            print("Right Rot mat:", fixed_rotmat_R)
            print("Right Hand Rot deg:", R.from_matrix(fixed_rotmat_R).as_euler('xyz', degrees=True))
            print("Right Hand Rot quat:", R.from_matrix(fixed_rotmat_R).as_quat())
            print("Ready Right Rot mat:", T["right"][:3,:3])
            print("Ready Right Rot deg:", deg["right"])
            print("Ready Right Rot quat:", quat["right"])
            time.sleep(0.5)