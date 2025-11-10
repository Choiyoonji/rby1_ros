#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from platform import node
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from rby1_ros.qos_profiles import qos_ctrl_latched

from MetaQuest_HandTracking.StereoStream.StereoStreamer import UdpImageSender
from cam_utils.shm_util import NamedSharedNDArray

import time
import cv2

SHM_NAME_LEFT = "zed_left"
SHM_NAME_RIGHT = "zed_right"

class ZedImageSender(Node):
    def __init__(self):
        super().__init__('zed_image_sender')
        ip = "192.168.0.106"
        self.sender = UdpImageSender(ip=ip, port=9003,
                                     width=1280, height=480,
                                     max_payload=1024*1024, jpeg_quality=50)
        self.sender.open()
        self.sender.connect()

        self.shmL = self.shmR = None
        t0 = time.time()

        while time.time() - t0 < 2.0:
            try:
                self.shmL = NamedSharedNDArray.open(SHM_NAME_LEFT)
                self.shmR = NamedSharedNDArray.open(SHM_NAME_RIGHT)
                break
            except FileNotFoundError:
                time.sleep(0.05)

        if self.shmL is None or self.shmR is None:
            print("Shared memory segments not found. Exiting.")
            return

        self.sub_record = self.create_subscription(Bool, "/record", self._on_record, qos_ctrl_latched)

        self.recording = False

        self.send_timer = self.create_timer(0.03, self.run)  # 약 30 FPS

    def _on_record(self, msg: Bool):
        self.recording = msg.data
        if self.recording:
            print("ZED Image Sender: Recording started.")
        else:
            print("ZED Image Sender: Recording stopped.")
            self.sender.close()
    

    def run(self):
        if self.shmL is None or self.shmR is None:
            return

        arrL = self.shmL.as_array()
        arrR = self.shmR.as_array()

        try:
            if self.recording:
                left_frame = arrL.copy()
                right_frame = arrR.copy()
                combined_frame = cv2.hconcat([left_frame, right_frame])
                combined_frame = cv2.resize(combined_frame, (1280, 480))
                self.sender.send_image(combined_frame)

        except KeyboardInterrupt:
            print("Interrupted by user. Exiting.")
        finally:
            for s in (self.shmL, self.shmR):
                try:
                    if s is not None:
                        s.close()
                except Exception:
                    pass
            self.sender.close()

def main():
    rclpy.init()
    node = ZedImageSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sender.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()