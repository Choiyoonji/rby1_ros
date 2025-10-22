import numpy as np
from MetaQuest_HandTracking.XRHandReceiver import XRHandReceiver
from MetaQuest_HandTracking.StereoStream.StereoStreamer import UdpImageSender

if __name__ == "__main__":
    receiver = XRHandReceiver(server_ip="192.168.0.99")
    receiver.connect()
    sender = UdpImageSender(ip="192.168.0.99", port=9003,
                                            width=1280, height=480,
                                            max_payload=1024*1024, jpeg_quality=50)
    sender.open()
    sender.connect()
    while True:
        data = receiver.get()
        parsed = receiver.parse(data)
        if parsed:
            print("Right Hand Position:", parsed["right_robot"]["pos"])