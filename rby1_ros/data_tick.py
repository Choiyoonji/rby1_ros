#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt64, Bool
import time

class TickPublisher(Node):
    def __init__(self):
        super().__init__('tick_publisher')
        self.tick_pub = self.create_publisher(UInt64, '/tick', 10)
        self.record_sub = self.create_subscription(
            Bool, '/record', self.record_callback, 10
        )
        self.recording = False
        self.tick_count = 0

        # 1 kHz = 0.001 s
        self.timer = self.create_timer(0.001, self.timer_callback)
        self.start_time_ns = None

    def record_callback(self, msg: Bool):
        """토글 신호 수신 콜백"""
        if msg.data and not self.recording:
            self.start_recording()
        elif not msg.data and self.recording:
            self.stop_recording()

    def start_recording(self):
        self.get_logger().info('Recording START')
        self.recording = True
        self.tick_count = 0
        self.start_time_ns = time.monotonic_ns()

    def stop_recording(self):
        self.get_logger().info(f'Recording STOP (total ticks = {self.tick_count})')
        self.recording = False
        self.start_time_ns = None

    def timer_callback(self):
        """1 kHz 타이머 콜백"""
        if not self.recording:
            return

        msg = UInt64()
        msg.data = self.tick_count
        self.tick_pub.publish(msg)
        self.tick_count += 1

        # 디버그용 로그 (1 초마다 한 번씩만 표시)
        if self.tick_count % 1000 == 0:
            elapsed = (time.monotonic_ns() - self.start_time_ns) * 1e-9
            self.get_logger().info(f'tick={self.tick_count}, elapsed={elapsed:.3f}s')

def main(args=None):
    rclpy.init(args=args)
    node = TickPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt → exit')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
