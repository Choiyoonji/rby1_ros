#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import pickle
import struct
import time
import collections

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rby1_interfaces.msg import State as RBY1State
from qos_profiles import qos_cmd, qos_state_latest

# === TCP 통신 유틸 (inference.py와 동일 프로토콜) ===
def send_packet(sock, data_obj):
    """Length-prefixed pickle send/recv."""
    data_bytes = pickle.dumps(data_obj)
    sock.sendall(struct.pack('>I', len(data_bytes)) + data_bytes)

    data_len_bytes = sock.recv(4)
    if not data_len_bytes:
        raise ConnectionAbortedError("Server closed connection while waiting for message length.")
    msg_len = struct.unpack('>I', data_len_bytes)[0]

    response_data = b""
    while len(response_data) < msg_len:
        packet = sock.recv(msg_len - len(response_data))
        if not packet:
            raise ConnectionAbortedError("Server closed connection while receiving message.")
        response_data += packet
    return pickle.loads(response_data)


class RBY1InferenceNode(Node):
    def __init__(self):
        super().__init__("rby1_inference_node")

        # ----- 파라미터 (IP/PORT/TASK) -----
        self.declare_parameter("server_ip", "127.0.0.1")
        self.declare_parameter("server_port", 12345)
        self.declare_parameter("task", "default task")  # 필요하면 바꾸기

        self.server_ip = self.get_parameter("server_ip").get_parameter_value().string_value
        self.server_port = self.get_parameter("server_port").get_parameter_value().integer_value
        self.task = self.get_parameter("task").get_parameter_value().string_value

        # ----- ROS I/O -----
        self.state_sub = self.create_subscription(
            RBY1State,
            "/rby1/state",
            self.state_callback,
            qos_state_latest,
        )

        self.action_pub = self.create_publisher(
            Float32MultiArray,
            "/control/action",
            qos_cmd,
        )

        # ----- 내부 상태 -----
        self.latest_state_msg = None
        self.sock = None
        self.connected = False

        # action plan (여러 step을 한번에 받아서 쓰고 싶으면 사용)
        self.action_plan = collections.deque()

        # 주기적으로 inference 돌리는 타이머 (예: 10 Hz)
        self.timer = self.create_timer(1/30.0, self.inference_step)

        self.get_logger().info(
            f"[init] RBY1InferenceNode started. "
            f"Server: {self.server_ip}:{self.server_port}, task='{self.task}'"
        )

    # ----- TCP 연결 -----
    def connect(self):
        if self.connected:
            return
        self.get_logger().info(f"Connecting to policy server {self.server_ip}:{self.server_port}...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.server_ip, self.server_port))
            sock.settimeout(2.0)  # recv timeout (초)
            self.sock = sock
            self.connected = True
            self.get_logger().info("Connected to policy server.")
        except Exception as e:
            self.connected = False
            self.sock = None
            self.get_logger().error(f"Failed to connect to policy server: {e}")

    def close_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self.connected = False

    # ----- /rby1/state 콜백 -----
    def state_callback(self, msg: RBY1State):
        self.latest_state_msg = msg

    # ----- state -> policy 입력 벡터로 변환 -----
    def build_state_vector(self, msg: RBY1State):
        """
        PiZero 학습 때 사용한 관측 형태에 맞게 수정해야 하는 부분.

        여기서는 예제로:
        - joint_positions (N개)
        - placeholder 0.0 하나
        - right_gripper_pos
        로 구성 (inference.py의 Kinova 예제와 같은 형태) 로 맞춰 둠.
        """
        jp = np.array(msg.joint_positions.data, dtype=np.float32)  # 관절 값
        right_gripper = float(msg.right_gripper_pos)

        # 필요하면 EE pose, FT, COM 등도 이어붙여서 obs를 키워도 됨.
        state_vec = jp.tolist() + [0.0] + [right_gripper]
        return state_vec

    # ----- 메인 inference loop -----
    def inference_step(self):
        # state가 아직 없음
        if self.latest_state_msg is None:
            return

        # 서버 연결 시도
        if not self.connected:
            self.connect()
            if not self.connected:
                # 아직 연결 안 됐으면 action 퍼블리시 안 함
                return

        # action_plan 비어있으면 새로 요청
        if not self.action_plan:
            state_vec = self.build_state_vector(self.latest_state_msg)

            # 이미지가 필요 없는 policy라면 image/wrist_image/right_wrist_image는
            # 서버 쪽에서 무시하게 구현해두면 됨.
            dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)

            payload = {
                "image": dummy_img,
                "wrist_image": dummy_img,
                "right_wrist_image": dummy_img,
                "state": state_vec,
                "task": self.task,
            }

            try:
                action_ = send_packet(self.sock, payload)
            except (ConnectionAbortedError, ConnectionResetError, socket.timeout) as e:
                self.get_logger().error(f"Server connection error: {e}. Will reconnect.")
                self.close_socket()
                return
            except Exception as e:
                self.get_logger().error(f"Error during send_packet: {e}")
                return

            # 서버가 여러 step을 리스트로 줄 수도 있고, 한 step만 줄 수도 있음
            if isinstance(action_, (list, tuple)):
                if len(action_) > 0 and isinstance(action_[0], (list, tuple, np.ndarray)):
                    # [step0, step1, ...] 형태라고 가정
                    self.action_plan.extend(action_)
                else:
                    # 단일 step 벡터라고 가정
                    self.action_plan.append(action_)
            else:
                self.get_logger().warn(f"Unexpected action type from server: {type(action_)}")
                return

        # 여기까지 왔으면 action_plan에 최소 1개 있음
        if not self.action_plan:
            return

        raw_action = self.action_plan.popleft()

        # numpy array로 정리
        try:
            action = np.array(raw_action, dtype=np.float32).ravel()
