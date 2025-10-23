#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from scipy.spatial.transform import Rotation as R
from meta_node import MetaNode

# =========================
# 여기에 네 코드의 T를 넣어 주세요
# 4x4 동차변환행렬: R|t ; 0 0 0 1
# 예시값(단위: m)
T_RIGHT_HAND = np.array([
    [-0.48622539, -0.38410535, -0.78488722, 0.0],
    [-0.24183169,  0.92227812, -0.30153026, 0.0],
    [ 0.8397037,   0.04319894, -0.54132388, 0.0],
    [ 0.0,         0.0,         0.0,         1.0],
], dtype=float)

T_READY_RIGHT = np.array([
 [ 0.18348889,  0.05939117, -0.98122603,  0.28849388],
 [ 0.3213938,   0.93969262,  0.11697778, -0.22      ],
 [ 0.9289983,  -0.33682409,  0.1533352,   0.90289964],
 [ 0.,          0.,          0.,          1.        ]
], dtype=float)

PARENT_FRAME = "map"          # 부모 프레임 이름
RIGHT_HAND_FRAME = "right_hand_frame"
READY_RIGHT_FRAME = "ready_right_frame"

# 축 길이와 두께
AXIS_LEN = 0.15
AXIS_RADIUS = 0.01
# =========================

def mat_to_quat(R: np.ndarray):
    """3x3 회전행렬 -> (x,y,z,w) 쿼터니언. 수치안정성 고려."""
    # 참고: 마커와 TF는 (x,y,z,w) 순서
    q = np.empty(4, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        q[3] = 0.25 * S
        q[0] = (R[2,1] - R[1,2]) / S
        q[1] = (R[0,2] - R[2,0]) / S
        q[2] = (R[1,0] - R[0,1]) / S
    else:
        # 대각 원소 중 최대값 기준 분기
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            q[3] = (R[2,1] - R[1,2]) / S
            q[0] = 0.25 * S
            q[1] = (R[0,1] + R[1,0]) / S
            q[2] = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            q[3] = (R[0,2] - R[2,0]) / S
            q[0] = (R[0,1] + R[1,0]) / S
            q[1] = 0.25 * S
            q[2] = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            q[3] = (R[1,0] - R[0,1]) / S
            q[0] = (R[0,2] + R[2,0]) / S
            q[1] = (R[1,2] + R[2,1]) / S
            q[2] = 0.25 * S

    # 정규화
    q = q / np.linalg.norm(q)
    return q  # x,y,z,w

def T_to_translation_quat(T: np.ndarray):
    assert T.shape == (4,4)
    R = T[:3,:3]
    t = T[:3, 3]
    qx,qy,qz,qw = mat_to_quat(R)
    return t[0], t[1], t[2], qx, qy, qz, qw

def make_axis_arrow(ns: str, mid: int, frame_id: str,
                    origin_T: np.ndarray,
                    axis_vec_local: np.ndarray,
                    color_rgb: tuple,
                    length=AXIS_LEN, radius=AXIS_RADIUS):
    """
    origin_T: 축의 원점이 되는 T(4x4)
    axis_vec_local: 로컬축 단위벡터 (예: X=[1,0,0])
    화살표는 시작점 origin, 끝점 origin + R*axis_vec_local*length
    """
    # 시작점과 끝점 계산
    R = origin_T[:3,:3]
    p = origin_T[:3, 3]
    end = p + R @ (axis_vec_local * length)

    m = Marker()
    m.header.frame_id = PARENT_FRAME
    m.ns = ns
    m.id = mid
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = length          # shaft length
    m.scale.y = radius * 2.0    # shaft diameter
    m.scale.z = radius * 2.0    # head diameter (RViz2는 ARROW에서 x,y,z 의미가 shaft/head로 매핑)

    # ARROW의 경우 points를 주면 시작-끝으로 렌더링됨
    from geometry_msgs.msg import Point
    m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))]

    m.color.r = float(color_rgb[0])
    m.color.g = float(color_rgb[1])
    m.color.b = float(color_rgb[2])
    m.color.a = 1.0
    return m

def make_frame_text(ns: str, mid: int, text: str, T: np.ndarray):
    from geometry_msgs.msg import Point
    m = Marker()
    m.header.frame_id = PARENT_FRAME
    m.ns = ns
    m.id = mid
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.scale.z = 0.05  # 글자 높이
    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
    # 약간 위로 띄워 보이게
    p = T[:3, 3] + np.array([0.0, 0.0, 0.03])
    m.pose.position.x = float(p[0])
    m.pose.position.y = float(p[1])
    m.pose.position.z = float(p[2])
    m.pose.orientation.w = 1.0
    m.text = text
    return m

class AxesAndTFPublisher(Node):
    def __init__(self):
        super().__init__("axes_and_tf_publisher")
        # TF 브로드캐스터
        self.tf_broadcaster = TransformBroadcaster(self)
        # 축 마커 퍼블리셔
        self.marker_pub = self.create_publisher(MarkerArray, "/axes_markers", 10)
        # 타이머
        self.timer = self.create_timer(0.1, self.on_timer)  # 10Hz

        self.meta = MetaNode()

    def publish_tf(self, child_frame: str, T: np.ndarray):
        tx, ty, tz, qx, qy, qz, qw = T_to_translation_quat(T)
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = PARENT_FRAME
        msg.child_frame_id = child_frame
        msg.transform.translation.x = float(tx)
        msg.transform.translation.y = float(ty)
        msg.transform.translation.z = float(tz)
        msg.transform.rotation.x = float(qx)
        msg.transform.rotation.y = float(qy)
        msg.transform.rotation.z = float(qz)
        msg.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(msg)

    def on_timer(self):
        data = self.meta.get_data()
        if data is None:
            return
        right_hand_data = data['left']
        right_hand_pos = right_hand_data['pos']
        right_hand_rotmat = np.asarray(right_hand_data['rotmat'])
        if right_hand_pos is None or right_hand_rotmat is None:
            return
        T_RIGHT_HAND = np.eye(4, dtype=float)
        T_RIGHT_HAND[:3, :3] = right_hand_rotmat @ R.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix()
        T_RIGHT_HAND[:3, 3] = right_hand_pos
        # 1) TF 브로드캐스트
        self.publish_tf(RIGHT_HAND_FRAME, T_RIGHT_HAND)
        self.publish_tf(READY_RIGHT_FRAME, T_READY_RIGHT)

        # 2) 축 마커 생성
        arr = MarkerArray()
        # right hand 축: 빨 x, 초 y, 파 z
        arr.markers.append(make_axis_arrow("right_hand", 0, PARENT_FRAME, T_RIGHT_HAND, np.array([1,0,0]), (1,0,0)))
        arr.markers.append(make_axis_arrow("right_hand", 1, PARENT_FRAME, T_RIGHT_HAND, np.array([0,1,0]), (0,1,0)))
        arr.markers.append(make_axis_arrow("right_hand", 2, PARENT_FRAME, T_RIGHT_HAND, np.array([0,0,1]), (0,0,1)))
        arr.markers.append(make_frame_text("right_hand", 3, RIGHT_HAND_FRAME, T_RIGHT_HAND))

        # ready right 축
        arr.markers.append(make_axis_arrow("ready_right", 10, PARENT_FRAME, T_READY_RIGHT, np.array([1,0,0]), (1,0,0)))
        arr.markers.append(make_axis_arrow("ready_right", 11, PARENT_FRAME, T_READY_RIGHT, np.array([0,1,0]), (0,1,0)))
        arr.markers.append(make_axis_arrow("ready_right", 12, PARENT_FRAME, T_READY_RIGHT, np.array([0,0,1]), (0,0,1)))
        arr.markers.append(make_frame_text("ready_right", 13, READY_RIGHT_FRAME, T_READY_RIGHT))

        # 공통 header
        for m in arr.markers:
            m.header.stamp = self.get_clock().now().to_msg()

        self.marker_pub.publish(arr)

def main():
    rclpy.init()
    node = AxesAndTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
