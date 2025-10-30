import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rby1_interfaces.srv import MetaDataReq, MetaInitialReq
from scipy.spatial.transform import Rotation as R
from MetaQuest_HandTracking.XRHandReceiver import XRHandReceiver
from MetaQuest_HandTracking.StereoStream.StereoStreamer import UdpImageSender
from rby1_ros.utils import *
from rby1_ros.meta_status import MetaStatus as MetaState


ROT_T = R.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix()
THUMB_TIP_IDX = 5
INDEX_TIP_IDX = 10
MAX_TIP_DISTANCE = 0.12  # 최대 거리 (m)
MIN_TIP_DISTANCE = 0.02  # 최소 거리 (m)

class MetaNode(Node):
    def __init__(self):
        super().__init__('meta_service_node')

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

        self.initialize_service = self.create_service(MetaInitialReq, '/meta/set_offset', self.set_init_offset)
        self.status_service = self.create_service(MetaDataReq, '/meta/get_data', self.get_meta_status)

    def get_data(self):
        parsed = self.receiver.parse(self.receiver.get())

        data = None
        if parsed:
            data = {}

            def _is_zero_pos(p):
                p = np.asarray(p, dtype=float)
                return np.allclose(p, 0.0, atol=1e-8)

            if _is_zero_pos(parsed["head_robot"]["pos"]) or _is_zero_pos(parsed["left_robot"]["pos"]) or _is_zero_pos(parsed["right_robot"]["pos"]):
                return None

            def _orthonormalize(Rm):
                U, _, Vt = np.linalg.svd(Rm)
                R_ortho = U @ Vt
                if np.linalg.det(R_ortho) < 0:  # 반사 방지
                    U[:, -1] *= -1
                    R_ortho = U @ Vt
                return R_ortho

            data["head"] = {"pos": parsed["head_robot"]["pos"], "rotmat": _orthonormalize(parsed["head_robot"]["rotmat"])}
            data["left"] = {"pos": parsed["left_robot"]["pos"], "rotmat": _orthonormalize(parsed["left_robot"]["rotmat"] @ ROT_T), "hand": parsed['left_raw']}
            data["right"] = {"pos": parsed["right_robot"]["pos"], "rotmat": _orthonormalize(parsed["right_robot"]["rotmat"] @ ROT_T), "hand": parsed['right_raw']}

            data["hand"]["left"]["pos"], data["hand"]["left"]["rotmat"] = self.receiver.update_hand(parsed['left_raw'])
            data["hand"]["right"]["pos"], data["hand"]["right"]["rotmat"] = self.receiver.update_hand(parsed['right_raw'])

        return data
    
    # ---------------------------
    # Float32MultiArray → np.ndarray 헬퍼
    # ---------------------------
    def _f32arr_to_np(self, arr_or_msg, expected_len):
        """
        arr_or_msg: std_msgs/Float32MultiArray 또는 시퀀스
        expected_len: 기대 길이(3 또는 4)
        return: np.ndarray(dtype=float)
        """
        if hasattr(arr_or_msg, "data"):
            data = np.asarray(arr_or_msg.data, dtype=float)
        else:
            data = np.asarray(arr_or_msg, dtype=float)
        if data.size != expected_len:
            raise ValueError(f"expected len {expected_len}, got {data.size}")
        return data

    # ---------------------------
    # (helper) 요청에서 ready pose 파싱
    # ---------------------------
    def _parse_ready_pose_from_request(self, request):
        """
        request.left_ready_pos.position/quaternion: Float32MultiArray
        request.right_ready_pos.position/quaternion: Float32MultiArray
        request.head_ready_pos.position/quaternion: Float32MultiArray
        return:
        ready_pos  dict: {'left','right','torso'} 각 (3,)
        ready_quat dict: {'left','right','torso'} 각 (4,) (정규화해서 반환)
        """
        # position (3)
        lp = self._f32arr_to_np(request.left_ready_pos.position, 3)
        rp = self._f32arr_to_np(request.right_ready_pos.position, 3)
        tp = self._f32arr_to_np(request.head_ready_pos.position, 3)  # head→torso로 사용

        # quaternion (quat, 4)
        lq = self._f32arr_to_np(request.left_ready_pos.quaternion, 4)
        rq = self._f32arr_to_np(request.right_ready_pos.quaternion, 4)
        tq = self._f32arr_to_np(request.head_ready_pos.quaternion, 4)

        # 쿼터니언 정규화
        def _norm_quat(q):
            n = np.linalg.norm(q)
            if n < 1e-12:
                raise ValueError("ready quaternion has near-zero norm")
            return q / n

        ready_pos  = {'left': lp, 'right': rp, 'torso': tp}
        ready_quat = {'left': _norm_quat(lq), 'right': _norm_quat(rq), 'torso': _norm_quat(tq)}
        return ready_pos, ready_quat
    
    def calc_offset(self, data, ready_pos, ready_quat):
        """
        data: get_data() 결과
        ready_pos, ready_quat: utils.get_ready_pos() 결과
        return:
        offset: dict(
            head_pos_offset, head_rot_offset,
            right_pos_offset, right_rot_offset,
            left_pos_offset, left_rot_offset
        )
        """
        pos_R = data["right"]["pos"];  pos_L = data["left"]["pos"];  pos_H = data["head"]["pos"]
        deg_R = data["right"]["rotmat"]; deg_L = data["left"]["rotmat"]; deg_H = data["head"]["rotmat"]

        # 회전행렬의 역행렬은 전치(R.T)를 사용 (정규 회전행렬 가정)
        offset = {
            "head_pos_offset":  ready_pos['torso'] - np.array(pos_H),
            "head_rot_offset":  deg_H.T @ R.from_quat(ready_quat['torso']).as_matrix(),
            "right_pos_offset": ready_pos['right'] - np.array(pos_R),
            "right_rot_offset": deg_R.T @ R.from_quat(ready_quat['right']).as_matrix(),
            "left_pos_offset":  ready_pos['left'] - np.array(pos_L),
            "left_rot_offset":  deg_L.T @ R.from_quat(ready_quat['left']).as_matrix(),
        }
        return offset
    
    def is_available_area(self, data, ready_pos, ready_quat, angle_thresh_deg=15):
        # check if the meta data is in available area to initialize offset
        # 1. -0.05 < (Left_x - Right_x), (Left_z - Right_z) < 0.05
        # 2. ready_pos_L-ready_pos_R - 0.02 < Left_y - Right_y < ready_pos_L-ready_pos_R + 0.02
        # 3. ready_deg - deg_R, L, H < 15 for all axes

        pos_R = data["right"]["pos"]
        pos_L = data["left"]["pos"]
        pos_H = data["head"]["pos"]

        deg_R = data["right"]["rotmat"]
        deg_L = data["left"]["rotmat"]
        deg_H = data["head"]["rotmat"]

        # 쿼터니언 각 차이
        diff_R = calc_diff_rot(ready_quat['right'], R.from_matrix(deg_R).as_quat())
        diff_L = calc_diff_rot(ready_quat['left'],  R.from_matrix(deg_L).as_quat())
        diff_H = calc_diff_rot(ready_quat['torso'], R.from_matrix(deg_H).as_quat())

        # 1) 좌우 손목 X, Z 정렬 범위
        check_1 = (-0.05 < (pos_L[0] - pos_R[0]) < 0.05) and (-0.05 < (pos_L[2] - pos_R[2]) < 0.05)
        # 2) 좌우 손목 Y 차이가 ready 기준 ±0.02 m
        check_2 = (ready_pos['left'][1] - ready_pos['right'][1] - 0.02
                < (pos_L[1] - pos_R[1]) <
                ready_pos['left'][1] - ready_pos['right'][1] + 0.02)
        # 3) 각도 차이(쿼터니언) < 임계값
        th = np.deg2rad(angle_thresh_deg)
        check_3 = (diff_R < th) and (diff_L < th)

        available = check_1 and check_2 and check_3
        return available, [bool(check_1), bool(check_2), bool(check_3)]

    def set_init_offset(self, request, response):
        """
        request.initialize: 초기화 수행 여부
        request.check_pose: True면 자세 가용성 검사 후 offset 계산, False면 검증 생략하고 바로 offset 계산
        """
        if not request.initialize:
            response.success = False
            response.check1 = False
            response.check2 = False
            response.check3 = False
            return response

        # 요청에서 ready pose 파싱
        try:
            ready_pos, ready_quat = self._parse_ready_pose_from_request(request)
        except Exception as e:
            response.success = False
            response.check1 = False; response.check2 = False; response.check3 = False
            return response

        attempt, max_attempts = 0, 10
        self.get_logger().info(f'Starting Meta offset initialization... (check_pose={request.check_pose})')

        while attempt < max_attempts:
            data = self.get_data()
            if data is None:
                attempt += 1
                time.sleep(0.05)
                continue

            if request.check_pose:
                available, checks = self.is_available_area(data, ready_pos, ready_quat)
                if not available:
                    # 상세 로그
                    self.get_logger().warning(f'Init failed. Checks: {checks}')
                    if not checks[0]:
                        self.get_logger().warning('Check1: Hand X/Z alignment failed')
                    if not checks[1]:
                        self.get_logger().warning('Check2: Hand Y-diff out of range')
                    if not checks[2]:
                        self.get_logger().warning('Check3: Quaternion angle diff exceeds threshold')
                    attempt += 1
                    time.sleep(0.05)
                    continue
            else:
                checks = [True, True, True]  # 검증 생략 시 전부 True로

            # 공통: 오프셋 계산
            offset = self.calc_offset(data, ready_pos, ready_quat)

            # 상태 반영
            self.meta_state.head_pos_offset  = offset['head_pos_offset']
            self.meta_state.head_rot_offset  = offset['head_rot_offset']
            self.meta_state.right_pos_offset = offset['right_pos_offset']
            self.meta_state.right_rot_offset = offset['right_rot_offset']
            self.meta_state.left_pos_offset  = offset['left_pos_offset']
            self.meta_state.left_rot_offset  = offset['left_rot_offset']
            self.meta_state.is_initialized   = True

            self.get_logger().info('Meta offset initialized successfully.')
            response.success, response.check1, response.check2, response.check3 = True, checks[0], checks[1], checks[2]
            self.get_logger().info(f'Offset details:\n'
                                   f' Head Pos Offset: {self.meta_state.head_pos_offset}\n'
                                   f' Head Rot Offset: {self.meta_state.head_rot_offset}\n'
                                   f' Right Pos Offset: {self.meta_state.right_pos_offset}\n'
                                   f' Right Rot Offset: {self.meta_state.right_rot_offset}\n'
                                   f' Left Pos Offset: {self.meta_state.left_pos_offset}\n'
                                   f' Left Rot Offset: {self.meta_state.left_rot_offset}\n')
            return response

        # 실패 처리
        self.get_logger().error('Max attempts reached. Meta offset initialization failed.')
        if 'checks' not in locals():
            checks = [False, False, False]
        response.success, response.check1, response.check2, response.check3 = False, checks[0], checks[1], checks[2]
        return response
                    
    def get_meta_status(self, request, response):
        if request.request:
            self.get_logger().info("Meta status requested.")
            if self.meta_state.is_initialized:
                data = self.get_data()

                if data:
                    response.error_msg = ""

                    # Apply offsets and low pass filter
                    self.meta_state.head_position = 0.7 * self.meta_state.head_position + 0.3 * (np.array(data["head"]["pos"]) + self.meta_state.head_pos_offset)
                    self.meta_state.head_rotation = 0.7 * self.meta_state.head_rotation + 0.3 * (np.array(data["head"]["rotmat"]) + self.meta_state.head_rot_offset)
                    self.meta_state.head_quaternion = R.from_matrix(self.meta_state.head_rotation).as_quat()

                    self.meta_state.right_arm_position = 0.7 * self.meta_state.right_arm_position + 0.3 * (np.array(data["right"]["pos"]) + self.meta_state.right_pos_offset)
                    self.meta_state.right_arm_rotation = 0.7 * self.meta_state.right_arm_rotation + 0.3 * (np.array(data["right"]["rotmat"]) + self.meta_state.right_rot_offset)
                    self.meta_state.right_arm_quaternion = R.from_matrix(self.meta_state.right_arm_rotation).as_quat()

                    self.meta_state.left_arm_position = 0.7 * self.meta_state.left_arm_position + 0.3 * (np.array(data["left"]["pos"]) + self.meta_state.left_pos_offset)
                    self.meta_state.left_arm_rotation = 0.7 * self.meta_state.left_arm_rotation + 0.3 * (np.array(data["left"]["rotmat"]) + self.meta_state.left_rot_offset)
                    self.meta_state.left_arm_quaternion = R.from_matrix(self.meta_state.left_arm_rotation).as_quat()

                    right_gripper_pos = (np.linalg.norm(data["hand"]["right"]["pos"][THUMB_TIP_IDX] - data["hand"]["right"]["pos"][INDEX_TIP_IDX]) - MIN_TIP_DISTANCE) / (MAX_TIP_DISTANCE - MIN_TIP_DISTANCE)
                    left_gripper_pos = (np.linalg.norm(data["hand"]["left"]["pos"][THUMB_TIP_IDX] - data["hand"]["left"]["pos"][INDEX_TIP_IDX]) - MIN_TIP_DISTANCE) / (MAX_TIP_DISTANCE - MIN_TIP_DISTANCE)

                    self.meta_state.right_hand_position = 0.7 * self.meta_state.right_hand_position + 0.3 * float(np.clip(right_gripper_pos, 0.0, 1.0))
                    self.meta_state.left_hand_position = 0.7 * self.meta_state.left_hand_position + 0.3 * float(np.clip(left_gripper_pos, 0.0, 1.0))

                    response.head_ee_pos.position   = Float32MultiArray(data=self.meta_state.head_position.tolist())
                    response.head_ee_pos.quaternion = Float32MultiArray(data=self.meta_state.head_quaternion.tolist())
                    response.right_ee_pos.position   = Float32MultiArray(data=self.meta_state.right_arm_position.tolist())
                    response.right_ee_pos.quaternion = Float32MultiArray(data=self.meta_state.right_arm_quaternion.tolist())
                    response.left_ee_pos.position    = Float32MultiArray(data=self.meta_state.left_arm_position.tolist())
                    response.left_ee_pos.quaternion  = Float32MultiArray(data=self.meta_state.left_arm_quaternion.tolist())
                    response.right_gripper_pos = self.meta_state.right_hand_position
                    response.left_gripper_pos = self.meta_state.left_hand_position

                else:
                    response.error_msg = "No valid Meta data available."
                    if self.meta_state.head_position.size != 0:
                        response.head_ee_pos.position   = Float32MultiArray(data=self.meta_state.head_position.tolist())
                        response.head_ee_pos.quaternion = Float32MultiArray(data=self.meta_state.head_quaternion.tolist())
                        response.right_ee_pos.position   = Float32MultiArray(data=self.meta_state.right_arm_position.tolist())
                        response.right_ee_pos.quaternion = Float32MultiArray(data=self.meta_state.right_arm_quaternion.tolist())
                        response.left_ee_pos.position    = Float32MultiArray(data=self.meta_state.left_arm_position.tolist())
                        response.left_ee_pos.quaternion  = Float32MultiArray(data=self.meta_state.left_arm_quaternion.tolist())
                        response.right_gripper_pos = self.meta_state.right_hand_position
                        response.left_gripper_pos = self.meta_state.left_hand_position
                        response.error_msg += " Returning last known positions."
                    else:
                        response.error_msg += " No last known positions available."

            else:
                response.error_msg = "Meta offsets are not initialized."

        return response
    

def main(args=None):
    rclpy.init(args=args)
    meta_node = MetaNode()
    rclpy.spin(meta_node)
    meta_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()