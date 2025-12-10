import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.functions_np import Quat2Rot, DH

FINGET_DIST_META = {
    "index": 0.087573798, 
    "middle": 0.098310369, 
    "ring": 0.092910406, 
    "little": 0.08313207384672585, 
    "thumb": 0.098747359371764078,
}

FINGER_DIST_INSPIRE = {
    "index": 0.07326299, 
    "middle": 0.07624115, 
    "ring": 0.073262975, 
    "little": 0.06611647, 
    "thumb": 0.10493251
}

_FINGER_ORDER = ("index", "middle", "ring", "little", "thumb")
_COL_SLICES = {
    "index":  slice(0,1),
    "middle": slice(1,2),
    "ring":   slice(2,3),
    "little": slice(3,4),
    "thumb":  slice(4,6),
}

def meta2inspire(meta_pos):
    """Meta의 손가락 회전행렬 → Inspire RH56F1 손가락 회전행렬 변환"""
    if len(meta_pos) == 0:
        meta_pos = np.array(meta_pos, dtype=np.float32).reshape((1,3))
    P_inspire = np.zeros((len(meta_pos),3), dtype=np.float32)
    P_inspire[:, 0] = -meta_pos[:, 2]
    P_inspire[:, 1] = meta_pos[:, 1]
    P_inspire[:, 2] = meta_pos[:, 0]
    return P_inspire

def _pack_pos_error(pos_error):
    """pos_error가 dict({'index':(1,3), ...}) 혹은 (5,3) array 모두 지원 → (5,3)"""
    if isinstance(pos_error, dict):
        return np.vstack([pos_error[k].reshape(1,3) for k in _FINGER_ORDER]).astype(np.float32, copy=False)
    arr = np.asarray(pos_error, dtype=np.float32)
    # 기대형상: (5,3)
    if arr.shape != (5,3):
        raise ValueError(f"pos_error shape must be (5,3), got {arr.shape}")
    return arr

def _pack_Jp(J):
    """J가 dict({'index':(3,1), ..., 'thumb':(3,2)}) 또는 (5,3,6) ndarray → (5,3,6)"""
    if isinstance(J, dict):
        Jp = np.zeros((5,3,6), dtype=np.float32)
        for f_idx, name in enumerate(_FINGER_ORDER):
            Jp[f_idx, :, _COL_SLICES[name]] = np.asarray(J[name], dtype=np.float32)
        return Jp
    J = np.asarray(J, dtype=np.float32)
    if J.shape != (5,3,6):
        raise ValueError(f"J shape must be (5,3,6), got {J.shape}")
    return J

def spring_damper_ik_step(q, pos_error, J, q_prev, K=5e5, D=0.1, beta=0.1, step_clip=None):
    """
    자코비안 기반 스프링-댐퍼 IK 업데이트 (벡터화)
    q        : (6,)
    pos_error: dict({'index':(1,3),...}) 또는 (5,3)
    J        : dict({'index':(3,1),...,'thumb':(3,2)}) 또는 (5,3,6)
    q_prev   : (6,)
    K, D, beta: 스프링/댐퍼/균등화 계수
    step_clip: float or None. 관절 증분 L2 노름 상한(안정성 옵션)
    """
    q = np.asarray(q, dtype=np.float32)
    q_prev = np.asarray(q_prev, dtype=np.float32)

    e = _pack_pos_error(pos_error)        # (5,3)
    Jp = _pack_Jp(J)                      # (5,3,6)

    dq_prev = (q - q_prev).reshape(6,1)   # (6,1)
    # EE 속도 추정: x_dot = J * dq_prev  => (5,3,1)
    x_dot = Jp @ dq_prev                  # (5,3,1)

    # 가상 힘: F = K*e - D*x_dot
    F = K * e.reshape(5,3,1) - D * x_dot  # (5,3,1)
    # 관절 속도 업데이트: sum_fingers (J^T * F)
    # (5,6,3) @ (5,3,1) -> (5,6,1) -> sum over fingers -> (6,1)
    dq_ = (np.transpose(Jp, (0,2,1)) @ F).sum(axis=0).reshape(6,)  # (6,)

    # 간단한 평균-리그 정규화(원 코드 유지)
    dq_mean = dq_.mean()
    dq = dq_ + beta * (dq_mean - dq_)

    # 안전 클립(선택): 너무 큰 스텝 방지
    if step_clip is not None:
        nrm = np.linalg.norm(dq)
        if nrm > step_clip and nrm > 0:
            dq *= (step_clip / nrm)

    return dq.astype(np.float32).reshape(6,)

def spring_damper_ik(
    q_init,
    target_pos,  # (5,3)
    type,
    base_pos=np.array([0, 0, 0], dtype=np.float32),
    base_rot=np.eye(3, dtype=np.float32),
    max_iterations=5,
    tol=1e-3,
    K=50.0,
    D=0.1,
    beta=0.1,
    step_clip=None
):
    """
    q_init    : (6,)
    target_pos: (5,3) (각 손가락 EE 목표)
    kin       : RH56F1_Kinematic 호환 객체 (DH_forward, get_jnt_axis, get_jacobian_pos_only or get_jacobian)
    """
    q = np.asarray(q_init, dtype=np.float32).squeeze().copy()
    q_prev = q.copy()
    kin = RH56F1_Kinematic(type=type, base_pos=base_pos, base_rot=base_rot)
    tau = 0.0
    for _ in range(max_iterations):
        p_EE, _, p_lnk, r_lnk = kin.DH_forward(q)
        jnt_axis = kin.get_jnt_axis(r_lnk)

        # 위치 오차 (5,3)
        pos_error = np.asarray(target_pos, dtype=np.float32) - np.asarray(p_EE, dtype=np.float32)

        # 자코비안: 가능하면 벡터화된 pos-only 사용
        if hasattr(kin, "get_jacobian_pos_only"):
            Jp = kin.get_jacobian_pos_only(p_EE, p_lnk, jnt_axis)  # (5,3,6)
        else:
            J_p, _, _ = kin.get_jacobian(p_EE, p_lnk, jnt_axis)
            Jp = _pack_Jp(J_p)  # dict → (5,3,6)

        dq = spring_damper_ik_step(q, pos_error, Jp, q_prev, K=K, D=D, beta=beta, step_clip=step_clip)

        q_prev[:] = q
        q = q + dq
        tau += dq
        # 수렴 체크
        if np.linalg.norm(pos_error) < tol:
            break
    return q.reshape(6,1)

def mapping_meta2inspire_l(meta_l, finger_dist_meta, p_lnk_inspire, r_lnk_inspire, finger_dist_inspire):
    name_dict = ["index", "middle", "ring", "little", "thumb"]
    p_EE_meta, p_lnk_meta, r_lnk_meta = meta_l
    p_EE_meta = meta2inspire(p_EE_meta)
    p_lnk_meta = meta2inspire(p_lnk_meta)
    coeff_mapping, offset = {}, {}
    for i, name in enumerate(name_dict):
        # mapping coefficient
        coeff_mapping[name] = finger_dist_inspire[name] / finger_dist_meta[name] 
        # apply coeff to meta
        p_lnk_meta[4*i:4*(i+1)] *= coeff_mapping[name]
        p_EE_meta[i] *= coeff_mapping[name]
        # mapping offset
        if name == "thumb":
            # y_axis = r_lnk_inspire[i + 1][:, 2]
            # for j in range(4):
            #     angle_rad = np.deg2rad(10.0 * i)
            #     K = np.array([
            #         [0, -y_axis[2], y_axis[1]],
            #         [y_axis[2], 0, -y_axis[0]],
            #         [-y_axis[1], y_axis[0], 0]
            #     ])
            #     R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
            #     p_lnk_meta[4*i + j] = p_lnk_meta[4*i + j] @ R.T
            # p_EE_meta[i] = p_EE_meta[i] @ R.T
            offset[name] = r_lnk_inspire[i + 1] @ np.array([0, 0, 0])
            offset[name] += p_lnk_inspire[i + 1] - p_lnk_meta[4 * i]
        else:
            # x_axis = r_lnk_inspire[i+1][:, 0]
            # angle_rad = np.deg2rad(-3.0 * i)
            # K = np.array([
            #     [0, -x_axis[2], x_axis[1]],
            #     [x_axis[2], 0, -x_axis[0]],
            #     [-x_axis[1], x_axis[0], 0]
            # ])
            # R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
            # p_lnk_meta[4*i:4*(i+1)] = p_lnk_meta[4*i:4*(i+1)] @ R.T
            # p_EE_meta[i] = p_EE_meta[i] @ R.T
            # mapping offset
            offset[name] = r_lnk_inspire[i + 1] @ np.array([0, 0, 0.0067])
            offset[name] += p_lnk_inspire[i + 1] - p_lnk_meta[4 * i]

        # apply offset
        p_lnk_meta[4*i:4*(i+1)] += offset[name]
        p_EE_meta[i] += offset[name]
    return np.array(p_EE_meta), p_lnk_meta

def mapping_meta2inspire_r(meta_r, finger_dist_meta, p_lnk_inspire, r_lnk_inspire, finger_dist_inspire):
    name_dict = ["index", "middle", "ring", "little", "thumb"]
    p_EE_meta, p_lnk_meta, r_lnk_meta = meta_r
    p_EE_meta = meta2inspire(p_EE_meta)
    p_lnk_meta = meta2inspire(p_lnk_meta)
    coeff_mapping, offset = {}, {}
    for i, name in enumerate(name_dict):
        # mapping coefficient
        coeff_mapping[name] = finger_dist_inspire[name] / finger_dist_meta[name] 
        # apply coeff to meta
        p_lnk_meta[4*i:4*(i+1)] *= coeff_mapping[name]
        p_EE_meta[i] *= coeff_mapping[name]
        # mapping offset
        if name == "thumb":
            # y_axis = r_lnk_inspire[i][:, 2]
            # for j in range(4):
            #     angle_rad = np.deg2rad(10.0 * i)
            #     K = np.array([
            #         [0, -y_axis[2], y_axis[1]],
            #         [y_axis[2], 0, -y_axis[0]],
            #         [-y_axis[1], y_axis[0], 0]
            #     ])
            #     R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
            #     p_lnk_meta[4*i + j] = p_lnk_meta[4*i + j] @ R.T
            # p_EE_meta[i] = p_EE_meta[i] @ R.T
            offset[name] = r_lnk_inspire[i + 1] @ np.array([0, 0, 0])
            offset[name] += p_lnk_inspire[i + 1] - p_lnk_meta[4 * i]
        else:
            # x_axis = r_lnk_inspire[i][:, 0]
            # angle_rad = np.deg2rad(4.0 * i)
            # K = np.array([
            #     [0, -x_axis[2], x_axis[1]],
            #     [x_axis[2], 0, -x_axis[0]],
            #     [-x_axis[1], x_axis[0], 0]
            # ])
            # R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
            # p_lnk_meta[4*i:4*(i+1)] = p_lnk_meta[4*i:4*(i+1)] @ R.T
            # p_EE_meta[i] = p_EE_meta[i] @ R.T
            # mapping offset
            offset[name] = r_lnk_inspire[i + 1] @ np.array([0, 0, -0.0067])
            offset[name] += p_lnk_inspire[i + 1] - p_lnk_meta[4 * i]

        # apply offset
        p_lnk_meta[4*i:4*(i+1)] += offset[name]
        p_EE_meta[i] += offset[name]
    return np.array(p_EE_meta), p_lnk_meta

class RH56F1_Kinematic:
    def __init__(self, type="left", base_pos=np.array([0, 0, 0], dtype=np.float32), base_rot=np.eye(3, dtype=np.float32)): 
        self.type = type
        self.dof_nums = 6
        self.finger_nums = 5
        self.base_pos = base_pos
        self.mj_world_rot2my_rot = base_rot.reshape((3,3)) # world 좌표계 설정
        if type in 'left':
            self.base2index = DH(np.deg2rad(-90), 0.13554, 0.034377, 0) @ DH(np.deg2rad(-90), 0, -0.000707, np.deg2rad(86.5))
            self.base2middle = DH(np.deg2rad(-90), 0.13648, 0.015767, 0) @ DH(np.deg2rad(-90), 0, -0.000707, np.deg2rad(90))
            self.base2ring = DH(np.deg2rad(-90), 0.13643, -0.0030876, 0) @ DH(np.deg2rad(-90), 0, -0.000707, np.deg2rad(93))
            self.base2little = DH(np.deg2rad(-90), 0.13531, -0.021094, 0) @ DH(np.deg2rad(-90), 0, -0.000707, np.deg2rad(96))
            self.base2thumb = DH(np.deg2rad(-90), 0.070511, 0.027241, 0) @ DH(np.deg2rad(90), 0, 0.01836, 0)
            self.Tt0_1 = DH(np.deg2rad(-90), -0.0049121, 0.01349, np.deg2rad(90)) @ DH(0, -0.005034, 0, 0)
            self.Tt1_2 = DH(np.deg2rad(90), -0.0029656, 0.034601, 0) @ DH(np.deg2rad(-90), 0, 0.033907, 0)
            self.Tt2_3 = DH(np.deg2rad(90), 0.0013, 0.014648, 0) @ DH(np.deg2rad(-90), 0, 0.01441, 0)
            self.Tt3_E = DH(0, 0.0067577, 0.016676, np.deg2rad(-90)) @ DH(0, 0.025314, 0, 0)
        else:
            self.base2index = DH(np.deg2rad(90), 0.13554, 0.034377, 0) @ DH(np.deg2rad(90), 0, -0.000707, np.deg2rad(93.5))
            self.base2middle = DH(np.deg2rad(90), 0.13648, 0.015767, 0) @ DH(np.deg2rad(90), 0, -0.000707, np.deg2rad(90))
            self.base2ring = DH(np.deg2rad(90), 0.13643, -0.0030876, 0) @ DH(np.deg2rad(90), 0, -0.000707, np.deg2rad(87))
            self.base2little = DH(np.deg2rad(90), 0.13531, -0.021094, 0) @ DH(np.deg2rad(90), 0, -0.000707, np.deg2rad(84))
            self.base2thumb = DH(np.deg2rad(90), 0.070511, 0.027241, np.deg2rad(180)) @ DH(np.deg2rad(-90), 0, -0.01836, 0)
            self.Tt0_1 = DH(np.deg2rad(90), 0.0049121, 0.01349, np.deg2rad(-90)) @ DH(0, 0.005034, 0, 0)
            self.Tt1_2 = DH(np.deg2rad(90), 0.0029656, 0.034601, 0) @ DH(np.deg2rad(-90), 0, 0.033907, 0)
            self.Tt2_3 = DH(np.deg2rad(90), -0.0013, 0.014648, 0) @ DH(np.deg2rad(-90), 0, 0.01441, 0)
            self.Tt3_E = DH(0, -0.0067577, 0.016676, np.deg2rad(-90)) @ DH(0, 0.025314, 0, 0)
        
        self.T0_1 = DH(np.deg2rad(90), 0.001, 0.032854, 0) @ DH(np.deg2rad(-90), 0, -0.0031056, 0)
        self.Ti1_E = DH(np.deg2rad(90), 0.006, 0.039631, 0) @ DH(np.deg2rad(-90), 0, -0.0049188, 0)
        self.Tm1_E = DH(np.deg2rad(90), 0.0061, 0.042512, 0) @ DH(np.deg2rad(-90), 0, -0.0059647, 0)
        self.Tr1_E = DH(np.deg2rad(90), 0.006, 0.039631, 0) @ DH(np.deg2rad(-90), 0, -0.0049188, 0)
        self.Tl1_E = DH(np.deg2rad(90), 0.0063511, 0.032544, 0) @ DH(np.deg2rad(-90), 0, -0.0051203, 0)
        

    # def convert_l2q(self, l, mode='isaac') -> np.ndarray:
    #     '''
    #     linear actuator 값을 revolute 값으로 변환
    #     l : [mm] 단위
    #     mode : 'isaac' or 'real'
    #     q : [deg] 단위
    #     실제 실험표 참고 -> fingers coeff vec
    #     엄지 제외 : 177.24 ~ 75.49 [deg] | Issac : 0 ~ 87.58 [deg] | linear 모터 : 0 ~ 10.735 [mm]
    #     엄지 측면 : 189.79 ~ 43.28 [deg] | Issac : 0 ~ 120.0 [deg] | linear 모터 : 0 ~ 11.500 [mm]
    #     엄지 굽힘 : 135.22 ~ 99.22 [deg] | Issac : 0 ~ 27.19 [deg] | linear 모터 : 0 ~ 11.500 [mm]
    #     '''
    #     except_thumb_coeff_vec = np.array([-0.000634697, 0.0149918, -0.1619489, 0.9344705, -10.949963, 177.273787], dtype=np.float32)
    #     thumb_yaw_coeff_vec = np.array([0.00000366301, -0.0000423763, -0.000712581, 0.0111166, -12.774385, 189.79223], dtype=np.float32)
    #     thumb_bending_coeff_vec = np.array([0.00000951131, -0.000174143, -0.000352853, 0.0261404, -3.2864925, 135.22145], dtype=np.float32)
        
    #     if mode == 'isaac':
    #         except_thumb_coeff_vec *= -1
    #         thumb_yaw_coeff_vec *= -1
    #         thumb_bending_coeff_vec *= -1
    #         except_thumb_coeff_vec[-1], thumb_yaw_coeff_vec[-1], thumb_bending_coeff_vec[-1] = 0.0, 0.0, 0.0

    #     l_mat = np.array([to_polynominal_5(l_) for l_ in l], dtype=np.float32)
    #     coeff_mat = np.array([except_thumb_coeff_vec, except_thumb_coeff_vec, except_thumb_coeff_vec, except_thumb_coeff_vec, thumb_yaw_coeff_vec, thumb_bending_coeff_vec], dtype=np.float32)
        
    #     q = l_mat * coeff_mat
    #     return q
    
    # def convert_q2l(self, q, mode='isaac') -> np.ndarray:
    #     '''
    #     revolute actuator 값을 linear 값으로 변환
    #     q : [deg] 단위
    #     mode : 'isaac' or 'real'
    #     l : [mm] 단위
    #     실제 실험표 참고 -> fingers coeff vec
    #     엄지 제외 : 177.24 ~ 75.49 [deg] | Issac : 0 ~ 87.58 [deg] | linear 모터 : 0 ~ 10.735 [mm]
    #     엄지 측면 : 189.79 ~ 43.28 [deg] | Issac : 0 ~ 120.0 [deg] | linear 모터 : 0 ~ 11.500 [mm]
    #     엄지 굽힘 : 135.22 ~ 99.22 [deg] | Issac : 0 ~ 27.19 [deg] | linear 모터 : 0 ~ 11.500 [mm]
    #     '''
    #     except_thumb_coeff_vec = np.array([-5.43395e-11, 4.61468e-8, -8.94849e-8, -1.3286e-4, 0.0391729, 11.019686])
    #     thumb_yaw_coeff_vec = np.array([-8.57919e-13, 6.87741e-10, -1.85983e-7, 2.12919e-5, -12.774385, 347.90514])
    #     thumb_bending_coeff_vec = np.array([-1.21058e-8, 7.39162e-6, -0.0017892, 0.2149159, -13.147242, 14.913055])
        
    #     if mode == 'isaac':
    #         q[:4] = -1 * q[:4] + 177.24
    #         q[4] = -1 * q[4] + 189.79
    #         q[5] = -1 * q[5] + 135.22 

    #     q_mat = np.array([to_polynominal_5(q_) for q_ in q])
    #     coeff_mat = np.array([except_thumb_coeff_vec, except_thumb_coeff_vec, except_thumb_coeff_vec, except_thumb_coeff_vec, thumb_yaw_coeff_vec, thumb_bending_coeff_vec])
        
    #     l = q_mat * coeff_mat
    #     return l

    def DH_forward(self, q: np.ndarray):
        '''
        q : [rad]
        '''
        P_lnk = np.empty((7, 3), dtype=np.float32)
        R_lnk = np.empty((7, 3, 3), dtype=np.float32)
        P_EE  = np.empty((5, 3), dtype=np.float32)
        R_EE  = np.empty((5, 3, 3), dtype=np.float32)

        base_T = np.array([[1, 0, 0, self.base_pos[0]],
                           [0, 1, 0, self.base_pos[1]],
                           [0, 0, 1, self.base_pos[2]],
                           [0, 0, 0,                1]], dtype=np.float32)
        base_T[:3, :3] = self.mj_world_rot2my_rot
        P_lnk[0] = self.base_pos
        R_lnk[0] = self.mj_world_rot2my_rot

        # index
        Ti0 = base_T @ self.base2index @ DH(q[0], 0, 0, 0)
        Ti1 = Ti0 @ self.T0_1 @ DH(q[0] * 1.1169, 0, 0, 0)
        P_lnk[1] = Ti0[:3, 3]
        R_lnk[1] = Ti0[:3, :3]

        # middle
        Tm0 = base_T @ self.base2middle @ DH(q[1], 0, 0, 0)
        Tm1 = Tm0 @ self.T0_1 @ DH(q[1] * 1.1169, 0, 0, 0)
        P_lnk[2] = Tm0[:3, 3]
        R_lnk[2] = Tm0[:3, :3]

        # ring
        Tr0 = base_T @ self.base2ring @ DH(q[2], 0, 0, 0)
        Tr1 = Tr0 @ self.T0_1 @ DH(q[2] * 1.1169, 0, 0, 0)
        P_lnk[3] = Tr0[:3, 3]
        R_lnk[3] = Tr0[:3, :3]

        # little
        Tl0 = base_T @ self.base2little @ DH(q[3], 0, 0, 0)
        Tl1 = Tl0 @ self.T0_1 @ DH(q[3] * 1.1169, 0, 0, 0)
        P_lnk[4] = Tl0[:3, 3]
        R_lnk[4] = Tl0[:3, :3]

        # thumb
        Tt0 = base_T @ self.base2thumb @ DH(q[4], 0, 0, 0)
        Tt1 = Tt0 @ self.Tt0_1 @ DH(q[5], 0, 0, 0)
        Tt2 = Tt1 @ self.Tt1_2 @ DH(q[5] * 1.1425, 0, 0, 0)
        Tt3 = Tt2 @ self.Tt2_3 @ DH(q[5] * 0.857789, 0, 0, 0)
        P_lnk[5] = Tt0[:3, 3]
        R_lnk[5] = Tt0[:3, :3]
        P_lnk[6] = Tt1[:3, 3]
        R_lnk[6] = Tt1[:3, :3]

        # End Effecter
        TiE = Ti1 @ self.Ti1_E
        TmE = Tm1 @ self.Tm1_E
        TrE = Tr1 @ self.Tr1_E
        TlE = Tl1 @ self.Tl1_E
        TtE = Tt3 @ self.Tt3_E
        P_EE[0] = TiE[:3, 3]
        R_EE[0] = TiE[:3, :3]
        P_EE[1] = TmE[:3, 3]
        R_EE[1] = TmE[:3, :3]
        P_EE[2] = TrE[:3, 3]
        R_EE[2] = TrE[:3, :3]
        P_EE[3] = TlE[:3, 3]
        R_EE[3] = TlE[:3, :3]
        P_EE[4] = TtE[:3, 3]
        R_EE[4] = TtE[:3, :3]
        
        return P_EE, R_EE, P_lnk, R_lnk
    
    def get_jnt_axis(self, R_lnk):
        '''
        Use it after DH_forward
        '''
        z = np.array([0, 0, 1], dtype=np.float32)
        return (R_lnk[1:] @ z)

    def get_jacobian_pos_only(self, end_effector_positions, link_positions, joint_axes):
        """
        end_effector_positions: (5,3)
        link_positions: (7,3)   # base + 6 joints
        joint_axes: (6,3)
        반환: Jp (5,3,6)  # 손가락×xyz×관절
        - index..little는 각자 1 DoF, thumb은 2 DoF(4,5)
        """
        Jp = np.zeros((5, 3, 6), dtype=np.float32)

        # 손가락별, 관여 관절 인덱스
        finger_q_cols = [
            [0],      # index
            [1],      # middle
            [2],      # ring
            [3],      # little
            [4, 5],   # thumb
        ]

        # 각 손가락 EE 위치
        for f, cols in enumerate(finger_q_cols):
            ee = end_effector_positions[f]  # (3,)
            for j in cols:
                pj = link_positions[j+1]    # joint position
                zj = joint_axes[j]          # axis
                Jp[f, :, j] = np.cross(zj, (ee - pj))
        return Jp
    
    def get_jacobian(self, end_effector_positions, link_positions, joint_axes):
        '''
        end_effector_positions : DH_forward -> P_EE
        joint_position : DH_forward -> P_lnk
        joint_axes : get_jnt_axis -> joint_axis
        '''
        J_p = np.zeros((3, self.dof_nums), dtype=np.float32)
        J_r = np.zeros((3, self.dof_nums), dtype=np.float32)

        for i in range(self.dof_nums):
            link_pos = link_positions[i + 1]
            joint_axis = joint_axes[i]
            if i < 4:
                ee_pos = end_effector_positions[i]
            else:
                ee_pos = end_effector_positions[-1]

            J_p[:, i] = np.cross(joint_axis, ee_pos - link_pos)
            J_r[:, i] = joint_axis
        
        J_p = {
            "index": J_p[:, 0:1],
            "middle": J_p[:, 1:2],
            "ring": J_p[:, 2:3],
            "little": J_p[:, 3:4],
            "thumb": J_p[:, 4:6],
        }
        J_r = {
            "index": J_r[:, 0:1],
            "middle": J_r[:, 1:2],
            "ring": J_r[:, 2:3],
            "little": J_r[:, 3:4],
            "thumb": J_r[:, 4:6],
        }
        J_pr = {
            "index": np.vstack((J_p["index"], J_r["index"])),
            "middle": np.vstack((J_p["middle"], J_r["middle"])),
            "ring": np.vstack((J_p["ring"], J_r["ring"])),
            "little": np.vstack((J_p["little"], J_r["little"])),
            "thumb": np.vstack((J_p["thumb"], J_r["thumb"])),
        }
        return J_p, J_r, J_pr
