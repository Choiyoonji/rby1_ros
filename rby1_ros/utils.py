import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List, Union, Dict

# ==============================================================================
# 1. Custom Quaternion Math (Convention: [w, x, y, z])
#    직접 구현된 수학 함수들은 실수부(w)가 먼저 오는 순서를 따릅니다.
# ==============================================================================

def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return q 
    return q / norm

def conjugation(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugation (w, -x, -y, -z)"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def inverse_quat(q: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(q)
    return conjugation(q) / (q_norm ** 2 + 1e-7)

def mul_quat(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication. Assumes [w, x, y, z] order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])

def quat_diff(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Computes relative rotation q_rel = q1^-1 * q2"""
    return mul_quat(inverse_quat(q1), q2)

def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Converts quaternion [w, x, y, z] to rotation vector."""
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2 * np.arccos(w)
    sin_half = np.sqrt(1 - w*w)
    
    if sin_half < 1e-6:
        return np.zeros(3)
    
    axis = q[1:] / sin_half
    return axis * angle

def rotvec_to_quat(vec: np.ndarray) -> np.ndarray:
    """Converts rotation vector to quaternion [w, x, y, z]."""
    angle = np.linalg.norm(vec)
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    axis = vec / angle
    half_angle = angle / 2
    return np.array([np.cos(half_angle), 
                     axis[0]*np.sin(half_angle), 
                     axis[1]*np.sin(half_angle), 
                     axis[2]*np.sin(half_angle)])

def Quat2Rot(quat: np.ndarray) -> np.ndarray:
    """Converts quaternion [w, x, y, z] to 3x3 Rotation Matrix."""
    q_w, q_x, q_y, q_z = quat
    
    return np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
        [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
        [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
    ])

def calc_diff_rot(quat1: np.ndarray, quat2: np.ndarray) -> float:
    """Calculates angular difference (radians) between two quaternions."""
    q1 = np.asarray(quat1)
    q2 = np.asarray(quat2)

    epsilon = 1e-15
    q1 = q1 / (np.linalg.norm(q1) + epsilon)
    q2 = q2 / (np.linalg.norm(q2) + epsilon)

    dot_product = np.dot(q1, q2)
    return 2.0 * np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0))


# ==============================================================================
# 2. Scipy Wrapper Functions (Convention: [x, y, z, w])
#    Scipy는 허수부(x, y, z)가 먼저 오고 실수부(w)가 마지막에 오는 순서를 따릅니다.
# ==============================================================================

def euler_to_quat(euler, seq='xyz', degrees=True) -> np.ndarray:
    """
    Args:
        euler: [roll, pitch, yaw]
    Returns:
        quaternion [x, y, z, w] (Scipy convention)
    """
    euler = np.asarray(euler, dtype=float)
    return R.from_euler(seq, euler, degrees=degrees).as_quat()

def quat_to_euler(quat, seq='xyz', degrees=True) -> np.ndarray:
    """
    Args:
        quat: [x, y, z, w] (Scipy convention)
    Returns:
        euler angles [roll, pitch, yaw]
    """
    quat = np.asarray(quat, dtype=float)
    return R.from_quat(quat).as_euler(seq, degrees=degrees)


# ==============================================================================
# 3. SE(3) Transformation Helpers
# ==============================================================================

def se3_to_pos_quat(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        position: [x, y, z]
        quaternion: [x, y, z, w] (Scipy convention from as_quat)
    """
    T = np.asarray(T, dtype=float)
    position = T[:3, 3].copy()
    rotation_quat = R.from_matrix(T[:3, :3]).as_quat()
    return position, rotation_quat

def se3_to_pos_euler(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float)
    position = T[:3, 3].copy()
    rotation_euler = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return position, rotation_euler

def pos_to_se3(position, rotation_quat) -> np.ndarray:
    """
    Args:
        rotation_quat: [x, y, z, w] (Scipy convention)
    """
    position = np.asarray(position, dtype=float)
    rotation_quat = np.asarray(rotation_quat, dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    T[:3, 3]  = position
    return T


# ==============================================================================
# 4. Utilities & Constants
# ==============================================================================

def mat4_to_list(T: np.ndarray) -> List[List[float]]:
    return T.tolist()

def nd_to_list(x):
    return x.tolist() if isinstance(x, np.ndarray) else x

def get_ready_pos() -> Tuple[Dict, Dict, Dict]:
    # Hardcoded transform matrices
    torso_T = np.array([
        [1.0, 0.0, -2.77555756e-16, -2.49596312e-17],
        [0.0, 1.0, 0.0, 0.0],
        [2.77555756e-16, 0.0, 1.0, 1.08490130e+00],
        [0.0, 0.0, 0.0, 1.0]
    ])

    right_T = np.array([
        [-4.44089210e-16, 0.0, -1.0, 2.88493878e-01],
        [0.0, 1.0, 0.0, -2.20000000e-01],
        [1.0, 0.0, -4.44089210e-16, 9.02899640e-01],
        [0.0, 0.0, 0.0, 1.0]
    ])

    left_T = np.array([
        [-4.44089210e-16, 0.0, -1.0, 2.88493878e-01],
        [0.0, 1.0, 0.0, 2.20000000e-01],
        [1.0, 0.0, -4.44089210e-16, 9.02899640e-01],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Compute derived representations
    torso_pos, torso_quat = se3_to_pos_quat(torso_T)
    right_pos, right_quat = se3_to_pos_quat(right_T)
    left_pos, left_quat   = se3_to_pos_quat(left_T)

    T_dict = {"torso": torso_T, "right": right_T, "left": left_T}
    pos_dict = {"torso": torso_pos, "right": right_pos, "left": left_pos}
    quat_dict = {"torso": torso_quat, "right": right_quat, "left": left_quat}

    return T_dict, pos_dict, quat_dict


# ==============================================================================
# Main / Test
# ==============================================================================
if __name__ == "__main__":
    # Test values
    test_q = [0.3607517182826996, 0.5801421403884888, 0.5909604430198669, -0.42902106046676636]
    
    # 주의: Scipy wrapper인 quat_to_euler는 [x,y,z,w] 입력을 기대합니다.
    # 만약 test_q가 [w,x,y,z]라면 순서를 바꿔서 넣어야 정확한 결과가 나옵니다.
    print("Euler angles (assuming input is xyzw):", quat_to_euler(test_q))