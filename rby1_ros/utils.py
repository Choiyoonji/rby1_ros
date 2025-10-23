import numpy as np
from scipy.spatial.transform import Rotation as R

# 1) Euler → Quaternion
def euler_to_quat(euler, seq='xyz', degrees=True):
    """
    euler: array-like, shape (3,)  [roll, pitch, yaw] (seq에 맞춤)
    return: np.ndarray, shape (4,) quaternion [x, y, z, w]
    """
    euler = np.asarray(euler, dtype=float)
    return R.from_euler(seq, euler, degrees=degrees).as_quat()

# 2) Quaternion → Euler
def quat_to_euler(quat, seq='xyz', degrees=True):
    """
    quat: array-like, shape (4,)  quaternion [x, y, z, w]
    return: np.ndarray, shape (3,) Euler angles in seq order
    """
    quat = np.asarray(quat, dtype=float)
    return R.from_quat(quat).as_euler(seq, degrees=degrees)

# 3) SE(3) → position, quaternion
def se3_to_pos_quat(T):
    """
    T: np.ndarray, shape (4,4)  homogeneous transform
    return: np.ndarray, shape (3,) position
            np.ndarray, shape (4,) quaternion
    """
    T = np.asarray(T, dtype=float)
    assert T.shape == (4, 4), "T must be 4x4"
    position = T[:3, 3].copy()
    rotation_quat = R.from_matrix(T[:3, :3]).as_quat()
    return position, rotation_quat

def se3_to_pos_euler(T):
    """
    T: np.ndarray, shape (4,4)  homogeneous transform
    return: np.ndarray, shape (3,) position
            np.ndarray, shape (3,) Euler angles
    """
    T = np.asarray(T, dtype=float)
    assert T.shape == (4, 4), "T must be 4x4"
    position = T[:3, 3].copy()
    rotation_euler = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return position, rotation_euler

# 4) position + quaternion → SE(3)
def pos_to_se3(position, rotation_quat):
    """
    position: array-like, shape (3,)
    rotation_quat: array-like, shape (4,) quaternion [x, y, z, w]
    return: np.ndarray, shape (4,4) SE(3)
    """
    position = np.asarray(position, dtype=float)
    rotation_quat = np.asarray(rotation_quat, dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    T[:3, 3]  = position
    return T

# 5) mat4 to list
def mat4_to_list(T: np.ndarray):
    return T.tolist()

# 6) ndarray to list
def nd_to_list(x):
    return x.tolist() if isinstance(x, np.ndarray) else x

def calc_diff_rot(quat1, quat2):
    q1 = np.asarray(quat1)
    q2 = np.asarray(quat2)

    epsilon = 1e-15
    q1 = q1 / (np.linalg.norm(q1) + epsilon)
    q2 = q2 / (np.linalg.norm(q2) + epsilon)

    dot_product = np.dot(q1, q2)

    dot_abs = np.abs(dot_product)

    clipped_dot = np.clip(dot_abs, -1.0, 1.0)

    diff_theta_rad = 2.0 * np.arccos(clipped_dot)

    return diff_theta_rad

def get_ready_pos():
    # torso_T = np.array([[ 1.00000000e+00,  0.00000000e+00, -2.77555756e-16, -2.49596312e-17],
    #                                   [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    #                                   [ 2.77555756e-16,  0.00000000e+00,  1.00000000e+00,  1.08490130e+00],
    #                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # right_T = np.array([[-4.44089210e-16,  2.41905781e-17, -1.00000000e+00,  2.88493878e-01],
    #                                   [ 8.71557427e-02,  9.96194698e-01, -1.56125113e-17, -2.42841351e-01],
    #                                   [ 9.96194698e-01, -8.71557427e-02, -4.16333634e-16,  9.03896915e-01],
    #                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # left_T = np.array([[-4.44089210e-16, -2.41905781e-17, -1.00000000e+00,  2.88493878e-01],
    #                                  [-8.71557427e-02,  9.96194698e-01,  1.56125113e-17,  2.42841351e-01],
    #                                  [ 9.96194698e-01,  8.71557427e-02, -4.16333634e-16,  9.03896915e-01],
    #                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    torso_T = np.array([[ 1.00000000e+00,  0.00000000e+00, -2.77555756e-16, -2.49596312e-17],
                                      [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                      [ 2.77555756e-16,  0.00000000e+00,  1.00000000e+00,  1.08490130e+00],
                                      [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    right_T = np.array([[-4.44089210e-16,  0.00000000e+00, -1.00000000e+00,  2.88493878e-01],
                                    [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -2.20000000e-01],
                                    [ 1.00000000e+00,  0.00000000e+00, -4.44089210e-16,  9.02899640e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    left_T = np.array([[-4.44089210e-16,  0.00000000e+00, -1.00000000e+00,  2.88493878e-01],
                                    [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  2.20000000e-01],
                                    [ 1.00000000e+00,  0.00000000e+00, -4.44089210e-16,  9.02899640e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    
    torso_pos, torso_quat = se3_to_pos_quat(torso_T)
    right_pos, right_quat = se3_to_pos_quat(right_T)
    left_pos, left_quat = se3_to_pos_quat(left_T)

    T = {"torso": torso_T, "right": right_T, "left": left_T}
    pos = {"torso": torso_pos, "right": right_pos, "left": left_pos}
    quat = {"torso": torso_quat, "right": right_quat, "left": left_quat}

    return T, pos, quat