import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R

def rotmat_weight(rotmat,rxryrz_weight=[-1.,-1.,1.]):
    rotvec = R.from_matrix(rotmat).as_rotvec()
    rotvec[0] *= rxryrz_weight[0];
    rotvec[1] *= rxryrz_weight[1];
    rotvec[2] *= rxryrz_weight[2];
    r_new = R.from_rotvec(rotvec)
    return r_new.as_matrix()

def pos_weight(pos, xyz_weight=[-1.,-1.,1.]):
    return np.multiply(np.array(pos),xyz_weight)

def Rot_x(q):
    q = np.deg2rad(q)
    r = np.array([[1,           0,          0],
                  [0,   np.cos(q), -np.sin(q)],
                  [0,   np.sin(q),  np.cos(q)]])
    return r

def Rot_y(q):
    q = np.deg2rad(q)
    r = np.array([[ np.cos(q),      0,  np.sin(q)],
                  [         0,      1,          0],
                  [-np.sin(q),      0,  np.cos(q)]])
    return r

def Rot_z(q):
    q = np.deg2rad(q)
    r = np.array([[np.cos(q),  -np.sin(q),          0],
                  [np.sin(q),   np.cos(q),          0],
                  [        0,           0,          1]])
    return r

def DH(phi, d, a, alpha):
    T = np.array([[np.cos(phi), -np.sin(phi) * np.cos(alpha),  np.sin(phi) * np.sin(alpha),  a * np.cos(phi)],
                [np.sin(phi),  np.cos(phi) * np.cos(alpha), -np.cos(phi) * np.sin(alpha),  a * np.sin(phi)],
                [0,            np.sin(alpha),               np.cos(alpha),               d],
                [0,            0,                           0,                           1]], dtype=np.float32)
    return T

def skew(vec):     
    skew = np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])
    return skew


def Rot2EulerZYZ(rot):
    '''
    Euler Angle Formulas
    @article{eberly2008euler,
    title = { Euler angle formulas },
    author = { Eberly, David },
    journal = { Geometric Tools, LLC, Technical Report },
    pages = { 1--18 },
    year = { 2008 }
    }
    '''
    euler = np.zeros(shape=(3,))
    if rot[2,2] < 1:
        if rot[2,2] > -1:
            euler[0] = np.atan2(rot[1,2], rot[0,2])
            euler[1] = np.acos(rot[2,2])
            euler[2] = np.atan2(rot[2,1], -rot[2,0])
        else:
            euler[0] = -np.atan2(rot[1,0], rot[1,1])
            euler[1] = pi
            euler[2] = 0
    else:
        euler[0] = np.atan2(rot[1,0], rot[1,1])
        euler[1] = 0
        euler[2] = 0
        
    return euler.T

def Rot2EulerZXY(rot):
    '''
    Euler Angle Formulas
    @article{eberly2008euler,
    title = { Euler angle formulas },
    author = { Eberly, David },
    journal = { Geometric Tools, LLC, Technical Report },
    pages = { 1--18 },
    year = { 2008 }
    }
    '''
    euler = np.zeros((3,))
    if rot[2,1] < 1:
        if rot[2,1] > -1:
            euler[0] = np.arctan2(-rot[0,1], rot[1,1])
            euler[1] = np.arcsin(rot[2,1])
            euler[2] = np.arctan2(-rot[2,0], rot[2,2])
        else:
            euler[0] = -np.arctan2(rot[0,2], rot[0,0])
            euler[1] = -pi/2
            euler[2] = 0
    else:
        euler[0] = np.arctan2(rot[0,2], rot[0,0])
        euler[1] = pi/2
        euler[2] = 0
    
    return euler.T

def Rot2Quat(R):
    # Calculate the trace of the matrix
    trace = np.trace(R)
    
    # Initialize quaternion array
    q = np.zeros(4)
    
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    
    return q

def Quat2Rot(quat):
    q_w, q_x, q_y, q_z = quat[0], quat[1], quat[2], quat[3]

    # 회전 행렬 계산
    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
        [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
        [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
    ])

    return R

def conjugation(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def inverse_quat(q):
    q_norm = np.linalg.norm(q)
    q_inverse = conjugation(q) / (q_norm ** 2 + 1e-7)
    return q_inverse

def mul_quat(q1,q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    mul = np.array([w, x, y, z])
    
    return mul

def quat_diff(q1, q2):
    """q1, q2: shape (..., 4), wxyz 순서"""

    q_rel = mul_quat(inverse_quat(q1), q2)
    return q_rel

def orientation_error(current, desired):
    cc = conjugation(current)
    q_r = mul_quat(desired, cc)
    return q_r[1:] * np.sign(q_r[0])

def Quat2Omega(quat, quatdot):
    trans = 2 * mul_quat(quatdot, inverse_quat(quat))
    Omega = np.array([trans[1], trans[2], trans[3]])
    return Omega

def Quat2OmegaDot(q, qdot, qddot):
    term1 = 2 * mul_quat(qddot, inverse_quat(q))
    term2 = 2 * mul_quat(qdot, inverse_quat(qdot))  # 추가 항
    trans = term1 + term2
    return trans[1:4]

def SVD_DLS_inverse(J, lambda_factor=1e-2):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    
    S_damped = np.diag([s / (s**2 + lambda_factor**2) for s in S])
    
    J_damped_inverse = Vt.T @ S_damped @ U.T
    
    return J_damped_inverse

def DLS_inverse(J, init_lambda=0.01, threshold=1):
    cond_number = np.linalg.cond(J)
    
    if cond_number < threshold:
        lambda_factor = 0
    else:
        lambda_factor = init_lambda * (cond_number - threshold)

    J_inverse = np.linalg.inv(J.T @ J + lambda_factor**2 * np.eye(J.shape[1])) @ J.T
    
    return J_inverse

def dls_right_pinv(J2d, init_lambda=1e-2, cond_thresh=100.0):
    """
    J2d: (m, n) with m >= n 권장. 반환: J# (n, m)
    Right pseudoinverse via (JᵀJ + λ²I)⁻¹ Jᵀ
    """
    J2d = np.asarray(J2d, dtype=np.float32)
    if J2d.ndim != 2:
        raise ValueError(f"J must be 2D, got shape {J2d.shape}")

    # cond 기반 λ 산정(선택)
    cond = np.linalg.cond(J2d)
    lam = 0.0 if cond < cond_thresh else init_lambda * (cond / cond_thresh - 1.0)

    JTJ = J2d.T @ J2d                     # (n,n)
    if lam > 0.0:
        JTJ = JTJ + (lam * lam) * np.eye(JTJ.shape[0], dtype=J2d.dtype)
    # J# = (JTJ)⁻¹ Jᵀ  → solve로 계산
    J_pinv = np.linalg.solve(JTJ, J2d.T)  # (n,n) \ (n,m) = (n,m)
    return J_pinv

def get_XYdeg(pos):
    x,y,z = pos[0], pos[1], pos[2]
    radian = np.arctan2(y,x)
    deg = radian * 180 / pi

    return deg

def get_XYdeg_error(pos1, pos2):
    deg1 = get_XYdeg(pos1)
    deg2 = get_XYdeg(pos2)

    error = deg2 - deg1

    return error 