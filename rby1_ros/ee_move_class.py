from trajectory import Trajectory
import numpy as np
from typing import List, Tuple
import time
from matplotlib import pyplot as plt

# ============Utils===============
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

def Quat2Rot(quat):
    q_w, q_x, q_y, q_z = quat[0], quat[1], quat[2], quat[3]

    # 회전 행렬 계산
    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
        [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
        [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
    ])
    return R
# ================================

class Move_ee:
    def __init__(self, Hz=100, duration=2.0, dist_step=0.01):
        self.Hz = Hz
        self.duration = duration
        self.total_step = Hz * duration
        
        self.dist_step = dist_step

        self.trajectory_ee = None
        self.last_desired_ee_pos = None
        self.plan_desired_ee_pos = []
        
        self.is_done = False
    
    def plan_move_ee(self, start_ee_pos:List|np.ndarray, delta_ee_pos:List|np.ndarray):
        self.trajectory_ee = Trajectory(0.0, self.duration)
        init_state = np.array(start_ee_pos)
        final_state = init_state + np.array(delta_ee_pos)
        self.trajectory_ee.get_coeff(init_state, final_state)
        
        prev_pos = start_ee_pos.copy()
        for step in range(1, int(self.total_step) + 1):
            current_time = step / self.Hz
            pos, vel, acc = self.trajectory_ee.calculate_pva(current_time)
            if np.linalg.norm(pos - prev_pos) > self.dist_step:
                pos = prev_pos + (pos - prev_pos) / np.linalg.norm(pos - prev_pos) * self.dist_step
            prev_pos = pos.copy()
            self.plan_desired_ee_pos.append(pos)
        
        self.is_done = False
        return self.plan_desired_ee_pos
    
    def plan_move_ee_by_distance(self, start_ee_pos: List | np.ndarray, 
                                delta_ee_pos: List | np.ndarray):
        if self.dist_step is None:
            raise ValueError("dist_step must be set for distance-based planning.")
        
        init_state = np.array(start_ee_pos, dtype=float)
        delta = np.array(delta_ee_pos, dtype=float)

        total_dist = np.linalg.norm(delta)

        self.plan_desired_ee_pos = []

        if total_dist < 1e-9:
            self.plan_desired_ee_pos.append(init_state.copy())
            self.is_done = False
            return self.plan_desired_ee_pos


        direction = delta / total_dist
        n_steps = int(np.floor(total_dist / self.dist_step))

        for k in range(1, n_steps + 1):
            s = k * self.dist_step
            pos = init_state + direction * s
            self.plan_desired_ee_pos.append(pos)

        final_state = init_state + delta
        if len(self.plan_desired_ee_pos) == 0 or not np.allclose(self.plan_desired_ee_pos[-1], final_state):
            self.plan_desired_ee_pos.append(final_state)

        self.is_done = False
        return self.plan_desired_ee_pos
            
    def move_ee(self) -> Tuple[np.ndarray, bool]:
        if len(self.plan_desired_ee_pos) != 0:
            desired_ee_pos = self.plan_desired_ee_pos.pop(0)
            self.last_desired_ee_pos = desired_ee_pos
            return desired_ee_pos, self.is_done
        else:
            self.is_done = True
            return self.last_desired_ee_pos, self.is_done
    
class Rotate_ee:
    def __init__(self, Hz=100, duration=2.0, degree_step=10.0):
        self.Hz = Hz
        self.duration = duration
        self.total_step = Hz * duration
        self.degree_step = degree_step

        self.trajectory_ee_quat = None
        self.last_desired_ee_quat = None
        self.plan_desired_ee_quat = []
        
        self.is_done = False
        
    def delta_rot2delta_quat(self, axis:str, degree:float) -> np.ndarray:
        rad = np.deg2rad(degree)
        half_rad = rad / 2.0
        if axis == 'x':
            delta_quat = np.array([np.cos(half_rad), np.sin(half_rad), 0.0, 0.0])
        elif axis == 'y':
            delta_quat = np.array([np.cos(half_rad), 0.0, np.sin(half_rad), 0.0])
        elif axis == 'z':
            delta_quat = np.array([np.cos(half_rad), 0.0, 0.0, np.sin(half_rad)])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        return delta_quat
    
    def plan_rotate_ee(self, start_ee_quat:List|np.ndarray, axis:str, degree:float, type:str):
        self.trajectory_ee_quat = Trajectory(0.0, self.duration)
        init_state = np.array(start_ee_quat)
        delta_ee_quat = self.delta_rot2delta_quat(axis, degree)
        
        if type == 'local':
            final_state = mul_quat(init_state, np.array(delta_ee_quat))
        elif type == 'global':
            final_state = mul_quat(np.array(delta_ee_quat), init_state)
        else:
            raise ValueError("Type must be 'local' or 'global'")
         
        self.trajectory_ee_quat.get_coeff_quat(init_state, final_state)
        
        prev_quat = start_ee_quat.copy()
        for step in range(1, int(self.total_step) + 1):
            current_time = step / self.Hz
            quat, quat_vel, quat_acc = self.trajectory_ee_quat.calculate_pva_quat(current_time)
            quat_rel = quat_diff(prev_quat, quat)
            theta_diff = 2.0 * np.arccos(np.clip(quat_rel[0], -1.0, 1.0))
            if theta_diff > np.deg2rad(self.degree_step):
                axis = quat_rel[1:4] / (np.linalg.norm(quat_rel[1:4]) + 1e-7)
                limited_theta = np.deg2rad(self.degree_step)
                limited_quat_diff = np.array([np.cos(limited_theta / 2.0),
                                             axis[0] * np.sin(limited_theta / 2.0),
                                             axis[1] * np.sin(limited_theta / 2.0),
                                             axis[2] * np.sin(limited_theta / 2.0)])
                quat = mul_quat(prev_quat, limited_quat_diff)
            prev_quat = quat.copy()
            self.plan_desired_ee_quat.append(quat)
        
        self.is_done = False
        return self.plan_desired_ee_quat
            
    def rotate_ee(self) -> Tuple[np.ndarray, bool]:
        if len(self.plan_desired_ee_quat) != 0:
            desired_ee_quat = self.plan_desired_ee_quat.pop(0)
            self.last_desired_ee_quat = desired_ee_quat
            return desired_ee_quat, self.is_done
        else:
            self.is_done = True
            return self.last_desired_ee_quat, self.is_done


# =========== Test code ==============
def main_pos():
    Hz = 100
    duration = 2.0

    mover = Move_ee(Hz=Hz, duration=duration, dist_step=0.01)

    start_ee_pos = [0.0, 0.0, 0.0]
    delta_ee_pos = [0.1, 0.0, 0.0]
    
    _ = mover.plan_move_ee(start_ee_pos, delta_ee_pos)
    # _ = mover.plan_move_ee_by_distance(start_ee_pos, delta_ee_pos)

    print("Start moving end-effector...")

    time_log = []
    pos_log = []

    step = 0
    while True:
        desired_pos, is_done = mover.move_ee()

        time_log.append(step / Hz)
        pos_log.append(np.array(desired_pos))

        print(f"t = {step/Hz:.3f}s, desired pos: {desired_pos}, is_done: {is_done}")

        if step == 300:
            delta_ee_pos = [0.0, 0.2, 0.0]
            _ = mover.plan_move_ee(mover.last_desired_ee_pos, delta_ee_pos)
            # _ = mover.plan_move_ee_by_distance(mover.last_desired_ee_pos, delta_ee_pos)
        elif step > 601:
            break

        step += 1

        time.sleep(1.0 / Hz)

    pos_log = np.stack(pos_log, axis=0)  # shape: (N, dof)
    time_log = np.array(time_log)

    dof = pos_log.shape[1]

    plt.figure()
    for i in range(dof):
        plt.plot(time_log, pos_log[:, i], label=f'axis {i}')

    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position")
    plt.title("EE trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main_rot():
    Hz = 50
    duration = 3.0

    rot = Rotate_ee(Hz=Hz, duration=duration, degree_step=10.0)

    start_ee_quat = np.array([0.7071, 0.7071, 0.0, 0.0]) # wxyz 순서
    
    axis = 'z'
    degree = 90.0
    rotate_type = 'local'  # 'global' 로 바꾸면 절대좌표 기준 회전

    rot.plan_rotate_ee(start_ee_quat, axis, degree, rotate_type)

    time_list = []
    quat_list = []

    print("Start rotating...")
    step_idx = 0
    while True:
        desired_quat, is_done = rot.rotate_ee()

        t = step_idx / Hz
        time_list.append(t)
        quat_list.append(desired_quat.copy())
        step_idx += 1

        print(f"t={t:.3f}  quat={desired_quat}  is_done={is_done}")

        if is_done:
            print("Rotation trajectory finished.")
            break

        time.sleep(1.0 / Hz)

    quat_array = np.array(quat_list)  # shape: (N, 4)
    time_array = np.array(time_list)

    plt.figure()
    plt.plot(time_array, quat_array[:, 0], label='w')
    plt.plot(time_array, quat_array[:, 1], label='x')
    plt.plot(time_array, quat_array[:, 2], label='y')
    plt.plot(time_array, quat_array[:, 3], label='z')
    plt.xlabel('time [s]')
    plt.ylabel('quaternion components')
    plt.title('EE quaternion trajectory')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    # main_pos()
    main_rot()