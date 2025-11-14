# Author : JWL2000

import numpy as np
from scipy.spatial.transform import Rotation as R
# from matplotlib import pyplot as plt

def mul_quat(q1,q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    mul = np.array([w, x, y, z])
    
    return mul

def conjugation(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def inverse_quat(q):
    q_norm = np.linalg.norm(q)
    q_inverse = conjugation(q) / (q_norm ** 2 + 1e-7)
    return q_inverse

class Trajectory:
    def __init__(self, init_time, time_to_reach_goal):
        self.final_time = time_to_reach_goal
        self.init_time = init_time
        self.excuted_once = False
        self.excuted_quat_once = False
        self.excuted_qpos_once = False
        self.coeff = None
        self.coeff_quat = None
        self.coeff_qpos = None

    def time_mat(self, current_time):
        tmat = np.array([[   current_time**5,    current_time**4,   current_time**3,   current_time**2, current_time**1, 1],
                         [ 5*current_time**4,  4*current_time**3, 3*current_time**2, 2*current_time**1,               1, 0],
                         [20*current_time**3, 12*current_time**2, 6*current_time**1,                 2,               0, 0]])
        
        return tmat
    
    def get_coeff(self, init_state, final_state):
        if self.excuted_once == False:
            self.final_state = final_state
            
            init_mat = self.time_mat(self.init_time)
            final_mat = self.time_mat(self.final_time)
            mat = np.vstack((init_mat, final_mat))
            
            target = np.zeros((6, len(init_state)))
            target[0, :] = init_state
            target[3, :] = self.final_state

            self.coeff = np.linalg.inv(mat) @ target    # 6x3 matrix
            self.excuted_once = True


    def calculate_pva(self, current_time):   # calculate pose, velocity, acceleration       
        if current_time <= self.final_time:
            pva = self.time_mat(current_time) @ self.coeff    # 3x3 matrix
        else:
            pva = self.time_mat(self.final_time) @ self.coeff
        
        pos = pva[0,:]
        vel = pva[1,:]
        acc = pva[2,:]
            
        return pos, vel, acc
    
    def get_coeff_quat(self, init_state, final_state):
        if self.excuted_quat_once == False:
            self.angle_diff = mul_quat(inverse_quat(init_state), final_state)
            self.final_state = final_state

            if np.dot(init_state, self.final_state) < 0:
                self.final_state = -final_state
                self.angle_diff = mul_quat(inverse_quat(init_state), self.final_state)
            
            init_mat = self.time_mat(self.init_time)
            final_mat = self.time_mat(self.final_time)
            mat = np.vstack((init_mat, final_mat))
            target = np.zeros((6, len(init_state)))
            target[0, :] = init_state
            target[3, :] = self.final_state
            
            self.coeff_quat = np.linalg.inv(mat) @ target    # 6x4 matrix
            self.excuted_quat_once = True

    def calculate_pva_quat(self, current_time):   # calculate pose, velocity, acceleration
        if current_time <= self.final_time:
            pva = self.time_mat(current_time) @ self.coeff_quat    # 3x4 matrix 
        else:
            pva = self.time_mat(self.final_time) @ self.coeff_quat
        pos = pva[0,:]
        vel = pva[1,:]
        acc = pva[2,:]
        
        return pos, vel, acc
    
    def get_coeff_qpos(self, init_state, init_vel, final_state):
        if self.excuted_qpos_once == False:
            self.final_state = final_state

            init_mat = self.time_mat(self.init_time)
            final_mat = self.time_mat(self.final_time)
            mat = np.vstack((init_mat, final_mat))
            
            target = np.zeros((6, len(init_state)))
            target[0, :] = init_state
            target[1, :] = init_vel[:]
            target[3, :] = self.final_state
            
            self.coeff_qpos = np.linalg.inv(mat) @ target    # 6x5 matrix
            self.excuted_qpos_once = True
        
    def calculate_pva_qpos(self, current_time):   # calculate pose, velocity, acceleration
        if current_time <= self.final_time:
            pva = self.time_mat(current_time) @ self.coeff_qpos    # 3x4 matrix 
        else:
            pva = self.time_mat(self.final_time) @ self.coeff_qpos

        pos = pva[0,:]
        vel = pva[1,:]
        acc = pva[2,:]
        
        return pos, vel, acc

