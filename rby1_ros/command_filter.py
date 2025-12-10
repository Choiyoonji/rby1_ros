import numpy as np
from .utils import *


class LinearCommandFilter:
    def __init__(self, dt, omega=10.0, zeta=1.0, 
                 x_bounds=None, y_bounds=None, z_bounds=None):
        """
        Cartesian Position (XYZ) 필터 with optional task space clipping
        
        Args:
            dt: Time step
            omega: Natural frequency (rad/s)
            zeta: Damping ratio (1.0 for critically damped)
            x_bounds: Tuple (min, max) for X position clipping, or None to disable
            y_bounds: Tuple (min, max) for Y position clipping, or None to disable
            z_bounds: Tuple (min, max) for Z position clipping, or None to disable
        """
        self.dt = dt
        self.omega = omega
        self.zeta = zeta
        
        # Task space bounds for clipping
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        
        # 상태 변수: 위치(3), 속도(3), 가속도(3)
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        
        self.initialized = False

    def set_workspace_bounds(self, x_bounds=None, y_bounds=None, z_bounds=None):
        """
        Set workspace bounds for task space clipping
        
        Args:
            x_bounds: Tuple (min, max) or None
            y_bounds: Tuple (min, max) or None
            z_bounds: Tuple (min, max) or None
        """
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds

    def _clip_position(self, pos):
        """
        Clip position to workspace bounds
        
        Args:
            pos: Position array [x, y, z]
            
        Returns:
            Clipped position array
        """
        clipped_pos = pos.copy()
        
        if self.x_bounds is not None:
            clipped_pos[0] = np.clip(clipped_pos[0], self.x_bounds[0], self.x_bounds[1])
        if self.y_bounds is not None:
            clipped_pos[1] = np.clip(clipped_pos[1], self.y_bounds[0], self.y_bounds[1])
        if self.z_bounds is not None:
            clipped_pos[2] = np.clip(clipped_pos[2], self.z_bounds[0], self.z_bounds[1])
            
        return clipped_pos

    def update(self, target_pos):
        """
        target_pos: [x, y, z] numpy array
        
        Returns:
            pos: Clipped and filtered position
            vel: Velocity
            acc: Acceleration
        """
        target_pos = np.array(target_pos, dtype=float)
        if not self.initialized:
            self.pos = self._clip_position(target_pos)
            self.initialized = True
            
        # 1. 오차 계산 (단순 벡터 차이)
        error = target_pos - self.pos
        
        # 2. 2차 시스템 역학 (F = ma => a = F/m)
        # 가속도 = k*오차 - c*속도
        self.acc = (self.omega ** 2) * error - (2 * self.zeta * self.omega) * self.vel
        
        # 3. 적분 (Euler Integration)
        self.vel += self.acc * self.dt
        self.pos += self.vel * self.dt
        
        # 4. Apply workspace clipping to prevent out-of-bounds motion
        self.pos = self._clip_position(self.pos)
        
        return self.pos, self.vel, self.acc


class AngularCommandFilter:
    def __init__(self, dt, omega=10.0, zeta=1.0):
        self.dt = dt
        self.omega = omega
        self.zeta = zeta
        
        self.quat = np.array([1.0, 0.0, 0.0, 0.0]) # [w, x, y, z]
        self.ang_vel = np.zeros(3) # frame 설정에 따라 Local 또는 Global 각속도
        self.ang_acc = np.zeros(3)
        
        self.initialized = False

    def update(self, target_quat, frame='local'):
        """
        frame: 'local' (Body Frame) 또는 'global' (World Frame)
        """
        # Edge Case 방어
        target_quat = np.array(target_quat, dtype=float)
        norm = np.linalg.norm(target_quat)
        if norm < 1e-6:
            return self.quat, self.ang_vel, self.ang_acc
        target_quat /= norm 

        if not self.initialized:
            self.quat = target_quat.copy()
            self.initialized = True
            return self.quat, self.ang_vel, self.ang_acc

        q_curr_inv = inverse_quat(self.quat)
        
        # ---------------------------------------------------------
        # 1. 쿼터니언 오차 계산 (Error Calculation)
        # ---------------------------------------------------------
        if frame == 'local':
            # Local Error: q_diff = q_curr^-1 * q_target
            # (현재 자세에서 목표 자세로 가기 위한 '내 몸 기준' 회전량)
            q_diff = mul_quat(q_curr_inv, target_quat)
        elif frame == 'global':
            # Global Error: q_diff = q_target * q_curr^-1
            # (현재 자세에서 목표 자세로 가기 위한 '월드 축 기준' 회전량)
            q_diff = mul_quat(target_quat, q_curr_inv)
        else:
            raise ValueError("frame must be 'local' or 'global'")
            
        # 오차 정규화 및 Double Cover 처리
        q_diff /= (np.linalg.norm(q_diff) + 1e-9)
        if q_diff[0] < 0:
            q_diff = -q_diff

        # 2. 오차를 회전 벡터로 변환
        rot_error_vec = quat_to_rotvec(q_diff)

        # 3. 2차 시스템 역학 (선택한 frame 기준의 각가속도/각속도가 계산됨)
        self.ang_acc = (self.omega ** 2) * rot_error_vec - (2 * self.zeta * self.omega) * self.ang_vel
        
        # 4. 적분 (각속도 업데이트)
        self.ang_vel += self.ang_acc * self.dt
        
        # ---------------------------------------------------------
        # 5. 자세 업데이트 (Integration)
        # ---------------------------------------------------------
        delta_quat = rotvec_to_quat(self.ang_vel * self.dt)

        if frame == 'local':
            # Local Update: q_new = q_old * delta (Post-multiply)
            self.quat = mul_quat(self.quat, delta_quat)
        else:
            # Global Update: q_new = delta * q_old (Pre-multiply)
            self.quat = mul_quat(delta_quat, self.quat)

        self.quat /= np.linalg.norm(self.quat)
        
        return self.quat, self.ang_vel, self.ang_acc
    

class PoseCommandFilter:
    def __init__(self, dt, omega=10.0, zeta=1.0):
        self.pos_filter = LinearCommandFilter(dt, omega, zeta)
        self.rot_filter = AngularCommandFilter(dt, omega, zeta)
        
    def update(self, target_pos, target_quat, rot_frame='local'):
        """
        rot_frame: 'local' 또는 'global' (기본값 'local')
        """
        smooth_pos, lin_vel, lin_acc = self.pos_filter.update(target_pos)
        
        # frame 옵션 전달
        smooth_quat, ang_vel, ang_acc = self.rot_filter.update(target_quat, frame=rot_frame)
        
        return smooth_pos, smooth_quat, lin_vel, ang_vel


class CommandFilter:
    def __init__(self, num_joints, dt, omega=10.0, zeta=1.0):
        """
        omega: 반응 속도 (클수록 빠름, 보통 5~15 사이)
        zeta: 감쇠비 (1.0이면 오버슈트 없이 부드럽게 도달)
        """
        self.num_joints = num_joints
        self.dt = dt
        self.omega = omega
        self.zeta = zeta
        
        # 필터 내부 상태 (현재의 추정 위치/속도)
        self.q = np.zeros(num_joints)
        self.qd = np.zeros(num_joints)
        self.qdd = np.zeros(num_joints)
        
        self.initialized = False

    def update(self, q_input):
        """
        q_input: 텔레오퍼레이션에서 들어온 날것의 목표 위치
        return: 부드러워진 q, qd, qdd
        """
        q_input = np.asarray(q_input, dtype=float) 
        if not self.initialized:
            self.q = q_input.copy()
            self.initialized = True
            
        # 2차 시스템 역학 (가상의 스프링-댐퍼 모델)
        # qdd = w^2 * (target - current) - 2*zeta*w * current_vel
        error = q_input - self.q
        self.qdd = (self.omega ** 2) * error - (2 * self.zeta * self.omega) * self.qd
        
        # 적분 (Euler Integration)
        self.qd += self.qdd * self.dt
        self.q += self.qd * self.dt
        
        return self.q, self.qd, self.qdd