from .trajectory import Trajectory
import numpy as np
from typing import List, Tuple, Union
import time
from matplotlib import pyplot as plt
from .utils import mul_quat_xyzw, mul_quat, quat_diff_xyzw, normalize_quat

class Move_ee:
    def __init__(self, Hz=100, duration=2.0, dist_step=0.01):
        self.Hz = Hz
        self.duration = duration
        self.total_step = int(Hz * duration)
        
        self.dist_step = dist_step

        self.trajectory_ee = None
        self.last_desired_ee_pos = None
        self.plan_desired_ee_pos = []
        
        self.is_done = False

        self.v_max_lin = 0.1    # [m/s] 최대 선형 속도
        self.a_max_lin = 1.0    # [m/s^2] 최대 선형 가속도
        self.min_duration = 0.1 # [s] 최소 도달 시간
        
        self.SPLINE_VEL_FACTOR = 1.875
        self.SPLINE_ACC_FACTOR = 5.774

        # TODO: get_bounding_box 코드로 범위 확인 후 설정
        self.lower_bound = {
            'left': np.array([0.4142, -0.0164, 0.8]),
            'right': np.array([0.4142,-0.5905, 0.8])
        }
        self.upper_bound = {
            'left': np.array([0.5100, 0.4319, 1.54]),
            'right': np.array([0.5369 , -0.1161, 1.54])
        }
    #         X 범위: 0.3211 ~ 0.4151 	(변화량: 0.0940)
    # Y 범위: -0.5661 ~ -0.1043 	(변화량: 0.4618)
    # Z 범위: 0.8279 ~ 0.9989 	(변화량: 0.1710)
    # X 범위: 0.4142 ~ 0.5100 	(변화량: 0.0958)
    # Y 범위: -0.0164 ~ 0.4319 	(변화량: 0.4483)
    # Z 범위: 0.8300 ~ 1.0463 	(변화량: 0.2163)
    # X 범위: 0.3672 ~ 0.5815 	(변화량: 0.2143)d
    # Y 범위: -0.3357 ~ -0.1161 	(변화량: 0.2195)
    # Z 범위: 0.8208 ~ 0.9848 	(변화량: 0.1640)



    def calculate_required_duration(self, delta_ee_pos: np.ndarray) -> float:
        """이동 거리에 따른 최소 시간 계산"""
        distance = np.linalg.norm(delta_ee_pos)
        
        if distance < 1e-6:
            return self.min_duration

        t_vel = (distance * self.SPLINE_VEL_FACTOR) / self.v_max_lin
        t_acc = np.sqrt((distance * self.SPLINE_ACC_FACTOR) / self.a_max_lin)
        
        req_time = max(t_vel, t_acc)
        return max(req_time, self.min_duration)
    
    def clip_to_bounds(self, arm, position: np.ndarray) -> np.ndarray:
        """주어진 위치를 설정된 경계 내로 클리핑"""
        return np.minimum(np.maximum(position, self.lower_bound[arm]), self.upper_bound[arm])

    def plan_move_ee(self, arm, start_ee_pos: Union[List, np.ndarray], delta_ee_pos: Union[List, np.ndarray], calc_duration: bool = False):
        init_state = np.array(start_ee_pos, dtype=float)
        delta_vec = np.array(delta_ee_pos, dtype=float)
        final_state = init_state + delta_vec

        if calc_duration:
            # 안전 계수 1.1배 적용
            duration = self.calculate_required_duration(delta_vec) * 1.1
        else:
            duration = self.duration
        
        self.total_step = int(duration * self.Hz)
        
        print(f"Planning EE move: Dist={np.linalg.norm(delta_vec):.3f}m, Time={duration:.3f}s, Steps={self.total_step}")

        self.trajectory_ee = Trajectory(0.0, duration)
        self.trajectory_ee.get_coeff(init_state, final_state)
        
        self.plan_desired_ee_pos = []
        prev_pos = init_state.copy()
        
        for step in range(1, self.total_step + 1):
            current_time = step / self.Hz
            pos, vel, acc = self.trajectory_ee.calculate_pva(current_time)
            
            dist = np.linalg.norm(pos - prev_pos)
            if dist > self.dist_step:
                pos = prev_pos + (pos - prev_pos) / dist * self.dist_step

            pos = self.clip_to_bounds(arm, pos)

            prev_pos = pos.copy()
            self.plan_desired_ee_pos.append(pos)
        
        self.is_done = False
        return self.plan_desired_ee_pos
    
    def plan_move_ee_by_distance(self, start_ee_pos: Union[List, np.ndarray], delta_ee_pos: Union[List, np.ndarray]):
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
        self.total_step = int(Hz * duration)
        self.degree_step = degree_step

        self.trajectory_ee_quat = None
        self.last_desired_ee_quat = None
        self.plan_desired_ee_quat = []
        
        self.is_done = False

        self.v_max_ang = 0.5   # [rad/s] (약 85 deg/s)
        self.a_max_ang = 3.0   # [rad/s^2]
        self.min_duration = 0.1
        
        self.SPLINE_VEL_FACTOR = 1.875
        self.SPLINE_ACC_FACTOR = 5.774
        
    def delta_rot2delta_quat(self, axis:str, degree:float) -> np.ndarray:
        rad = np.deg2rad(degree)
        half_rad = rad / 2.0
        if axis == 'x':
            delta_quat = np.array([np.sin(half_rad), 0.0, 0.0, np.cos(half_rad)])
        elif axis == 'y':
            delta_quat = np.array([0.0, np.sin(half_rad), 0.0, np.cos(half_rad)])
        elif axis == 'z':
            delta_quat = np.array([0.0, 0.0, np.sin(half_rad), np.cos(half_rad)])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        return delta_quat

    def calculate_required_duration(self, degree: float) -> float:
        theta_rad = np.abs(np.deg2rad(degree))
        
        if theta_rad < 1e-6:
            return self.min_duration

        t_vel = (theta_rad * self.SPLINE_VEL_FACTOR) / self.v_max_ang
        t_acc = np.sqrt((theta_rad * self.SPLINE_ACC_FACTOR) / self.a_max_ang)
        
        req_time = max(t_vel, t_acc)
        return max(req_time, self.min_duration)
    
    def plan_rotate_ee(self, start_ee_quat: Union[List, np.ndarray], axis: str, degree: float, type: str, calc_duration: bool = False):
        if calc_duration:
            duration = self.calculate_required_duration(degree) * 1.1
        else:
            duration = self.duration
            
        self.total_step = int(duration * self.Hz)

        print(f"Planning EE Rotation: Deg={degree}, Time={duration:.3f}s, Steps={self.total_step}")

        self.trajectory_ee_quat = Trajectory(0.0, duration)
        init_state = np.array(start_ee_quat)
        delta_ee_quat = self.delta_rot2delta_quat(axis, degree)
        
        if type == 'local':
            final_state = mul_quat_xyzw(init_state, np.array(delta_ee_quat))
        elif type == 'global':
            final_state = mul_quat_xyzw(np.array(delta_ee_quat), init_state)
        else:
            raise ValueError("Type must be 'local' or 'global'")
         
        self.trajectory_ee_quat.get_coeff_quat(init_state, final_state)
        
        self.plan_desired_ee_quat = []
        prev_quat = init_state.copy()
        
        for step in range(1, self.total_step + 1):
            current_time = step / self.Hz
            quat, quat_vel, quat_acc = self.trajectory_ee_quat.calculate_pva_quat(current_time)
            quat_rel = normalize_quat(quat_diff_xyzw(prev_quat, quat))
            w_val = np.clip(quat_rel[3], -1.0, 1.0)
            theta_diff = 2.0 * np.arccos(w_val)
            
            if theta_diff > np.deg2rad(self.degree_step):
                vec_part = quat_rel[0:3]
                
                limited_theta = np.deg2rad(self.degree_step)
                half_limited = limited_theta / 2.0
                
                limited_quat_diff = np.array([
                    vec_part[0] * np.sin(half_limited),
                    vec_part[1] * np.sin(half_limited),
                    vec_part[2] * np.sin(half_limited),
                    np.cos(half_limited)
                ])
                quat = mul_quat_xyzw(prev_quat, limited_quat_diff)
            
            quat = normalize_quat(quat)
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
    default_duration = 1.0 # 기본값은 짧게 설정

    # 1. 초기 설정
    mover = Move_ee(Hz=Hz, duration=default_duration, dist_step=0.01)

    # 테스트를 위해 로봇의 속도 제한을 일부러 낮게 설정 (0.1m/s)
    mover.v_max_lin = 0.1 
    mover.a_max_lin = 0.5
    print(f"--- [Position Test] Limits: Max Vel={mover.v_max_lin}m/s, Max Acc={mover.a_max_lin}m/s^2 ---")

    start_ee_pos = [0.0, 0.0, 0.0]
    
    # 2. 첫 번째 이동: 0.3m 이동 (속도제한 0.1m/s 이므로 최소 3초 이상 필요)
    # calc_duration=True 옵션을 켜서 시간이 자동으로 늘어나는지 확인
    delta_ee_pos_1 = [0.3, 0.0, 0.0]
    print("\n>>> Move 1: Distance 0.3m with calc_duration=True")
    mover.plan_move_ee(start_ee_pos, delta_ee_pos_1, calc_duration=True)

    print("Start moving end-effector...")

    time_log = []
    pos_log = []

    step = 0
    
    # 시뮬레이션 루프
    while True:
        desired_pos, is_done = mover.move_ee()

        current_time = step / Hz
        time_log.append(current_time)
        pos_log.append(np.array(desired_pos))

        # 로그 출력을 줄이기 위해 0.5초마다 출력
        if step % 50 == 0:
            print(f"t = {current_time:.2f}s, pos: {np.round(desired_pos, 3)}, done: {is_done}")

        # 3. 중간 개입 테스트 (첫 번째 이동이 끝나갈 때쯤 새로운 명령)
        # 이번에는 짧은 거리를 이동하므로 시간이 짧게 재계산되어야 함
        if is_done and step > 10: 
            # 한 번만 실행하기 위한 플래그 처리 대신 간단히 break 후 두 번째 루프 혹은 여기서 재계획
            print("\n>>> Move 1 Finished. Planning Move 2 immediately...")
            delta_ee_pos_2 = [0.0, 0.05, 0.0] # 5cm 이동 (짧음)
            
            # 현재 위치에서 시작
            mover.plan_move_ee(mover.last_desired_ee_pos, delta_ee_pos_2, calc_duration=True)
            
            # 무한 루프 방지를 위해 내부 플래그나 카운터 사용 (테스트용 단순화)
            if step > 1000: # 안전장치
                break
        
        # 전체 종료 조건 (넉넉하게 잡음)
        if step > 800 and is_done:
            print("All trajectories finished.")
            break

        step += 1
        # time.sleep(1.0 / Hz) # 그래프 확인을 위해 sleep은 주석처리 가능

    # 그래프 그리기
    pos_log = np.stack(pos_log, axis=0)
    time_log = np.array(time_log)
    dof = pos_log.shape[1]

    plt.figure(figsize=(10, 6))
    labels = ['x', 'y', 'z']
    for i in range(dof):
        plt.plot(time_log, pos_log[:, i], label=f'{labels[i]}-axis')

    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.title("EE Position Trajectory (Check duration adjustment)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main_rot():
    Hz = 50
    default_duration = 1.0

    # 1. 초기 설정
    rot = Rotate_ee(Hz=Hz, duration=default_duration, degree_step=5.0) # step을 좀 더 부드럽게
    
    # 테스트를 위해 각속도 제한 설정 (약 45도/초 = 0.78 rad/s)
    rot.v_max_ang = 0.785 
    print(f"--- [Rotation Test] Limits: Max Vel={rot.v_max_ang:.3f} rad/s ---")

    start_ee_quat = np.array([0.0, 0.0, 0.0, 1.0]) # wxyz (Identity)
    
    # 2. 회전 명령: 90도 회전
    # 제한 속도가 45도/초 이므로, 90도 회전에는 스플라인 고려 시 약 3~4초가 필요할 것임.
    axis = 'z'
    degree = 90.0
    rotate_type = 'local'

    print(f"\n>>> Rotate 1: {degree} deg around {axis} with calc_duration=True")
    rot.plan_rotate_ee(start_ee_quat, axis, degree, rotate_type, calc_duration=True)

    time_list = []
    quat_list = [] # w, x, y, z

    print("Start rotating...")
    step_idx = 0
    
    while True:
        desired_quat, is_done = rot.rotate_ee()

        t = step_idx / Hz
        time_list.append(t)
        quat_list.append(desired_quat.copy())
        
        if step_idx % 25 == 0:
             print(f"t={t:.2f}s, quat={np.round(desired_quat, 3)}")

        if is_done:
            print(f"Rotation finished at t={t:.2f}s")
            
            # 연속 동작 테스트: 반대 방향으로 -45도 빠르게 회전
            if t < 10.0 and step_idx > 10: # 중복 실행 방지용 조건
                print("\n>>> Rotate 2: -45 deg (Small move)")
                rot.plan_rotate_ee(rot.last_desired_ee_quat, 'y', -45.0, 'local', calc_duration=True)
                # 루프가 끝나지 않도록 조건문 통과하게 둠
            else:
                break
        
        step_idx += 1
        # time.sleep(1.0 / Hz)

    quat_array = np.array(quat_list)
    time_array = np.array(time_list)

    plt.figure(figsize=(10, 6))
    plt.plot(time_array, quat_array[:, 3], label='w (cos)', linewidth=2)
    plt.plot(time_array, quat_array[:, 0], label='x', linestyle='--')
    plt.plot(time_array, quat_array[:, 1], label='y', linestyle='--')
    plt.plot(time_array, quat_array[:, 2], label='z (sin)', linestyle='--')
    
    plt.xlabel('time [s]')
    plt.ylabel('Quaternion components')
    plt.title('EE Quaternion Trajectory (Check Smoothness & Duration)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 하나씩 주석을 풀고 테스트 해보세요
    # main_pos()
    main_rot()