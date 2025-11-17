import numpy as np
from dataclasses import dataclass, field

@dataclass
class RBY1Status:
    timestamp: float = 0.0

    is_robot_connected: bool = False

    q_limits_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    q_limits_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    qdot_limits_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    qddot_limits_upper: np.ndarray = field(default_factory=lambda: np.array([]))

    dt : float = 1.0

    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_currents: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_torques: np.ndarray = field(default_factory=lambda: np.array([]))

    right_force_sensor: np.ndarray = field(default_factory=lambda: np.array([]))
    left_force_sensor: np.ndarray = field(default_factory=lambda: np.array([]))

    right_torque_sensor: np.ndarray = field(default_factory=lambda: np.array([]))
    left_torque_sensor: np.ndarray = field(default_factory=lambda: np.array([]))

    right_ee_position: np.ndarray = field(default_factory=lambda: np.array([]))
    left_ee_position: np.ndarray = field(default_factory=lambda: np.array([]))
    torso_ee_position: np.ndarray = field(default_factory=lambda: np.array([]))

    right_gripper_position: float = 1.0
    left_gripper_position: float = 1.0

    mobile_linear_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    mobile_angular_velocity: float = 0.0

    center_of_mass: np.ndarray = field(default_factory=lambda: np.array([]))

    right_arm_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_arm_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    torso_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    
    right_arm_locked_angle: np.ndarray = field(default_factory=lambda: np.array([]))

    # Flags
    is_initialized: bool = False
    is_stopped: bool = True
    is_torso_following: bool = False
    is_right_following: bool = False
    is_left_following: bool = False