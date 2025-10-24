import numpy as np
from dataclasses import dataclass, field

@dataclass
class RBY1Status:
    timestamp: float = 0.0

    is_robot_connected: bool = False

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

    mobile_linear_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    mobile_angular_velocity: float = 0.0

    center_of_mass: np.ndarray = field(default_factory=lambda: np.array([]))

    right_arm_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_arm_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    torso_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))

    # Flags
    is_initialized: bool = False
    is_stopped: bool = True
    is_torso_following: bool = False
    is_right_following: bool = False
    is_left_following: bool = False