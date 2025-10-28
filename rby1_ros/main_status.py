import numpy as np
from dataclasses import dataclass, field

@dataclass
class MainStatus:
    is_robot_connected: bool = False
    is_robot_in_ready_pose: bool = False
    is_robot_initialized: bool = False
    is_robot_stopped: bool = False

    is_meta_initialized: bool = False
    is_meta_ready: bool = False

    is_controller_connected: bool = False
    is_controller_initialized: bool = False

    current_torso_position: np.ndarray = field(default_factory=lambda: np.array([]))
    current_torso_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))
    current_right_arm_position: np.ndarray = field(default_factory=lambda: np.array([]))
    current_right_arm_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))
    current_left_arm_position: np.ndarray = field(default_factory=lambda: np.array([]))
    current_left_arm_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))

    desired_head_position: np.ndarray = field(default_factory=lambda: np.array([]))
    desired_head_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))
    desired_right_arm_position: np.ndarray = field(default_factory=lambda: np.array([]))
    desired_right_arm_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))
    desired_left_arm_position: np.ndarray = field(default_factory=lambda: np.array([]))
    desired_left_arm_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))

    desired_joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))