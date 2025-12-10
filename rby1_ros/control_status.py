import numpy as np
from dataclasses import dataclass, field

@dataclass
class ControlStatus:
    timestamp: float = 0.0

    is_controller_connected: bool = False
    is_active: bool = False
    
    is_button_right_pressed: bool = False
    is_button_left_pressed: bool = False

    ready: bool = False
    move: bool = False
    stop: bool = False
    estop: bool = False

    control_mode: str = "component"  # Options: "whole_body"

    desired_right_ee_pose: dict = field(default_factory=dict)
    desired_left_ee_pose: dict = field(default_factory=dict)
    desired_torso_ee_pose: dict = field(default_factory=dict)

    desired_right_ee_T: np.ndarray = field(default_factory=lambda: np.identity(4))
    desired_left_ee_T: np.ndarray = field(default_factory=lambda: np.identity(4))
    desired_torso_ee_T: np.ndarray = field(default_factory=lambda: np.identity(4))

    desired_joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))

    desired_right_gripper_position: float = 1.0
    desired_left_gripper_position: float = 1.0