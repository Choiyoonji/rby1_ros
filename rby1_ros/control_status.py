import numpy as np
from dataclasses import dataclass, field

@dataclass
class ControlStatus:
    timestamp: float = 0.0

    is_controller_connected: bool = False

    ready: bool = False
    move: bool = False
    stop: bool = False
    estop: bool = False

    control_mode: str = "component"  # Options: "whole_body"

    desired_right_ee_position: dict = field(default_factory=dict)
    desired_left_ee_position: dict = field(default_factory=dict)
    desired_head_ee_position: dict = field(default_factory=dict)

    desired_right_ee_T: np.ndarray = field(default_factory=lambda: np.identity(4))
    desired_left_ee_T: np.ndarray = field(default_factory=lambda: np.identity(4))
    desired_head_ee_T: np.ndarray = field(default_factory=lambda: np.identity(4))

    desired_joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))