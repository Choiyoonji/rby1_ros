import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class MetaStatus:
    timestamp: float = 0.0

    head_pos_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    head_rot_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    right_pos_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    right_rot_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    left_pos_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    left_rot_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    
    torso_anchor_position: np.ndarray = field(default_factory=lambda: np.array([]))

    head_position: np.ndarray = field(default_factory=lambda: np.array([]))
    head_rotation: np.ndarray = field(default_factory=lambda: np.identity(3))
    head_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))

    torso_position: np.ndarray = field(default_factory=lambda: np.array([]))
    torso_rotation: np.ndarray = field(default_factory=lambda: np.identity(3))
    torso_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))
    
    right_arm_position: np.ndarray = field(default_factory=lambda: np.array([]))
    right_arm_rotation: np.ndarray = field(default_factory=lambda: np.identity(3))
    right_arm_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))

    left_arm_position: np.ndarray = field(default_factory=lambda: np.array([]))
    left_arm_rotation: np.ndarray = field(default_factory=lambda: np.identity(3))
    left_arm_quaternion: np.ndarray = field(default_factory=lambda: np.array([]))

    right_hand_position: float = 0.0
    left_hand_position: float = 0.0
    # JWL2000 - add right hand data
    right_hand_EE_position: np.ndarray = field(default_factory=lambda: np.array([]))
    right_hand_lnk_position: np.ndarray = field(default_factory=lambda: np.array([]))
    right_hand_lnk_rotation: np.ndarray = field(default_factory=lambda: np.array([]))
    # JWL2000 - add left hand data
    left_hand_EE_position: np.ndarray = field(default_factory=lambda: np.array([]))
    left_hand_lnk_position: np.ndarray = field(default_factory=lambda: np.array([]))
    left_hand_lnk_rotation: np.ndarray = field(default_factory=lambda: np.array([]))

    # Flags
    is_connected: bool = False
    is_initialized: bool = False