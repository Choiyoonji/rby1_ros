import numpy as np
from dataclasses import dataclass, field

@dataclass
class MetaStatus:
    timestamp: float = 0.0

    head_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    right_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    left_offset: np.ndarray = field(default_factory=lambda: np.array([]))

    head_position: np.ndarray = field(default_factory=lambda: np.array([]))
    head_rotation: np.ndarray = field(default_factory=lambda: np.identity(3))
    
    right_arm_position: np.ndarray = field(default_factory=lambda: np.array([]))
    right_arm_rotation: np.ndarray = field(default_factory=lambda: np.identity(3))
    left_arm_position: np.ndarray = field(default_factory=lambda: np.array([]))
    left_arm_rotation: np.ndarray = field(default_factory=lambda: np.identity(3))

    right_hand_position: np.ndarray = field(default_factory=lambda: np.array([]))
    left_hand_position: np.ndarray = field(default_factory=lambda: np.array([]))

    # Flags
    is_connected: bool = False
    is_initialized: bool = False