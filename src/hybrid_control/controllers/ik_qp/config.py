"""Targets and tuning shared by the IK/QP controllers (ik_qp_mujoco and ik_qp)."""

from dataclasses import dataclass

import numpy as np

from hybrid_control.robots.state import Pose


@dataclass(frozen=True)
class PoseTarget:
    """Desired world-frame pose of a named body origin."""

    frame_name: str
    pose: Pose

    def __post_init__(self):
        if not isinstance(self.frame_name, str) or not self.frame_name:
            raise ValueError("frame_name must be a nonempty body name")
        if not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose")


@dataclass(frozen=True)
class IKQPConfig:
    """IK tuning and joint velocity bounds in robot joint order.

    Bounds have shape (n,), in rad/s for hinge joints and m/s for slides.
    Gains are inverse seconds; weights and damping retain the imported QP's
    numerical scaling. Experiments supply their own tuning explicitly.
    """

    velocity_lower: np.ndarray
    velocity_upper: np.ndarray
    kp_pos: float = 150.0
    kp_ori: float = 60.0
    position_weight: float = 1.0
    orientation_weight: float = 1.0
    damping: float = 1e-3

    def __post_init__(self):
        lower = np.array(self.velocity_lower, dtype=float, copy=True)
        upper = np.array(self.velocity_upper, dtype=float, copy=True)
        if lower.ndim != 1 or not lower.size or upper.shape != lower.shape:
            raise ValueError("Velocity bounds must be nonempty vectors of equal shape")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Velocity bounds must be finite")
        if np.any(lower > upper):
            raise ValueError("velocity_lower must not exceed velocity_upper")
        for name, value in (("velocity_lower", lower), ("velocity_upper", upper)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ("kp_pos", "kp_ori", "position_weight", "orientation_weight", "damping"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.position_weight == self.orientation_weight == 0:
            raise ValueError("At least one task weight must be positive")
