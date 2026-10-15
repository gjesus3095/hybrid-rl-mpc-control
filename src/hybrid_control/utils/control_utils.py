"""Controller data contracts; poses and twists use world coordinates and xyzw quaternions."""

from dataclasses import dataclass, field

import numpy as np

from hybrid_control.robots.state import Pose
from hybrid_control.utils.math_helpers import finite_vector
from hybrid_control.utils.manipulability import ROWS as MANIPULABILITY_ROWS


@dataclass(frozen=True)
class CartesianSample:
    """Reference pose with world twist velocity (m/s, rad/s) and acceleration (m/s², rad/s²)."""

    pose: Pose
    velocity: np.ndarray
    acceleration: np.ndarray



@dataclass(frozen=True)
class TaskSpaceTarget:
    """Desired world-frame pose, twist and twist acceleration of a named body origin.

    ``velocity`` (m/s, rad/s) and ``acceleration`` (m/s², rad/s²) have shape (6,),
    ordered [linear, angular] like Jacobian rows. Both default to zero (regulation).
    Impedance uses all fields; the IK controllers track only the pose.
    """

    frame_name: str
    pose: Pose
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def __post_init__(self):
        if not isinstance(self.frame_name, str) or not self.frame_name:
            raise ValueError("frame_name must be a nonempty body name")
        if not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose")
        object.__setattr__(self, "velocity", finite_vector(self.velocity, 6, "velocity"))
        object.__setattr__(self, "acceleration", finite_vector(self.acceleration, 6, "acceleration"))

    @classmethod
    def from_sample(cls, frame_name: str, sample: CartesianSample) -> "TaskSpaceTarget":
        """Build a target from a trajectory sample."""
        return cls(frame_name, sample.pose, sample.velocity, sample.acceleration)



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



@dataclass(frozen=True)
class ImpedanceQPConfig:
    """Impedance gains, QP weights and joint acceleration/torque bounds in robot joint order.

    Stiffness gains ``kp_*`` are in 1/s² and damping gains ``kd_*`` in 1/s, acting on
    the task acceleration (unit-mass impedance). ``None`` damping selects critical
    damping 2·sqrt(kp). Acceleration bounds have shape (n,) in rad/s² (m/s² for
    slides). Torque bounds default to the robot's actuator limits and are always
    intersected with them.

    The posture term regularizes ddq towards ``posture_kp (q_posture - q) - posture_kd dq``,
    which fixes the redundant (null-space) motion of a 7-DoF arm. With both posture
    gains zero it reduces to plain Tikhonov damping ``λ_ddq ||ddq||²``. ``posture=None``
    uses the joint configuration seen by the first ``compute`` after ``reset``.

    Singularity avoidance (all disabled by default):

    - ``manipulability_gain`` adds ``k_w ∇w`` to the regularization reference, so the
      redundant motion climbs the manipulability of the chosen Jacobian ``manipulability_rows``
      ("position", "orientation" or "full"). This needs the selected joints to form one
      serial chain of revolute joints, ordered base to tip.
    - ``manipulability_min > 0`` adds the barrier constraint keeping w >= w_min, with
      convergence rate ``manipulability_barrier_rate`` (1/s). A slack penalized by
      ``manipulability_slack_weight`` keeps the QP feasible when the floor cannot be met.
    - ``joint_limit_margin`` (rad) adds barriers keeping each joint that far inside its
      position limits, with rate ``joint_limit_rate`` (1/s), by tightening the ddq bounds.

    Null-space posture torque (disabled when ``nullspace_posture`` is None): after the QP,
    ``N (nullspace_stiffness (q_null - q) - nullspace_damping dq)`` is added to the torque,
    with N the dynamically consistent null-space projector of the task Jacobian. Stiffness
    (N·m/rad) and damping (N·m·s/rad) are scalars or (n,) vectors. Set ``posture_kp`` and
    ``posture_kd`` to zero when using it, so the QP does not fight the projected torque.
    """

    acceleration_lower: np.ndarray
    acceleration_upper: np.ndarray
    torque_lower: np.ndarray | None = None
    torque_upper: np.ndarray | None = None
    kp_pos: float = 400.0
    kd_pos: float | None = None
    kp_ori: float = 200.0
    kd_ori: float | None = None
    position_weight: float = 1.0
    orientation_weight: float = 1.0
    acceleration_regularization: float = 1e-4
    torque_regularization: float = 1e-6
    posture_kp: float = 10.0
    posture_kd: float | None = None
    posture: np.ndarray | None = None
    manipulability_gain: float = 0.0
    manipulability_min: float = 0.0
    manipulability_rows: str = "position"
    manipulability_barrier_rate: float = 10.0
    manipulability_slack_weight: float = 1e3
    joint_limit_margin: float | None = None
    joint_limit_rate: float = 10.0
    nullspace_posture: np.ndarray | None = None
    nullspace_stiffness: float | np.ndarray = 10.0
    nullspace_damping: float | np.ndarray = 1.0

    def __post_init__(self):
        lower = np.array(self.acceleration_lower, dtype=float, copy=True)
        upper = np.array(self.acceleration_upper, dtype=float, copy=True)
        if lower.ndim != 1 or not lower.size or upper.shape != lower.shape:
            raise ValueError("Acceleration bounds must be nonempty vectors of equal shape")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("Acceleration bounds must be ordered and not NaN")
        values = {"acceleration_lower": lower, "acceleration_upper": upper}
        if (self.torque_lower is None) != (self.torque_upper is None):
            raise ValueError("Set both torque bounds or neither")
        if self.torque_lower is not None:
            tau_lower = np.array(self.torque_lower, dtype=float, copy=True)
            tau_upper = np.array(self.torque_upper, dtype=float, copy=True)
            if tau_lower.shape != lower.shape or tau_upper.shape != lower.shape:
                raise ValueError("Torque bounds must match the acceleration bound shape")
            if np.any(np.isnan(tau_lower)) or np.any(np.isnan(tau_upper)) or np.any(tau_lower > tau_upper):
                raise ValueError("Torque bounds must be ordered and not NaN")
            values.update(torque_lower=tau_lower, torque_upper=tau_upper)
        if self.posture is not None:
            posture = np.array(self.posture, dtype=float, copy=True)
            if posture.shape != lower.shape or not np.all(np.isfinite(posture)):
                raise ValueError("posture must be finite and match the acceleration bound shape")
            values["posture"] = posture
        if self.nullspace_posture is not None:
            posture = np.array(self.nullspace_posture, dtype=float, copy=True)
            if posture.shape != lower.shape or not np.all(np.isfinite(posture)):
                raise ValueError("nullspace_posture must be finite and match the acceleration bound shape")
            values["nullspace_posture"] = posture
        for name in ("nullspace_stiffness", "nullspace_damping"):
            gains = np.broadcast_to(np.asarray(getattr(self, name), dtype=float), lower.shape).copy()
            if not np.all(np.isfinite(gains)) or np.any(gains < 0):
                raise ValueError(f"{name} must be finite and nonnegative, scalar or one per joint")
            values[name] = gains
        for name, value in values.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        for kp_name, kd_name in (("kp_pos", "kd_pos"), ("kp_ori", "kd_ori"),
                                 ("posture_kp", "posture_kd")):
            kp = getattr(self, kp_name)
            if not np.isfinite(kp) or kp < 0:
                raise ValueError(f"{kp_name} must be finite and nonnegative")
            if getattr(self, kd_name) is None:
                object.__setattr__(self, kd_name, 2.0 * float(np.sqrt(kp)))
        for name in ("kd_pos", "kd_ori", "posture_kd", "position_weight", "orientation_weight",
                     "acceleration_regularization", "torque_regularization"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.position_weight == self.orientation_weight == 0:
            raise ValueError("At least one task weight must be positive")
        if self.acceleration_regularization == 0:
            raise ValueError("acceleration_regularization must be positive for a unique solution")
        if self.manipulability_rows not in MANIPULABILITY_ROWS:
            raise ValueError(f"manipulability_rows must be one of {sorted(MANIPULABILITY_ROWS)}")
        for name in ("manipulability_gain", "manipulability_min", "manipulability_barrier_rate",
                     "joint_limit_rate"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.manipulability_slack_weight) or self.manipulability_slack_weight <= 0:
            raise ValueError("manipulability_slack_weight must be finite and positive")
        if self.joint_limit_margin is not None and (
            not np.isfinite(self.joint_limit_margin) or self.joint_limit_margin < 0
        ):
            raise ValueError("joint_limit_margin must be finite and nonnegative")



@dataclass(frozen=True)
class ImpedanceSolution:
    """Diagnostics from the last QP.

    ``manipulability`` is w at the state the QP was solved for, and ``manipulability_slack``
    is how far the floor constraint had to be relaxed (0 when it held or is disabled).
    """

    joint_accelerations: np.ndarray
    torques: np.ndarray
    desired_task_acceleration: np.ndarray
    manipulability: float = float("nan")
    manipulability_slack: float = 0.0
    nullspace_torques: np.ndarray | None = None
