import numpy as np
import scipy.sparse as sp
import osqp
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
from hybrid_control.controllers.base import Controller
from hybrid_control.robots.model import TorqueRobotModel
from hybrid_control.robots.state import JointTorqueCommand, Pose
from hybrid_control.utils.manipulability import ROWS as MANIPULABILITY_ROWS, manipulability_gradient
from hybrid_control.utils.trajectories import CartesianSample

# ==================================================================================================
# TASK-SPACE IMPEDANCE CONTROLLER (INVERSE-DYNAMICS QP)
# ==================================================================================================
#
# Decision variable z = [ddq, tau, s] (2n + 1,). Each control step solves
#
#   min   || ddx_des - (J ddq + dJ dq) ||²_W
#         + λ_ddq || ddq - ddq_ref ||² + λ_tau || tau ||²            (Tikhonov regularization)
#         + ρ s²                                                      (manipulability-floor slack)
#   s.t.  [M, -I] [ddq; tau] = -h                                     (rigid-body dynamics)
#         ∇wᵀ ddq + s >= -(d∇w/dt)ᵀ dq - 2α ∇wᵀ dq - α² (w - w_min)   (manipulability floor, optional)
#         ddq_min <= ddq <= ddq_max   (tightened by joint-limit barriers, optional)
#         tau_min <= tau <= tau_max,  s >= 0
#
# with the Cartesian impedance (PD + feedforward) law
#
#   ddx_des = ddx_ref + Kp (x_ref - x) + Kd (dx_ref - dx)
#
# where the orientation error is the world-frame rotation vector of R_ref R^T, and the
# secondary-objective reference
#
#   ddq_ref = k_w ∇w(q) + Kp_posture (q_posture - q) - Kd_posture dq
#
# climbs the manipulability w = sqrt(det(J Jᵀ)) gradient. Because λ_ddq is small next to
# the task weight, it acts mostly in the null space of the task (soft priority). The floor
# is a second-order control barrier function on h = w - w_min: ḧ + 2α ḣ + α² h >= 0.
#
# Optionally, a null-space posture torque is added after the QP:
#
#   Λ = (J M⁻¹ Jᵀ)⁻¹,   N = I - Jᵀ Λ J M⁻¹               (dynamically consistent projector)
#   tau = tau_QP + N (P_null (q_null - q) - D_null dq)
#
# N removes any task-space force from the posture torque, so it only moves the arm's
# redundant (self-motion) DoF. The sum is clipped to the torque bounds.


def _vector3or6(value, size: int, name: str) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TaskSpaceTarget:
    """Desired world-frame pose, twist and twist acceleration of a named body origin.

    ``velocity`` (m/s, rad/s) and ``acceleration`` (m/s², rad/s²) have shape (6,),
    ordered [linear, angular] like Jacobian rows. Both default to zero (regulation).
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
        object.__setattr__(self, "velocity", _vector3or6(self.velocity, 6, "velocity"))
        object.__setattr__(self, "acceleration", _vector3or6(self.acceleration, 6, "acceleration"))

    @classmethod
    def from_sample(cls, frame_name: str, sample: CartesianSample) -> "TaskSpaceTarget":
        """Build a target from a trajectory sample."""
        return cls(frame_name, sample.pose, sample.velocity, sample.acceleration)


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


class ImpedanceController(Controller[TaskSpaceTarget, JointTorqueCommand]):
    """Inverse-dynamics QP that computes joint torques for task-space impedance tracking.

    The QP is set up once and then updated in place each step, warm-starting OSQP
    from the previous solution. Like the IK controller, it only reads the robot;
    the caller applies the returned torques and steps the simulation.
    """

    def __init__(self, robot: TorqueRobotModel, config: ImpedanceQPConfig):
        self.robot = robot
        self.config = config
        self.n_dofs = len(robot.joint_names)
        if config.acceleration_lower.shape != (self.n_dofs,):
            raise ValueError("Acceleration bounds must match the model's selected joint count")
        tau_lower, tau_upper = (np.asarray(limit, dtype=float) for limit in robot.torque_limits)
        if config.torque_lower is not None:
            tau_lower = np.maximum(tau_lower, config.torque_lower)
            tau_upper = np.minimum(tau_upper, config.torque_upper)
        if np.any(tau_lower > tau_upper):
            raise ValueError("Configured torque bounds do not overlap the robot torque limits")
        self.torque_lower, self.torque_upper = tau_lower, tau_upper
        self._build_sparsity()
        self.reset()

    def reset(self) -> None:
        """Forget the warm start, the captured posture and the manipulability history."""
        self._solver = None
        self._posture = self.config.posture
        self._previous_gradient: np.ndarray | None = None
        self.last_solution: ImpedanceSolution | None = None

    # ----------------------------------------------------------------------------------------------
    # 1. QP STRUCTURE
    # ----------------------------------------------------------------------------------------------
    def _build_sparsity(self) -> None:
        """Fix the CSC patterns of P (upper triangle) and A so OSQP can be updated in place."""
        n = self.n_dofs
        size = 2 * n + 1  # [ddq, tau, slack]
        p_pattern = np.zeros((size, size))
        p_pattern[:n, :n] = 1.0
        p_pattern[n:, n:] = np.eye(n + 1)
        a_pattern = np.vstack([
            np.hstack([np.ones((n, n)), np.eye(n), np.zeros((n, 1))]),   # Dynamics: [M, -I, 0].
            np.hstack([np.ones((1, n)), np.zeros((1, n)), np.ones((1, 1))]),  # Floor: [∇wᵀ, 0, 1].
            np.eye(size),                                                # Boxes on ddq, tau, slack.
        ])
        self._p_pattern = sp.csc_matrix(np.triu(p_pattern))
        self._a_pattern = sp.csc_matrix(a_pattern)
        self._p_index = self._csc_index(self._p_pattern)
        self._a_index = self._csc_index(self._a_pattern)
        self._a_box = np.eye(size)

    @staticmethod
    def _csc_index(pattern: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
        """Return (row, col) of each stored entry, in CSC data order."""
        cols = np.repeat(np.arange(pattern.shape[1]), np.diff(pattern.indptr))
        return pattern.indices.copy(), cols

    def _solve(self, H: np.ndarray, f: np.ndarray, A: np.ndarray,
               lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Solve min 1/2 zᵀHz + fᵀz s.t. lower <= Az <= upper, reusing the OSQP workspace."""
        Px = H[self._p_index]
        Ax = A[self._a_index]
        if self._solver is None:
            P = self._p_pattern.copy()
            P.data = Px
            A_csc = self._a_pattern.copy()
            A_csc.data = Ax
            self._solver = osqp.OSQP()
            self._solver.setup(
                P=P, q=f, A=A_csc, l=lower, u=upper, verbose=False, polish=True,
                eps_abs=1e-6, eps_rel=1e-6, max_iter=10000,
            )
        else:
            self._solver.update(Px=Px, Ax=Ax, q=f, l=lower, u=upper)
        result = self._solver.solve()
        # Check solver status (1=solved, 2=solved_inaccurate)
        if result.info.status_val not in (1, 2):
            self._solver = None  # Do not warm-start from a failed solve.
            raise RuntimeError(f"OSQP Solver failed: {result.info.status}")
        return result.x

    # ----------------------------------------------------------------------------------------------
    # 2. CORE CONTROL LOGIC
    # ----------------------------------------------------------------------------------------------
    def desired_task_acceleration(self, target: TaskSpaceTarget) -> np.ndarray:
        """Return ddx_des = ddx_ref + Kp (x_ref - x) + Kd (dx_ref - dx), shape (6,)."""
        config = self.config
        state = self.robot.get_state()
        pose = self.robot.get_pose(target.frame_name)
        jacobian = np.asarray(self.robot.get_jacobian(target.frame_name), dtype=float)
        twist = jacobian @ state.velocities

        position_error = target.pose.position - pose.position
        orientation_error = (
            Rotation.from_quat(target.pose.quaternion) * Rotation.from_quat(pose.quaternion).inv()
        ).as_rotvec()
        velocity_error = target.velocity - twist
        return target.acceleration + np.hstack([
            config.kp_pos * position_error + config.kd_pos * velocity_error[:3],
            config.kp_ori * orientation_error + config.kd_ori * velocity_error[3:],
        ])

    def compute(self, target: TaskSpaceTarget, dt: float) -> JointTorqueCommand:
        """Return joint torques for the current state; dt (s) is validated but not integrated.

        Reads use the current simulation snapshot, which must include up-to-date
        kinematics and dynamics. The caller applies the torques separately.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(target, TaskSpaceTarget):
            raise TypeError("target must be a TaskSpaceTarget")
        config, n, robot = self.config, self.n_dofs, self.robot
        state = robot.get_state()
        q, dq = state.positions, state.velocities
        if q.shape != (n,):
            raise ValueError("Robot state must match the selected joint count")
        jacobian = np.asarray(robot.get_jacobian(target.frame_name), dtype=float)
        jacobian_dot = np.asarray(robot.get_jacobian_derivative(target.frame_name), dtype=float)
        mass = np.asarray(robot.get_mass_matrix(), dtype=float)
        nonlinear = np.asarray(robot.get_nonlinear_forces(), dtype=float)
        for name, value, shape in (
            ("Jacobian", jacobian, (6, n)), ("Jacobian derivative", jacobian_dot, (6, n)),
            ("Mass matrix", mass, (n, n)), ("Nonlinear forces", nonlinear, (n,)),
        ):
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")

        if self._posture is None:
            self._posture = q.copy()
        ddq_reference = config.posture_kp * (self._posture - q) - config.posture_kd * dq

        # Manipulability gradient (secondary objective) and floor (barrier constraint).
        use_floor = config.manipulability_min > 0
        floor_row, floor_lower = np.zeros(n), -np.inf
        w = float("nan")
        if config.manipulability_gain > 0 or use_floor:
            w, gradient = manipulability_gradient(jacobian, config.manipulability_rows)
            ddq_reference = ddq_reference + config.manipulability_gain * gradient
            gradient_rate = (
                np.zeros(n) if self._previous_gradient is None
                else (gradient - self._previous_gradient) / dt
            )
            self._previous_gradient = gradient
            if use_floor:
                alpha = config.manipulability_barrier_rate
                floor_row = gradient
                floor_lower = (
                    -gradient_rate @ dq - 2 * alpha * (gradient @ dq)
                    - alpha**2 * (w - config.manipulability_min)
                )

        # Joint position-limit barriers tighten the ddq box, never beyond the physical bounds.
        ddq_lower, ddq_upper = config.acceleration_lower, config.acceleration_upper
        if config.joint_limit_margin is not None:
            q_min, q_max = (np.asarray(limit, dtype=float) for limit in robot.position_limits)
            alpha, margin = config.joint_limit_rate, config.joint_limit_margin
            with np.errstate(invalid="ignore"):  # Unlimited joints give ±inf bounds.
                barrier_lower = -2 * alpha * dq - alpha**2 * (q - (q_min + margin))
                barrier_upper = -2 * alpha * dq + alpha**2 * ((q_max - margin) - q)
            barrier_lower = np.where(np.isfinite(q_min), barrier_lower, -np.inf)
            barrier_upper = np.where(np.isfinite(q_max), barrier_upper, np.inf)
            ddq_upper = np.clip(np.minimum(ddq_upper, barrier_upper), ddq_lower, ddq_upper)
            ddq_lower = np.minimum(np.clip(barrier_lower, config.acceleration_lower, ddq_upper), ddq_upper)

        # Task term: ||W^1/2 (b - J ddq)||² with b = ddx_des - dJ dq.
        desired = self.desired_task_acceleration(target)
        b = desired - jacobian_dot @ dq
        W = np.diag([config.position_weight] * 3 + [config.orientation_weight] * 3)
        lam_q, lam_tau = config.acceleration_regularization, config.torque_regularization
        H = np.zeros((2 * n + 1, 2 * n + 1))
        H[:n, :n] = jacobian.T @ W @ jacobian + lam_q * np.eye(n)
        H[n:2 * n, n:2 * n] = lam_tau * np.eye(n)
        H[-1, -1] = config.manipulability_slack_weight
        f = np.hstack([-(jacobian.T @ W @ b) - lam_q * ddq_reference, np.zeros(n + 1)])

        # Equality M ddq - tau = -h, the manipulability floor, then box bounds on [ddq, tau, s].
        A = np.vstack([
            np.hstack([mass, -np.eye(n), np.zeros((n, 1))]),
            np.hstack([floor_row, np.zeros(n), [1.0]]),
            self._a_box,
        ])
        lower = np.hstack([-nonlinear, floor_lower, ddq_lower, self.torque_lower, 0.0])
        upper = np.hstack([-nonlinear, np.inf, ddq_upper, self.torque_upper,
                           np.inf if use_floor else 0.0])
        z = self._solve(H, f, A, lower, upper)

        ddq = z[:n]
        tau = z[n:2 * n]
        tau_null = None
        if config.nullspace_posture is not None:
            tau_null = self.nullspace_torque(jacobian, mass, q, dq)
            tau = tau + tau_null
        tau = np.clip(tau, self.torque_lower, self.torque_upper)  # Also removes solver-tolerance overshoot.
        self.last_solution = ImpedanceSolution(ddq, tau, desired, w, max(float(z[-1]), 0.0), tau_null)
        return JointTorqueCommand(robot.joint_names, tau)

    def nullspace_torque(self, jacobian: np.ndarray, mass: np.ndarray,
                         q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Return N (P (q_null - q) - D dq) with N = I - Jᵀ Λ J M⁻¹ and Λ = (J M⁻¹ Jᵀ)⁻¹.

        Λ uses a pseudo-inverse so the projector stays defined at task singularities.
        """
        config = self.config
        mass_inverse = np.linalg.inv(mass)
        task_inertia = np.linalg.pinv(jacobian @ mass_inverse @ jacobian.T)
        projector = np.eye(self.n_dofs) - jacobian.T @ task_inertia @ jacobian @ mass_inverse
        posture_torque = (
            config.nullspace_stiffness * (config.nullspace_posture - q)
            - config.nullspace_damping * dq
        )
        return projector @ posture_torque
