import numpy as np
import scipy.sparse as sp
import osqp
from scipy.spatial.transform import Rotation
from hybrid_control.controllers.base import Controller
from hybrid_control.utils.control_utils import ImpedanceQPConfig, ImpedanceSolution, TaskSpaceTarget
from hybrid_control.robots.model import ControlMode, RobotModel
from hybrid_control.robots.state import JointTorqueCommand
from hybrid_control.utils.manipulability import manipulability_gradient
from hybrid_control.utils.osqp_utils import suppress_c_stdout

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


class ImpedanceController(Controller[TaskSpaceTarget, JointTorqueCommand]):
    """Inverse-dynamics QP that computes joint torques for task-space impedance tracking.

    The QP is set up once and then updated in place each step, warm-starting OSQP
    from the previous solution. Like the IK controller, it only reads the robot;
    the caller applies the returned torques and steps the simulation.
    """

    @property
    def control_mode(self) -> ControlMode:
        return "torque"

    def __init__(self, robot: RobotModel, config: ImpedanceQPConfig):
        super().__init__(robot)
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
        with suppress_c_stdout():
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
            Rotation.from_quat(np.array(target.pose.quaternion)) * Rotation.from_quat(np.array(pose.quaternion)).inv()
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
