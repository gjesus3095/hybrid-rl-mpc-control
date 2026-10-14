import numpy as np
import scipy.sparse as sp
import osqp
from hybrid_control.controllers.base import Controller
from hybrid_control.controllers.ik_qp.config import IKQPConfig, PoseTarget
from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import JointPositionCommand
from hybrid_control.utils.math_helpers import MathHelpers
from hybrid_control.utils.osqp_utils import suppress_c_stdout

# ==================================================================================================
# DIFF INVERSE KINEMATICS CONTROLLER (QP)
# ==================================================================================================

class InverseKinematicsController(Controller[PoseTarget, JointPositionCommand], MathHelpers):
    """Differential IK/QP that computes position references without applying them."""

    def __init__(self, robot: RobotModel, config: IKQPConfig):
        self.robot = robot
        self.config = config
        self.n_dofs = len(robot.joint_names)
        if config.velocity_lower.shape != (self.n_dofs,):
            raise ValueError("Velocity bounds must match the model's selected joint count")

    # ----------------------------------------------------------------------------------------------
    # 1. SOLVERS
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def solve_box_qp(H: np.ndarray, f: np.ndarray,
                     lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """
        Solves the box-constrained Quadratic Program:
            min  1/2 x^T H x + f^T x
            s.t. lb <= x <= ub

        Args:
            H (np.ndarray): Hessian matrix (nxn).
            f (np.ndarray): Gradient vector (n,).
            lb (np.ndarray): Lower bound vector (n,).
            ub (np.ndarray): Upper bound vector (n,).

        Returns:
            np.ndarray: Optimal solution vector x.
        """
        n = H.shape[0]

        # Convert to sparse CSC format for OSQP
        P = sp.csc_matrix((H + H.T) * 0.5)
        q = f
        A = sp.eye(n, format="csc")

        # Initialize solver
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, u=ub, l=lb, verbose=False, polish=True)

        with suppress_c_stdout():
            res = prob.solve()

        # Check solver status (1=solved, 2=solved_inaccurate)
        if res.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP Solver failed: {res.info.status}")

        return res.x

    def quat_error_eps_form(self, q_des: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
        """
        Computes orientation error using the epsilon/eta formulation.
        Quaternion Error Calculation CLIK with Lyapunov Theoretical Guarantee:
        "The Unit Quaternion: A Useful Tool for Inverse Kinematics of Robot Manipulators"
        by S. Chiaverini and B. Siciliano


        Args:
            q_des (np.ndarray): Desired quaternion [x, y, z, w].
            q_cur (np.ndarray): Current quaternion [x, y, z, w].

        Returns:
            e_phi (np.ndarray): 3D orientation error vector.
        """
        q_des = self.normalize_quaternion(q_des)
        q_cur = self.normalize_quaternion(q_cur)

        # Split Vector (eps) and Scalar (eta) parts
        eps_des = q_des[:3]
        eta_des = q_des[3]

        eps_cur = q_cur[:3]
        eta_cur = q_cur[3]

        # Ensure shortest path (same hemisphere)
        if np.dot(q_des, q_cur) < 0.0:
            eps_des = -eps_des
            eta_des = -eta_des

        # Calculate error: e = eta_c*eps_d - eta_d*eps_c - S(eps_d)eps_c
        e_phi = (
                eta_cur * eps_des
                - eta_des * eps_cur
                - np.cross(eps_des, eps_cur)
        )

        return e_phi

    # ----------------------------------------------------------------------------------------------
    # 3. CORE CONTROL LOGIC
    # ----------------------------------------------------------------------------------------------
    def compute(self, target: PoseTarget, dt: float) -> JointPositionCommand:
        """Return q + dq*dt in model joint order; dt is in seconds.

        Reads use the current simulation snapshot. The caller must keep physics
        paused during computation and apply the returned command separately.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(target, PoseTarget):
            raise TypeError("target must be a PoseTarget")
        config = self.config
        state = self.robot.get_state()
        q_current = state.positions
        if q_current.shape != (self.n_dofs,):
            raise ValueError("Robot state must match the selected joint count")
        q_min, q_max = self.robot.position_limits
        q_min, q_max = np.asarray(q_min), np.asarray(q_max)
        if (
            q_min.shape != (self.n_dofs,) or q_max.shape != (self.n_dofs,)
            or np.any(np.isnan(q_min)) or np.any(np.isnan(q_max))
            or np.any(q_min > q_max)
        ):
            raise ValueError("Robot position limits must be ordered vectors matching the selected joints")
        pose = self.robot.get_pose(target.frame_name)
        jacobian = np.asarray(self.robot.get_jacobian(target.frame_name), dtype=float)
        if jacobian.shape != (6, self.n_dofs) or not np.all(np.isfinite(jacobian)):
            raise ValueError("Robot Jacobian must be finite with shape (6, n_joints)")

        linear_velocity = config.kp_pos * (target.pose.position - pose.position)
        angular_velocity = config.kp_ori * self.quat_error_eps_form(
            target.pose.quaternion, pose.quaternion,
        )
        error_des = np.hstack([linear_velocity, angular_velocity])

        # Preserve the imported objective and its task scaling.
        task_weight = 1e8
        K = np.diag([config.position_weight] * 3 + [config.orientation_weight] * 3)
        W = np.eye(6) * task_weight
        JW = W @ jacobian
        bW = K @ error_des
        H = jacobian.T @ JW + config.damping * np.eye(self.n_dofs)
        f = -(bW.T @ JW).T

        # Intersect velocity bounds with one-step joint position feasibility.
        lower = np.maximum(config.velocity_lower, (q_min - q_current) / dt)
        upper = np.minimum(config.velocity_upper, (q_max - q_current) / dt)
        lower = np.minimum(lower, upper)  # Preserve the imported feasibility clamp.
        dq = self.solve_box_qp(H, f, lower, upper)

        q_command = q_current + dt * dq
        return JointPositionCommand(self.robot.joint_names, q_command)
