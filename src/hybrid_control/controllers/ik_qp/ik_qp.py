"""Differential IK/QP on Isaac Lab's Franka Panda description, solved with Pink.

Kinematic model (TEMPORARY VENDORED COPY)
-----------------------------------------
``DEFAULT_URDF`` is a verbatim copy of Isaac Lab's
``source/isaaclab/isaaclab/controllers/config/data/lula_franka_gen.urdf``
(github.com/isaac-sim/IsaacLab @ 6ec3e9b, BSD-3, see ``docs/licenses/IsaacLab-LICENSE``).
It is copied because Isaac Lab 3 requires Python 3.12, while this project still pins
Python < 3.12 (and numpy 1.23). When that requirement is resolved, delete
``sim_interfaces/isaac/assets/franka/`` and load the file from the installed package::

    from importlib.resources import files
    DEFAULT_URDF = files("isaaclab.controllers.config") / "data" / "lula_franka_gen.urdf"

Its arm chain is identical to the Isaac Sim Pink extension's ``robot_configurations/franka/robot.urdf``.

Naming follows the Isaac Lab asset, so the ``RobotModel`` must expose these names:
joints ``panda_joint1..7``; frames ``panda_link0..8``, ``panda_hand``, ``panda_end_effector``.
"""

from importlib.resources import files
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin
import qpsolvers
import scipy.sparse as sp
from pink import Configuration, build_ik
from pink.exceptions import NoSolutionFound
from pink.limits import Limit
from pink.tasks import FrameTask

from hybrid_control.controllers.base import Controller
from hybrid_control.controllers.ik_qp.config import IKQPConfig, PoseTarget
from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import JointPositionCommand, Pose

DEFAULT_URDF = files("hybrid_control.sim_interfaces.isaac.assets.franka") / "lula_franka_gen.urdf"


def _se3(pose: Pose) -> pin.SE3:
    return pin.SE3(pin.Quaternion(pose.quaternion).toRotationMatrix(), pose.position)


class _JointBoxLimit(Limit):
    """Box limit lower <= dq <= upper (rad/s), as Pink inequality G @ Delta_q <= h; zero bounds lock a joint."""

    def __init__(self, n: int):
        self.G = np.vstack([np.eye(n), -np.eye(n)])
        self.lower = np.zeros(n)
        self.upper = np.zeros(n)

    def compute_qp_inequalities(self, q: np.ndarray, dt: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self.G, np.hstack([self.upper * dt, -self.lower * dt])


class PinkInverseKinematicsController(Controller[PoseTarget, JointPositionCommand]):
    """Same objective and constraints as ``ik_qp_mujoco.InverseKinematicsController``, solved with Pink.

    Kinematics come from the URDF, joint state/limits and the root-frame pose from ``robot``.
    World-frame targets are re-expressed relative to ``root_frame``, so a moving base (whose
    DoFs are not optimized, as in ik_qp_mujoco) is handled. URDF joints that are not in
    ``robot.joint_names`` (e.g. fingers) stay at their neutral position via zero velocity bounds.
    (Pinocchio 2.6's buildReducedModel rejects this URDF: link and joint ``panda_end_effector``
    share a name.)
    """

    TASK_WEIGHT = 1e8  # task weight of ik_qp_mujoco; config.damping is relative to it
    # Supported qpsolvers backends and their options. "daqp" (dense active-set, exact) is what
    # Isaac Lab's PinkIKController uses; "osqp" (sparse ADMM) is the Isaac Sim Pink default.
    # Pink's QP is O(dt^2) in scale; OSQP's default tolerances are too loose for it, especially
    # with an active bound. polish=True recovers the exact active set.
    SOLVER_OPTIONS = {"osqp": {"eps_abs": 1e-7, "eps_rel": 1e-7, "polish": True}, "daqp": {}}

    def __init__(self, robot: RobotModel, config: IKQPConfig, urdf_path=DEFAULT_URDF,
                 root_frame: str = "panda_link0", solver: str = "osqp"):
        self.robot = robot
        self.config = config
        self.root_frame = root_frame
        if solver not in self.SOLVER_OPTIONS or solver not in qpsolvers.available_solvers:
            raise ValueError(f"Solver '{solver}' must be one of {sorted(self.SOLVER_OPTIONS)} and installed "
                             f"(available: {qpsolvers.available_solvers}); install with `uv sync --extra isaac`")
        self.solver = solver
        names = tuple(robot.joint_names)
        if config.velocity_lower.shape != (len(names),):
            raise ValueError("Velocity bounds must match the model's selected joint count")

        model = pin.buildModelFromUrdf(str(urdf_path))
        missing = [n for n in names if not model.existJointName(n)]
        if missing:
            raise ValueError(f"Joints {missing} are not in {urdf_path}; the RobotModel must use the URDF joint names")
        if not model.existFrame(root_frame):
            raise ValueError(f"Root frame '{root_frame}' is not in {urdf_path}")

        # Robot joint i lives at q index _idx_q[i] and tangent index _idx_v[i] of the URDF model
        joints = [model.joints[model.getJointId(n)] for n in names]
        if any(j.nq != 1 or j.nv != 1 for j in joints):
            raise ValueError("Only one-DoF joints are supported")
        self._idx_q = np.array([j.idx_q for j in joints])
        self._idx_v = np.array([j.idx_v for j in joints])
        self._q_pin = pin.neutral(model)
        self._configuration = Configuration(model, model.createData(), self._q_pin)
        self._limit = _JointBoxLimit(model.nv)  # joints outside robot.joint_names keep zero bounds
        self._tasks = {}

    def _frame_tasks(self, frame_name: str) -> Tuple[FrameTask, FrameTask]:
        """Separate position/orientation tasks so kp_pos and kp_ori act as separate gains."""
        if frame_name not in self._tasks:
            if not self._configuration.model.existFrame(frame_name):
                raise ValueError(f"Frame '{frame_name}' is not in the URDF")
            self._tasks[frame_name] = (
                FrameTask(frame_name, position_cost=1.0, orientation_cost=0.0),
                FrameTask(frame_name, position_cost=0.0, orientation_cost=1.0),
            )
        return self._tasks[frame_name]

    def compute(self, target: PoseTarget, dt: float) -> JointPositionCommand:
        """Return q + dq*dt in robot joint order, with dq from the Pink QP (dt in seconds).

        In Pink's variable Delta_q = dq*dt the problem is
            min  sum_i ||J_i Delta_q + alpha_i e_i||^2 + (damping / TASK_WEIGHT) ||Delta_q||^2
            s.t. max(dq_lower, (q_min - q)/dt) <= dq <= min(dq_upper, (q_max - q)/dt)
        with alpha_pos = position_weight*kp_pos*dt and alpha_ori = 0.5*orientation_weight*kp_ori*dt.
        The 0.5 matches ik_qp_mujoco's quaternion eps-error (~theta/2); Pink uses the SE(3) log (~theta).
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(target, PoseTarget):
            raise TypeError("target must be a PoseTarget")
        config = self.config
        q = self.robot.get_state().positions
        q_min, q_max = (np.asarray(limit, dtype=float) for limit in self.robot.position_limits)

        # 1. URDF kinematics at the measured joint positions
        self._q_pin[self._idx_q] = q
        self._configuration.update(self._q_pin)

        # 2. World target -> URDF world frame: T_urdf_root * T_world_root^-1 * T_world_target
        T_urdf_root = self._configuration.get_transform_frame_to_world(self.root_frame)
        target_urdf = T_urdf_root * _se3(self.robot.get_pose(self.root_frame)).actInv(_se3(target.pose))
        pos_task, ori_task = self._frame_tasks(target.frame_name)
        for task in (pos_task, ori_task):
            task.set_target(target_urdf)
        pos_task.gain = config.position_weight * config.kp_pos * dt
        ori_task.gain = 0.5 * config.orientation_weight * config.kp_ori * dt

        # 3. Velocity bounds intersected with one-step position feasibility (Pinocchio tangent order)
        lower = np.maximum(config.velocity_lower, (q_min - q) / dt)
        upper = np.minimum(config.velocity_upper, (q_max - q) / dt)
        lower = np.minimum(lower, upper)
        self._limit.lower[self._idx_v], self._limit.upper[self._idx_v] = lower, upper

        # 4. Solve; OSQP gets sparse P/G to avoid qpsolvers' dense->sparse conversion warning
        problem = build_ik(self._configuration, [pos_task, ori_task], dt,
                           damping=config.damping / self.TASK_WEIGHT, limits=[self._limit])
        if self.solver == "osqp":
            problem.P, problem.G = sp.csc_matrix(problem.P), sp.csc_matrix(problem.G)
        result = qpsolvers.solve_problem(problem, solver=self.solver, **self.SOLVER_OPTIONS[self.solver])
        if not result.found or result.x is None:
            raise NoSolutionFound(problem, result)

        dq = result.x[self._idx_v] / dt
        return JointPositionCommand(self.robot.joint_names, q + dt * dq)
