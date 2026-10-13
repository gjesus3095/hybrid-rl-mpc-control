"""Pink IK/QP (ik_qp) on the Isaac Lab Franka URDF against the exact optimum of ik_qp_mujoco's QP.

The robot is a Pinocchio model of the vendored URDF on a displaced, yawed base, so no
simulator is needed. The exact optimum comes from bounded least squares (BVLS); OSQP in
ik_qp_mujoco is not used as reference because its default tolerances are loose. Seeds are fixed.
"""

import numpy as np
import pytest
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

pin = pytest.importorskip("pinocchio")
pytest.importorskip("pink")

from hybrid_control.controllers.ik_qp.config import IKQPConfig, PoseTarget
from hybrid_control.controllers.ik_qp.ik_qp import DEFAULT_URDF, PinkInverseKinematicsController, build_model
from hybrid_control.controllers.ik_qp.ik_qp_mujoco import InverseKinematicsController
from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import JointState, Pose

ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
Q_HOME = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])  # FRANKA_PANDA_CFG init state
FRAME = "panda_hand"


class PinocchioRobot(RobotModel):
    """Read-only RobotModel on the URDF; fingers fixed at 0, base at world pose T_world_base."""

    def __init__(self, q, base: pin.SE3):
        self.model = pin.buildModelFromUrdf(str(DEFAULT_URDF))
        self.data = self.model.createData()
        self.base = base
        self.q = np.zeros(self.model.nq)
        self.q[:7] = q
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    @property
    def joint_names(self):
        return ARM_JOINTS

    @property
    def position_limits(self):
        return self.model.lowerPositionLimit[:7].copy(), self.model.upperPositionLimit[:7].copy()

    def get_state(self):
        return JointState(0.0, self.q[:7].copy(), np.zeros(7))

    def get_com_state(self):
        raise NotImplementedError

    def get_pose(self, frame_name):
        T = self.base * self.data.oMf[self.model.getFrameId(frame_name)]
        return Pose(T.translation, Rotation.from_matrix(T.rotation).as_quat())

    def get_jacobian(self, frame_name):
        J = pin.getFrameJacobian(self.model, self.data, self.model.getFrameId(frame_name),
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, :7]
        R = self.base.rotation
        return np.vstack([R @ J[:3], R @ J[3:]])

    def apply_command(self, command):
        raise NotImplementedError


def make_config(dq_max=20.0, damping=1e5):
    return IKQPConfig(np.full(7, -dq_max), np.full(7, dq_max), kp_pos=2 / 0.01, kp_ori=0.3 / 0.01, damping=damping)


def perturbed_target(robot, rng, scale):
    pose = robot.get_pose(FRAME)
    rotation = Rotation.from_rotvec(rng.normal(size=3) * scale) * Rotation.from_quat(np.array(pose.quaternion))
    return PoseTarget(FRAME, Pose(pose.position + rng.normal(size=3) * scale, rotation.as_quat()))


BASE = pin.SE3(Rotation.from_euler("z", 0.7).as_matrix(), np.array([0.4, -0.3, 0.1]))
DT = 0.01


def exact_dq(robot, config, target):
    """Optimum of ik_qp_mujoco's QP: min 1e8*||J dq - b||^2 + damping*||dq||^2 within its bounds."""
    q, pose, J = robot.get_state().positions, robot.get_pose(FRAME), robot.get_jacobian(FRAME)
    e_ori = InverseKinematicsController(robot, config).quat_error_eps_form(target.pose.quaternion, pose.quaternion)
    b = np.r_[config.kp_pos * (target.pose.position - pose.position), config.kp_ori * e_ori]
    q_min, q_max = robot.position_limits
    lower = np.maximum(config.velocity_lower, (q_min - q) / DT)
    upper = np.minimum(config.velocity_upper, (q_max - q) / DT)
    A = np.vstack([1e4 * J, np.sqrt(config.damping) * np.eye(7)])
    return lsq_linear(A, np.r_[1e4 * b, np.zeros(7)], bounds=(lower, upper), method="bvls", tol=1e-14).x


SOLVERS = ("osqp", "daqp")


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("damping,dq_max", [(1e5, 20.0), (5e3, 20.0), (1e8, 20.0), (1e5, 0.3)])
def test_pink_matches_exact_qp_optimum(damping, dq_max, solver):
    rng = np.random.default_rng(0)
    config = make_config(dq_max, damping)
    for _ in range(10):
        robot = PinocchioRobot(Q_HOME + rng.uniform(-0.3, 0.3, 7), BASE)
        lower, upper = robot.position_limits
        robot = PinocchioRobot(np.clip(robot.get_state().positions, lower + 1e-3, upper - 1e-3), BASE)
        target = perturbed_target(robot, rng, 1e-3)
        dq = (PinkInverseKinematicsController(robot, config, solver=solver).compute(target, DT).positions
              - robot.get_state().positions) / DT
        expected = exact_dq(robot, config, target)
        # Residual is the second-order gap between Pink's SE(3) log error and the quaternion error
        np.testing.assert_allclose(dq, expected, rtol=0, atol=2e-3 * np.abs(expected).max())


@pytest.mark.parametrize("solver", SOLVERS)
def test_command_respects_velocity_and_position_bounds(solver):
    rng = np.random.default_rng(1)
    q = Q_HOME.copy()
    q[3] = -0.0698 - 1e-4  # joint 4 just inside its upper limit
    robot = PinocchioRobot(q, BASE)
    command = PinkInverseKinematicsController(robot, make_config(dq_max=0.5), solver=solver).compute(
        perturbed_target(robot, rng, 0.1), DT)
    lower, upper = robot.position_limits
    assert np.all(np.abs(command.positions - q) <= 0.5 * DT + 1e-6)
    assert np.all(command.positions <= upper + 1e-6) and np.all(command.positions >= lower - 1e-6)
    assert command.joint_names == ARM_JOINTS


def test_rejects_non_urdf_joint_names():
    robot = PinocchioRobot(Q_HOME, BASE)
    robot.__class__ = type("Renamed", (PinocchioRobot,), {"joint_names": property(lambda s: tuple(f"joint{i}" for i in range(1, 8)))})
    with pytest.raises(ValueError, match="URDF joint names"):
        PinkInverseKinematicsController(robot, make_config())


def test_rejects_unsupported_solver():
    with pytest.raises(ValueError, match="Solver"):
        PinkInverseKinematicsController(PinocchioRobot(Q_HOME, BASE), make_config(), solver="quadprog")


TCP_OFFSET = Pose(np.array([0.0, 0.0, 0.1121]), np.array([0.0, 0.0, 0.0, 1.0]))


def test_extra_frame_is_parent_times_offset():
    model = build_model(extra_frames={"tcp": ("panda_hand", TCP_OFFSET)})
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    expected = data.oMf[model.getFrameId("panda_hand")] * pin.SE3(np.eye(3), TCP_OFFSET.position)
    np.testing.assert_allclose(data.oMf[model.getFrameId("tcp")].homogeneous, expected.homogeneous, atol=1e-12)


@pytest.mark.parametrize("solver", SOLVERS)
def test_tracks_extra_frame(solver):
    """A target that only moves the TCP must be reached by rotating about the TCP, not the hand."""
    robot = PinocchioRobot(Q_HOME, BASE)
    frames = {"tcp": ("panda_hand", TCP_OFFSET)}
    hand = robot.get_pose("panda_hand")
    rotation = Rotation.from_quat(np.array(hand.quaternion))
    tcp = Pose(hand.position + rotation.apply(np.array(TCP_OFFSET.position)), hand.quaternion)
    controller = PinkInverseKinematicsController(robot, make_config(), solver=solver, extra_frames=frames)
    command = controller.compute(PoseTarget("tcp", tcp), DT)
    np.testing.assert_allclose(command.positions, robot.get_state().positions, atol=1e-9)
