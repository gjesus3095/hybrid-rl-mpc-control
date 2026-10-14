"""Isaac adapters on a fake articulation whose data mimics Isaac Lab's ProxyArray (``.torch``)."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hybrid_control.robots.state import JointPositionCommand, JointTorqueCommand, Pose
from hybrid_control.sim_interfaces.isaac.adapter import (
    IsaacArticulationBuffer, IsaacDynamicsBuffer, IsaacRobotModel, IsaacTorqueRobotModel,
)


class _Array:
    """Stands in for ProxyArray.torch -> tensor.cpu().numpy()."""

    def __init__(self, value):
        self.torch, self.value = self, np.asarray(value, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()

    def __getitem__(self, index):
        return _Array(self.value[index])

    def __add__(self, other):
        return _Array(self.value + np.asarray(other))


JOINTS = ("base_x", "j1", "j2")
BODIES = ("world", "base", "hand")


def fake_articulation(num_envs=2):
    pose = np.zeros((num_envs, 3, 7))
    pose[..., 6] = 1.0
    pose[:, 2, :3] = [1.0, 2.0, 3.0]
    pose[:, 2, 3:] = Rotation.from_euler("z", np.pi / 2).as_quat()
    com = np.zeros((num_envs, 3, 3))
    com[:, 1], com[:, 2] = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    data = SimpleNamespace(
        joint_pos=_Array(np.tile([0.5, 0.1, 0.2], (num_envs, 1))), joint_vel=_Array(np.zeros((num_envs, 3))),
        joint_pos_limits=_Array(np.tile([[-1, 1], [-2, 2], [-3, 3]], (num_envs, 1, 1))),
        body_link_pose_w=_Array(pose), body_com_pos_w=_Array(com),
        body_mass=_Array(np.tile([0.0, 3.0, 1.0], (num_envs, 1))))
    return SimpleNamespace(joint_names=list(JOINTS), body_names=list(BODIES), data=data)


def test_reads_selected_joints_and_bodies():
    buffer = IsaacArticulationBuffer(fake_articulation(), ("j1", "j2"))
    robot = IsaacRobotModel(buffer, env=1)
    np.testing.assert_allclose(robot.get_state().positions, [0.1, 0.2])
    np.testing.assert_allclose(robot.position_limits[1], [2, 3])
    np.testing.assert_allclose(robot.get_pose("hand").position, [1, 2, 3])
    np.testing.assert_allclose(robot.get_com_state().position, [0.25, 0, 0])
    with pytest.raises(ValueError, match="not in the articulation"):
        robot.get_pose("tcp")


def test_extra_frame_and_override():
    buffer = IsaacArticulationBuffer(fake_articulation(), ("j1", "j2"))
    offset = Pose(np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    robot = IsaacRobotModel(buffer, env=0, extra_frames={"tcp": ("hand", offset)})
    np.testing.assert_allclose(robot.get_pose("tcp").position, [1.0, 2.1, 3.0], atol=1e-12)  # hand yawed 90 deg
    override = Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    robot.pose_overrides["hand"] = override
    np.testing.assert_allclose(robot.get_pose("tcp").position, [0.1, 0.0, 0.0], atol=1e-12)


def test_apply_command_only_touches_its_env():
    buffer = IsaacArticulationBuffer(fake_articulation(), ("j1", "j2"))
    IsaacRobotModel(buffer, env=1).apply_command(JointPositionCommand(("j1", "j2"), np.array([0.3, 0.4])))
    np.testing.assert_allclose(buffer.targets, [[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="joint order"):
        IsaacRobotModel(buffer, env=0).apply_command(JointPositionCommand(("j2", "j1"), np.zeros(2)))


def test_coms_matches_per_env_com():
    articulation = fake_articulation(num_envs=3)
    articulation.data.body_com_pos_w.value[2, 2] = [3.0, 1.0, 0.0]  # env 2 differs
    buffer = IsaacArticulationBuffer(articulation, ("j1", "j2"))
    expected = np.array([buffer.com(env, ("base", "hand")) for env in range(3)])
    np.testing.assert_allclose(buffer.coms(("base", "hand")), expected)
    np.testing.assert_allclose(expected[2], [0.75, 0.25, 0.0])


def fake_dynamics(num_envs=2):
    """Fixed base: the Jacobian has no row for the root body ("world")."""
    articulation = fake_articulation(num_envs)
    rng = np.random.default_rng(0)
    jacobian = np.zeros((num_envs, len(BODIES) - 1, 6, len(JOINTS)))
    jacobian[:, 1] = rng.normal(size=(6, len(JOINTS)))  # hand
    articulation.data.__dict__.update(
        has_joint_ordering=False,
        joint_effort_limits=_Array(np.tile([100.0, 87.0, 12.0], (num_envs, 1))),
        mass_matrix=_Array(np.tile(np.arange(9.0).reshape(3, 3), (num_envs, 1, 1))),
        gravity_compensation_forces=_Array(np.tile([0.0, 1.0, 2.0], (num_envs, 1))),
        body_link_jacobian_w=_Array(jacobian))
    articulation.__dict__.update(num_base_dofs=0, is_fixed_base=True)
    return articulation


@pytest.fixture
def no_warp(monkeypatch):
    monkeypatch.setattr(IsaacDynamicsBuffer, "_coriolis", lambda self: np.full((self.num_envs, 3), 0.5))


def test_dynamics_of_selected_joints(no_warp):
    buffer = IsaacDynamicsBuffer(fake_dynamics(), ("j1", "j2"), jacobian_bodies=["hand"])
    robot = IsaacTorqueRobotModel(buffer, env=1)
    np.testing.assert_allclose(robot.get_mass_matrix(), [[4, 5], [7, 8]])
    np.testing.assert_allclose(robot.get_nonlinear_forces(), [1.5, 2.5])
    np.testing.assert_allclose(robot.torque_limits[0], [-87, -12])
    np.testing.assert_allclose(robot.get_jacobian("hand"), buffer.articulation.data.body_link_jacobian_w.value[1, 1][:, 1:])


def test_extra_frame_jacobian_is_shifted_to_its_origin(no_warp):
    buffer = IsaacDynamicsBuffer(fake_dynamics(), ("j1", "j2"), jacobian_bodies=["hand"])
    offset = Pose(np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    robot = IsaacTorqueRobotModel(buffer, env=0, extra_frames={"tcp": ("hand", offset)})
    hand, tcp = robot.get_jacobian("hand"), robot.get_jacobian("tcp")
    r = np.array([0.0, 0.1, 0.0])  # offset in world (hand yawed 90 deg)
    dq = np.array([0.3, -0.7])
    np.testing.assert_allclose(tcp @ dq, np.r_[hand[:3] @ dq + np.cross(hand[3:] @ dq, r), hand[3:] @ dq])


def test_jacobian_derivative_is_a_backward_difference(no_warp):
    articulation = fake_dynamics()
    buffer = IsaacDynamicsBuffer(articulation, ("j1", "j2"), jacobian_bodies=["hand"])
    robot = IsaacTorqueRobotModel(buffer, env=0)
    first = robot.get_jacobian("hand")
    np.testing.assert_allclose(robot.get_jacobian_derivative("hand"), 0.0)
    articulation.data.body_link_jacobian_w.value[:, 1] += 1.0
    buffer.read(0.5)
    np.testing.assert_allclose(robot.get_jacobian_derivative("hand"), (robot.get_jacobian("hand") - first) / 0.5)
    np.testing.assert_allclose(robot.get_jacobian_derivative("hand"), 2.0)  # same step: same value


def test_torque_command_only_touches_its_env(no_warp):
    buffer = IsaacDynamicsBuffer(fake_dynamics(), ("j1", "j2"), jacobian_bodies=["hand"])
    IsaacTorqueRobotModel(buffer, env=1).apply_command(JointTorqueCommand(("j1", "j2"), np.array([3.0, 4.0])))
    np.testing.assert_allclose(buffer.efforts, [[0, 0], [3, 4]])


def test_controller_modes_match_isaac_adapters(no_warp):
    from hybrid_control.controllers.ik_qp.ik_qp_mujoco import InverseKinematicsController
    from hybrid_control.utils.control_utils import IKQPConfig
    from hybrid_control.controllers.impedance.impedance_control import ImpedanceController
    from hybrid_control.utils.control_utils import ImpedanceQPConfig

    position = IsaacRobotModel(IsaacArticulationBuffer(fake_articulation(), ("j1", "j2")), env=0)
    torque = IsaacTorqueRobotModel(IsaacDynamicsBuffer(fake_dynamics(), ("j1", "j2"), ["hand"]), env=0)
    ik = InverseKinematicsController(position, IKQPConfig([-1, -1], [1, 1]))
    impedance = ImpedanceController(torque, ImpedanceQPConfig([-1, -1], [1, 1]))
    assert ik.control_mode == position.control_mode == "position"
    assert impedance.control_mode == torque.control_mode == "torque"
    with pytest.raises(ValueError, match="requires control_mode='position'.*control_mode='torque'"):
        InverseKinematicsController(torque, None)
    with pytest.raises(ValueError, match="requires control_mode='torque'.*control_mode='position'"):
        ImpedanceController(position, None)
    for robot in (position, torque):
        with pytest.raises(AttributeError):
            robot.control_mode = "velocity"


@pytest.mark.parametrize("model_type,mode", [
    (IsaacRobotModel, "position"), (IsaacTorqueRobotModel, "torque"),
])
def test_model_constructor_validates_control_mode(model_type, mode):
    # Construction needs no simulator reads, and rejects mismatches immediately.
    assert model_type(None, 0, expected_control_mode=mode).control_mode == mode
    other_mode = "torque" if mode == "position" else "position"
    with pytest.raises(ValueError, match="Controller requires control_mode"):
        model_type(None, 0, expected_control_mode=other_mode)
    with pytest.raises(ValueError, match="Unknown control mode"):
        model_type(None, 0, expected_control_mode="invalid")

    class InvalidModeModel(model_type):
        @property
        def control_mode(self):
            return "invalid"

    with pytest.raises(ValueError, match="Unknown control mode"):
        InvalidModeModel(None, 0)
