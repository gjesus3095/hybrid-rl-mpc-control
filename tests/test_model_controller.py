"""Contract checks for robot models and controllers, independent of Panda indexing."""

from contextlib import nullcontext
from importlib.resources import files
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hybrid_control.sim_interfaces.mujoco.adapter import MujocoRobotModel
from hybrid_control.controllers.base import Controller
from hybrid_control.controllers.ik_qp.ik_qp_mujoco import IKQPConfig, InverseKinematicsController, PoseTarget
from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import CenterOfMassState, JointPositionCommand, JointState, Pose


# A free joint makes qpos and qvel addresses differ; an unselected joint splits
# the controlled group. Actuator order also differs from joint declaration order.
MODEL_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="floating" pos="0 0 2">
      <freejoint name="free"/>
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="base" pos="0.2 -0.1 0.3" euler="0.2 -0.3 0.4">
      <joint name="hinge" axis="0 0 1" limited="true" range="-1 1"/>
      <geom type="sphere" size="0.05"/>
      <body name="middle" pos="0.4 0 0.1">
        <joint name="unselected" axis="0 1 0"/>
        <geom type="sphere" size="0.05"/>
        <body name="slider" pos="0.3 0.1 0.2">
          <joint name="extension" type="slide" axis="1 0 0" limited="true" range="-0.2 0.2"/>
          <geom type="sphere" size="0.05"/>
          <body name="tool" pos="0.2 0.1 0.15" euler="0.3 0.2 -0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="slide_servo" joint="extension" kp="100"/>
    <position name="unused_servo" joint="unselected" kp="100"/>
    <position name="hinge_servo" joint="hinge" kp="100"/>
  </actuator>
</mujoco>
"""
PAIRS = (("extension", "slide_servo"), ("hinge", "hinge_servo"))


@pytest.fixture
def robot():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    data.qpos[7:] = [0.25, -0.2, 0.08]
    data.qvel[6:] = [-0.4, 0.1, 0.3]
    data.ctrl[:] = [0.08, -0.2, 0.25]
    data.time = 1.25
    mujoco.mj_forward(model, data)
    return MujocoRobotModel(model, data, PAIRS)


def test_state_and_limits_follow_named_joint_order(robot):
    assert robot.joint_names == ("extension", "hinge")
    state = robot.get_state()
    assert state.time == 1.25
    np.testing.assert_array_equal(state.positions, [0.08, 0.25])
    np.testing.assert_array_equal(state.velocities, [0.3, -0.4])
    lower, upper = robot.position_limits
    np.testing.assert_array_equal(lower, [-0.2, -1])
    np.testing.assert_array_equal(upper, [0.2, 1])

    robot.data.qpos[9] = -0.05
    np.testing.assert_array_equal(state.positions, [0.08, 0.25])
    assert not np.shares_memory(state.positions, robot.data.qpos)
    assert not np.shares_memory(state.velocities, robot.data.qvel)
    lower[:] = -99
    np.testing.assert_array_equal(robot.position_limits[0], [-0.2, -1])


def test_unlimited_joint_limits_are_infinite(robot):
    other = MujocoRobotModel(robot.model, robot.data, [("unselected", "unused_servo")])
    lower, upper = other.position_limits
    assert lower[0] == -np.inf
    assert upper[0] == np.inf


def test_com_is_mass_weighted_over_robot_links_and_excludes_other_objects(robot):
    body_ids = [robot.model.body(name).id for name in ("base", "middle", "slider", "tool")]
    masses = robot.model.body_mass[body_ids]
    expected = np.average(robot.data.xipos[body_ids], weights=masses, axis=0)
    com = robot.get_com_state()
    assert com.time == robot.data.time
    np.testing.assert_allclose(com.position, expected, atol=1e-14)
    assert not np.allclose(com.position, robot.data.subtree_com[0])
    assert not np.shares_memory(com.position, robot.data.subtree_com)

    before = com.position.copy()
    robot.initialize([0.15, -0.4])
    assert not np.allclose(robot.get_com_state().position, before)
    np.testing.assert_array_equal(com.position, before)


def test_com_root_can_be_explicit_and_must_contain_selected_joints(robot):
    subtree = MujocoRobotModel(
        robot.model, robot.data, [("extension", "slide_servo")], root_body="slider",
    )
    np.testing.assert_allclose(
        subtree.get_com_state().position, robot.data.xipos[robot.model.body("slider").id],
    )
    for root in ("missing", "floating", "slider", "world"):
        with pytest.raises(ValueError, match="Unknown|root_body|non-world"):
            MujocoRobotModel(robot.model, robot.data, PAIRS, root_body=root)


@pytest.mark.parametrize("time, position", [
    (np.nan, [0, 0, 0]), (0, [0, 0]), (0, [0, np.inf, 0]),
])
def test_com_state_rejects_invalid_snapshots(time, position):
    with pytest.raises(ValueError):
        CenterOfMassState(time, position)


def test_load_and_initialize_panda_scene():
    scene = files("hybrid_control.sim_interfaces.mujoco.models.panda") / "scene.xml"
    robot = MujocoRobotModel.from_xml_path(
        scene, [(f"joint{i}", f"actuator{i}") for i in range(1, 8)],
    )
    initial = [0, 0, 0, -np.pi / 2, 0, np.pi / 2, -np.pi / 4]
    robot.initialize(
        initial,
        extra_joint_positions={"finger_joint1": 0.04, "finger_joint2": 0.04},
        extra_actuator_controls={"actuator8": 255},
    )
    np.testing.assert_array_equal(robot.get_state().positions, initial)
    np.testing.assert_array_equal(robot.get_state().velocities, np.zeros(7))
    np.testing.assert_array_equal(robot.data.ctrl, [*initial, 255])
    for name in ("finger_joint1", "finger_joint2"):
        assert robot.data.qpos[robot.model.joint(name).qposadr[0]] == 0.04
    assert robot.get_state().time == 0
    assert np.linalg.norm(robot.get_pose("hand").position) > 0
    # The Panda scene has no other massive objects: this includes both fingers.
    np.testing.assert_allclose(
        robot.get_com_state().position,
        np.average(robot.data.xipos[1:], weights=robot.model.body_mass[1:], axis=0),
    )


def test_panda_scene_contains_robot_and_ground():
    directory = files("hybrid_control.sim_interfaces.mujoco.models.panda")
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    robot_only = mujoco.MjModel.from_xml_path(str(directory / "panda.xml"))
    assert (model.nq, model.nv, model.nu) == (9, 9, 8)
    assert model.ngeom == robot_only.ngeom + 1, "Scene must contain the robot and one ground plane"
    ground = model.geom("ground")
    assert ground.type[0] == mujoco.mjtGeom.mjGEOM_PLANE, "Ground must be a plane"
    assert ground.bodyid[0] == 0, "Ground must be fixed to the world"
    np.testing.assert_array_equal(ground.pos, [0, 0, 0])


def test_initialize_uses_selected_order_and_preserves_unselected_state(robot):
    before_position = robot.get_pose("tool").position.copy()
    free_state = robot.data.qpos[:7].copy()
    robot.data.qfrc_applied[:] = 0.5
    robot.initialize([0.1, -0.4])
    np.testing.assert_array_equal(robot.get_state().positions, [0.1, -0.4])
    np.testing.assert_array_equal(robot.get_state().velocities, [0, 0])
    np.testing.assert_array_equal(robot.data.ctrl, [0.1, -0.2, -0.4])
    np.testing.assert_array_equal(robot.data.qpos[:7], free_state)
    assert robot.data.qpos[8] == -0.2
    assert robot.data.qvel[7] == 0.1
    np.testing.assert_array_equal(robot.data.qfrc_applied, np.full(robot.model.nv, 0.5))
    assert robot.data.time == 1.25
    assert not np.allclose(robot.get_pose("tool").position, before_position)


@pytest.mark.parametrize("positions, extras", [
    ([0.1], {}),
    ([np.nan, 0.1], {}),
    ([0.3, 0.1], {}),
    ([0.1, 0.1], {"extra_joint_positions": {"missing": 0}}),
    ([0.1, 0.1], {"extra_joint_positions": {"hinge": 0}}),
    ([0.1, 0.1], {"extra_joint_positions": {"free": 0}}),
    ([0.1, 0.1], {"extra_actuator_controls": {"hinge_servo": 0}}),
    ([0.1, 0.1], {"extra_actuator_controls": {"unused_servo": np.nan}}),
])
def test_initialize_validates_all_inputs_before_writing(robot, positions, extras):
    before = {name: getattr(robot.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
    with pytest.raises(ValueError):
        robot.initialize(positions, **extras)
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(robot.data, name), values)


def test_pose_is_world_frame_xyzw_and_owns_its_arrays(robot):
    pose = robot.get_pose("tool")
    np.testing.assert_allclose(pose.position, robot.data.body("tool").xpos)
    np.testing.assert_allclose(
        Rotation.from_quat(pose.quaternion).as_matrix(),
        robot.data.body("tool").xmat.reshape(3, 3), atol=1e-14,
    )
    assert not np.shares_memory(pose.position, robot.data.xpos)
    assert not np.shares_memory(pose.quaternion, robot.data.xquat)


def test_world_jacobian_matches_position_and_rotation_finite_differences(robot):
    model, data = robot.model, robot.data
    baseline = data.qpos.copy()
    jacobian = robot.get_jacobian("tool")
    numerical = np.zeros_like(jacobian)
    epsilon = 1e-6
    for column, name in enumerate(robot.joint_names):
        tangent = np.zeros(model.nv)
        tangent[model.joint(name).dofadr[0]] = 1.0
        poses = []
        for sign in (-1, 1):
            data.qpos[:] = baseline
            mujoco.mj_integratePos(model, data.qpos, tangent, sign * epsilon)
            mujoco.mj_forward(model, data)
            poses.append(robot.get_pose("tool"))
        minus, plus = poses
        numerical[:3, column] = (plus.position - minus.position) / (2 * epsilon)
        rotation_delta = Rotation.from_quat(plus.quaternion) * Rotation.from_quat(minus.quaternion).inv()
        numerical[3:, column] = rotation_delta.as_rotvec() / (2 * epsilon)
    data.qpos[:] = baseline
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(jacobian, numerical, rtol=1e-6, atol=1e-8)


def test_apply_command_only_changes_selected_actuator_references(robot):
    positions, velocities = robot.data.qpos.copy(), robot.data.qvel.copy()
    # Applying a command does not clamp it to joint limits or teleport the robot.
    robot.apply_command(JointPositionCommand(robot.joint_names, [0.3, -0.5]))
    np.testing.assert_array_equal(robot.data.ctrl, [0.3, -0.2, -0.5])
    np.testing.assert_array_equal(robot.data.qpos, positions)
    np.testing.assert_array_equal(robot.data.qvel, velocities)
    assert robot.data.time == 1.25


def test_panda_arm_mapping_leaves_gripper_command_untouched():
    scene = files("hybrid_control.sim_interfaces.mujoco.models.panda") / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.ctrl[model.actuator("actuator8").id] = 255
    robot = MujocoRobotModel(model, data, [(f"joint{i}", f"actuator{i}") for i in range(7, 0, -1)])
    values = np.array([-0.7, 1.5, 0, -1.3, 0.1, 0.2, 0.3])
    robot.apply_command(JointPositionCommand(robot.joint_names, values))
    np.testing.assert_array_equal(data.ctrl[:7], values[::-1])
    assert data.ctrl[7] == 255
    with pytest.raises(ValueError, match="directly drive"):
        MujocoRobotModel(model, data, [("finger_joint1", "actuator8")])


@pytest.mark.parametrize("pairs, message", [
    ([], "name pairs"),
    (["ab"], "name pairs"),
    ([("hinge",)], "name pairs"),
    ([("missing", "hinge_servo")], "Unknown"),
    ([("hinge", "missing")], "Unknown"),
    ([("hinge", "hinge_servo"), ("hinge", "unused_servo")], "unique"),
    ([("hinge", "hinge_servo"), ("extension", "hinge_servo")], "unique"),
    ([("hinge", "slide_servo")], "directly drive"),
    ([("free", "hinge_servo")], "scalar hinge or slide"),
])
def test_rejects_invalid_joint_actuator_mappings(robot, pairs, message):
    with pytest.raises(ValueError, match=message):
        MujocoRobotModel(robot.model, robot.data, pairs)


@pytest.mark.parametrize("actuator", [
    '<motor name="hinge_servo" joint="hinge"/>',
    '<velocity name="hinge_servo" joint="hinge" kv="100"/>',
    '<position name="hinge_servo" joint="hinge" kp="100" gear="2"/>',
    '<position name="hinge_servo" joint="hinge" kp="100" timeconst="0.1"/>',
])
def test_rejects_nonposition_or_nonunit_actuators(actuator):
    xml = MODEL_XML.replace('<position name="hinge_servo" joint="hinge" kp="100"/>', actuator)
    model = mujoco.MjModel.from_xml_string(xml)
    with pytest.raises(ValueError, match="position servo|unit gear"):
        MujocoRobotModel(model, mujoco.MjData(model), PAIRS)


def test_unknown_frames_and_command_order_fail_clearly(robot):
    for read in (robot.get_pose, robot.get_jacobian):
        with pytest.raises(ValueError, match="Unknown"):
            read("missing_body")
    before = robot.data.ctrl.copy()
    with pytest.raises(ValueError, match="names/order"):
        robot.apply_command(JointPositionCommand(("hinge", "extension"), [0.1, 0.2]))
    np.testing.assert_array_equal(robot.data.ctrl, before)


@pytest.mark.parametrize("positions", [[0], [[0, 0]], [0, np.nan], [np.inf, 0]])
def test_command_rejects_invalid_positions(positions):
    with pytest.raises(ValueError, match="positions"):
        JointPositionCommand(("hinge", "extension"), positions)


@pytest.mark.parametrize("quaternion", [[0, 0, 0, 0], [0, 0, 1], [0, 0, 0, np.nan]])
def test_pose_rejects_invalid_quaternions(quaternion):
    with pytest.raises(ValueError, match="quaternion"):
        Pose(np.zeros(3), quaternion)


def test_shared_types_copy_inputs_and_normalize_quaternions():
    position = np.array([0.1, 0.2, 0.3])
    quaternion = np.array([0.0, 0.0, 0.0, 2.0])
    pose = Pose(position, quaternion)
    command = JointPositionCommand(("a", "b", "c"), position)
    position[:] = 9
    quaternion[:] = 9
    np.testing.assert_array_equal(pose.position, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(command.positions, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(pose.quaternion, [0, 0, 0, 1])


def make_controller(robot: RobotModel) -> Controller[PoseTarget, JointPositionCommand]:
    return InverseKinematicsController(robot, IKQPConfig(
        velocity_lower=np.full(len(robot.joint_names), -0.5),
        velocity_upper=np.full(len(robot.joint_names), 0.5),
        kp_pos=10, kp_ori=10, damping=1e5,
    ))


def test_compute_and_reset_do_not_mutate_the_simulation(robot):
    controller = make_controller(robot)
    pose = robot.get_pose("tool")
    target = PoseTarget("tool", Pose(pose.position + [0.01, -0.02, 0.01], pose.quaternion))
    snapshot = {name: getattr(robot.data, name).copy() for name in (
        "qpos", "qvel", "qacc", "ctrl", "qfrc_applied", "xpos", "xquat",
    )}
    command = controller.compute(target, 0.02)
    controller.reset()
    assert isinstance(controller, Controller)
    assert command.joint_names == robot.joint_names
    assert not np.array_equal(command.positions, robot.get_state().positions)
    for name, previous in snapshot.items():
        np.testing.assert_array_equal(getattr(robot.data, name), previous, err_msg=name)
    assert robot.data.time == 1.25
    np.testing.assert_allclose(controller.compute(target, 0.02).positions, command.positions)


def test_equivalent_quaternion_signs_give_identical_commands(robot):
    controller = make_controller(robot)
    pose = robot.get_pose("tool")
    quaternion = (Rotation.from_euler("z", 0.1) * Rotation.from_quat(pose.quaternion)).as_quat()
    commands = [
        controller.compute(PoseTarget("tool", Pose(pose.position, sign * quaternion)), 0.02)
        for sign in (-1, 1)
    ]
    np.testing.assert_allclose(commands[0].positions, commands[1].positions, atol=1e-12)


@pytest.mark.parametrize("dt", [0, -0.01, np.nan, np.inf])
def test_compute_rejects_invalid_timestep(robot, dt):
    with pytest.raises(ValueError, match="dt"):
        make_controller(robot).compute(PoseTarget("tool", robot.get_pose("tool")), dt)


@pytest.mark.parametrize("lower, upper", [
    ([0], [0, 1]), ([np.nan], [1]), ([-1], [np.inf]), ([1], [-1]),
])
def test_invalid_velocity_bounds_are_rejected(lower, upper):
    with pytest.raises(ValueError):
        IKQPConfig(lower, upper)


@pytest.mark.parametrize("tuning", [
    {"kp_pos": np.nan}, {"kp_ori": -1}, {"damping": np.inf},
    {"position_weight": 0, "orientation_weight": 0},
])
def test_invalid_ik_tuning_is_rejected(tuning):
    with pytest.raises(ValueError):
        IKQPConfig([-1], [1], **tuning)


def test_controller_rejects_wrong_joint_count(robot):
    with pytest.raises(ValueError, match="joint count"):
        InverseKinematicsController(robot, IKQPConfig([-1], [1]))


class SliderModel(RobotModel):
    """A one-axis kinematic model; exercises IK without simulator-specific access."""

    joint_names = ("slider",)

    @property
    def position_limits(self):
        return np.array([0.0]), np.array([1.0])

    def get_state(self):
        return JointState(0.0, [0.99], [0.0])

    def get_com_state(self):
        return CenterOfMassState(0.0, [0.99, 0, 0])

    def get_pose(self, frame_name):
        return Pose([0.99, 0, 0], [0, 0, 0, 1])

    def get_jacobian(self, frame_name):
        return np.array([[1.0], [0], [0], [0], [0], [0]])

    def apply_command(self, command):
        raise AssertionError("Controller must not apply commands")


def test_controller_obeys_velocity_and_position_limits_with_a_generic_model():
    robot = SliderModel()
    controller = make_controller(robot)
    for reference, expected in (([2.0, 0, 0], 1.0), ([0.0, 0, 0], 0.94)):
        command = controller.compute(PoseTarget("tool", Pose(reference, [0, 0, 0, 1])), 0.1)
        np.testing.assert_allclose(command.positions, [expected], atol=1e-6)


def test_viewer_com_marker_tracks_robot_without_changing_physics():
    """Exercise the viewer geometry without opening a graphical window."""
    COM_MARKER_DIAMETER = 0.05
    JOINT_ACTUATOR_PAIRS = tuple((f"joint{i}", f"actuator{i}") for i in range(1, 8))
    Q_INIT = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])
    FRAME_CONTROL = "hand"

    def update_viewer_markers(handle, robot, target):
        com = robot.get_com_state()
        with handle.lock():
            handle.user_scn.ngeom = 2
            mujoco.mjv_initGeom(
                handle.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.012), target.pose.position, np.eye(3).ravel(),
                np.array([1, 0.2, 0.2, 0.7], dtype=np.float32),
            )
            mujoco.mjv_initGeom(
                handle.user_scn.geoms[1], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, COM_MARKER_DIAMETER / 2), com.position, np.eye(3).ravel(),
                np.array([0.2, 1, 0.3, 0.9], dtype=np.float32),
            )

    scene = files("hybrid_control.sim_interfaces.mujoco.models.panda") / "scene.xml"
    robot = MujocoRobotModel.from_xml_path(str(scene), JOINT_ACTUATOR_PAIRS)
    robot.initialize(Q_INIT)
    handle = SimpleNamespace(
        user_scn=mujoco.MjvScene(robot.model, maxgeom=2), lock=nullcontext,
    )
    target = PoseTarget(FRAME_CONTROL, robot.get_pose(FRAME_CONTROL))
    for joint1 in (0.0, 0.4):
        positions = Q_INIT.copy()
        positions[0] = joint1
        robot.initialize(positions)
        qpos, controls = robot.data.qpos.copy(), robot.data.ctrl.copy()
        update_viewer_markers(handle, robot, target)
        assert handle.user_scn.ngeom == 2
        np.testing.assert_allclose(handle.user_scn.geoms[0].pos, target.pose.position)
        marker = handle.user_scn.geoms[1]
        assert marker.type == mujoco.mjtGeom.mjGEOM_SPHERE
        np.testing.assert_allclose(marker.pos, robot.get_com_state().position)
        np.testing.assert_allclose(marker.size, np.full(3, 0.025))
        np.testing.assert_array_equal(robot.data.qpos, qpos)
        np.testing.assert_array_equal(robot.data.ctrl, controls)
