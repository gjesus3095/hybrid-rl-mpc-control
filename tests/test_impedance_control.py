"""Checks for torque-level Panda control: trajectories, torque adapter and impedance QP."""

from importlib.resources import files

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hybrid_control.controllers.impedance.impedance_control import (
    ImpedanceController,
    ImpedanceQPConfig,
    TaskSpaceTarget,
)
from hybrid_control.robots.state import JointPositionCommand, JointTorqueCommand, Pose
from hybrid_control.sim_interfaces.mujoco.adapter import MujocoRobotModel, MujocoTorqueRobotModel
from hybrid_control.sim_interfaces.mujoco.modifier import MujocoModifier
from hybrid_control.utils.trajectories import (
    QuinticTrajectory,
    SinusoidalTrajectory,
    quintic_time_scaling,
)


XML_FILE = str(files("hybrid_control.sim_interfaces.mujoco.models.panda") / "scene.xml")
PAIRS = tuple((f"joint{i}", f"actuator{i}") for i in range(1, 8))
Q_INIT = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])
DDQ_MAX = np.full(7, 50.0)


@pytest.fixture
def robot():
    model = MujocoModifier.build_torque_spec(XML_FILE, [a for _, a in PAIRS]).compile()
    data = mujoco.MjData(model)
    robot = MujocoTorqueRobotModel(model, data, PAIRS)
    robot.initialize(
        Q_INIT,
        extra_joint_positions={"finger_joint1": 0.04, "finger_joint2": 0.04},
        extra_actuator_controls={"actuator8": 255},
    )
    return robot


def make_controller(robot, **overrides):
    return ImpedanceController(robot, ImpedanceQPConfig(-DDQ_MAX, DDQ_MAX, **overrides))


def finite_difference(trajectory, t, h=1e-5):
    """Central differences of position and world rotation vector."""
    before, after = trajectory.sample(t - h), trajectory.sample(t + h)
    linear = (after.pose.position - before.pose.position) / (2 * h)
    angular = (
        Rotation.from_quat(after.pose.quaternion) * Rotation.from_quat(before.pose.quaternion).inv()
    ).as_rotvec() / (2 * h)
    return np.hstack([linear, angular]), (after.velocity - before.velocity) / (2 * h)


# --------------------------------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------------------------------
def test_quintic_scaling_is_rest_to_rest():
    assert quintic_time_scaling(0.0, 2.0) == (0.0, 0.0, 0.0)
    assert quintic_time_scaling(2.0, 2.0) == (1.0, 0.0, 0.0)
    assert quintic_time_scaling(5.0, 2.0) == (1.0, 0.0, 0.0)
    s, ds, _ = quintic_time_scaling(1.0, 2.0)
    assert s == pytest.approx(0.5) and ds == pytest.approx(0.9375)
    with pytest.raises(ValueError):
        quintic_time_scaling(0.0, 0.0)


def test_quintic_trajectory_derivatives_match_finite_differences():
    start = Pose([0.5, 0.0, 0.5], Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_quat())
    goal_rotation = Rotation.from_rotvec([0.2, -0.3, 0.4]) * Rotation.from_quat(start.quaternion)
    goal = Pose([0.6, 0.1, 0.4], goal_rotation.as_quat())
    trajectory = QuinticTrajectory(start, goal, 1.5)
    for t in (0.2, 0.75, 1.3):
        velocity, acceleration = finite_difference(trajectory, t)
        sample = trajectory.sample(t)
        np.testing.assert_allclose(sample.velocity, velocity, atol=1e-6)
        np.testing.assert_allclose(sample.acceleration, acceleration, atol=1e-4)
    end = trajectory.sample(1.5)
    np.testing.assert_allclose(end.pose.position, goal.position)
    assert abs(np.dot(end.pose.quaternion, goal.quaternion)) == pytest.approx(1.0)
    np.testing.assert_array_equal(end.velocity, np.zeros(6))


def test_sinusoidal_trajectory_derivatives_match_finite_differences():
    center = Pose([0.5, 0.0, 0.5], [0, 0, 0, 1])
    trajectory = SinusoidalTrajectory(center, [0.1, 0.05, 0.02], [0.5, 1.0, 2.0], phase=[0, 1, 2])
    for t in (0.0, 0.37, 1.1):
        velocity, acceleration = finite_difference(trajectory, t)
        sample = trajectory.sample(t)
        np.testing.assert_allclose(sample.velocity, velocity, atol=1e-6)
        np.testing.assert_allclose(sample.acceleration, acceleration, atol=1e-4)
        np.testing.assert_array_equal(sample.pose.quaternion, center.quaternion)


# --------------------------------------------------------------------------------------------------
# Torque adapter
# --------------------------------------------------------------------------------------------------
def test_torque_spec_turns_panda_servos_into_bounded_motors(robot):
    model = robot.model
    ids = robot._actuator_ids
    np.testing.assert_array_equal(model.actuator_gainprm[ids, 0], 1)
    assert np.all(model.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_NONE)
    lower, upper = robot.torque_limits
    np.testing.assert_array_equal(lower, -87)
    np.testing.assert_array_equal(upper, 87)
    # The gripper stays a position servo, and initialization starts the motors at zero torque.
    gripper = model.actuator("actuator8").id
    assert model.actuator_biastype[gripper] == mujoco.mjtBias.mjBIAS_AFFINE
    np.testing.assert_array_equal(robot.data.ctrl[ids], 0)
    assert robot.data.ctrl[gripper] == 255


def test_adapters_reject_the_wrong_actuator_type(robot):
    with pytest.raises(ValueError, match="unit-gain torque motor"):
        MujocoTorqueRobotModel.from_xml_path(XML_FILE, PAIRS)
    with pytest.raises(ValueError, match="position servo"):
        MujocoRobotModel(robot.model, robot.data, PAIRS)
    with pytest.raises(ValueError, match="Unknown actuator"):
        MujocoModifier.build_torque_spec(XML_FILE, ["missing"])


def test_dynamics_match_mujoco_forward_dynamics(robot):
    model, data = robot.model, robot.data
    rng = np.random.default_rng(0)
    data.qvel[:7] = rng.uniform(-1, 1, 7)
    data.ctrl[:7] = rng.uniform(-5, 5, 7)
    mujoco.mj_forward(model, data)
    mass = robot.get_mass_matrix()
    np.testing.assert_allclose(mass, mass.T)
    # M qacc + h = tau on the arm rows (finger coupling is not in the selected block).
    residual = mass @ data.qacc[:7] + robot.get_nonlinear_forces() - data.ctrl[:7]
    coupling = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, coupling, data.qM)
    residual += coupling[:7, 7:] @ data.qacc[7:]
    np.testing.assert_allclose(residual, 0, atol=1e-8)


def test_task_acceleration_matches_mujoco_body_acceleration(robot):
    # J ddq + dJ dq must equal MuJoCo's own world-frame body acceleration (gravity off,
    # since MuJoCo reports accelerations relative to free fall).
    model, data = robot.model, robot.data
    model.opt.gravity[:] = 0
    data.qvel[:7] = np.linspace(-0.8, 0.8, 7)
    data.ctrl[:7] = np.linspace(3, -3, 7)
    mujoco.mj_forward(model, data)
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    body = model.body("grasp").id
    jacobian_dot = robot.get_jacobian_derivative("grasp")
    np.testing.assert_array_equal(data.qpos, qpos)  # Reads leave the simulation untouched.
    predicted = robot.get_jacobian("grasp") @ data.qacc[:7] + jacobian_dot @ qvel[:7]
    mujoco.mj_rnePostConstraint(model, data)
    measured = np.zeros(6)  # [angular, linear] at the body origin, world frame.
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, measured, 0)
    np.testing.assert_allclose(predicted, np.hstack([measured[3:], measured[:3]]), atol=1e-5)


def test_apply_command_requires_torque_commands(robot):
    robot.apply_command(JointTorqueCommand(robot.joint_names, np.arange(7.0)))
    np.testing.assert_array_equal(robot.data.ctrl[:7], np.arange(7.0))
    with pytest.raises(TypeError):
        robot.apply_command(JointPositionCommand(robot.joint_names, Q_INIT))
    with pytest.raises(ValueError, match="names/order"):
        robot.apply_command(JointTorqueCommand(robot.joint_names[::-1], np.zeros(7)))


# --------------------------------------------------------------------------------------------------
# Impedance controller
# --------------------------------------------------------------------------------------------------
def test_holding_pose_at_rest_compensates_gravity(robot):
    controller = make_controller(robot)
    qpos, ctrl = robot.data.qpos.copy(), robot.data.ctrl.copy()
    command = controller.compute(TaskSpaceTarget("grasp", robot.get_pose("grasp")), 0.002)
    np.testing.assert_allclose(command.torques, robot.get_nonlinear_forces(), atol=1e-2)
    np.testing.assert_allclose(controller.last_solution.joint_accelerations, 0, atol=1e-2)
    np.testing.assert_array_equal(robot.data.qpos, qpos)
    np.testing.assert_array_equal(robot.data.ctrl, ctrl)


def test_solution_satisfies_dynamics_and_bounds(robot):
    tau_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    ddq_max = np.full(7, 2.0)
    controller = ImpedanceController(robot, ImpedanceQPConfig(
        -ddq_max, ddq_max, torque_lower=-tau_max, torque_upper=tau_max,
    ))
    pose = robot.get_pose("grasp")
    far = TaskSpaceTarget("grasp", Pose(pose.position + [0.3, 0.2, -0.2], pose.quaternion))
    command = controller.compute(far, 0.002)
    ddq = controller.last_solution.joint_accelerations
    assert np.all(np.abs(ddq) <= ddq_max + 1e-4)
    assert np.all(np.abs(command.torques) <= tau_max)
    np.testing.assert_allclose(
        robot.get_mass_matrix() @ ddq + robot.get_nonlinear_forces(), command.torques, atol=1e-3,
    )


def test_desired_acceleration_is_feedforward_plus_pd(robot):
    controller = make_controller(robot, kp_pos=100.0, kp_ori=50.0)
    assert controller.config.kd_pos == pytest.approx(20.0)
    pose = robot.get_pose("grasp")
    acceleration = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.1])
    velocity = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    target = TaskSpaceTarget("grasp", Pose(pose.position + [0.01, 0, 0], pose.quaternion),
                             velocity, acceleration)
    expected = acceleration + np.hstack([100.0 * np.array([0.01, 0, 0]) + 20.0 * velocity[:3],
                                         np.zeros(3)])
    np.testing.assert_allclose(controller.desired_task_acceleration(target), expected, atol=1e-9)


def test_quintic_trajectory_is_tracked_in_closed_loop(robot):
    model, data = robot.model, robot.data
    controller = make_controller(robot)
    start = robot.get_pose("grasp")
    goal_rotation = Rotation.from_rotvec([0.0, 0.0, 0.3]) * Rotation.from_quat(start.quaternion)
    goal = Pose(start.position + [0.05, -0.05, 0.05], goal_rotation.as_quat())
    trajectory = QuinticTrajectory(start, goal, 1.0)
    dt = model.opt.timestep
    worst = 0.0
    for step in range(int(1.5 / dt)):
        sample = trajectory.sample(step * dt)
        robot.apply_command(controller.compute(TaskSpaceTarget.from_sample("grasp", sample), dt))
        mujoco.mj_step(model, data)
        worst = max(worst, np.linalg.norm(robot.get_pose("grasp").position - sample.pose.position))
    final = robot.get_pose("grasp")
    assert worst < 2e-3
    np.testing.assert_allclose(final.position, goal.position, atol=5e-4)
    assert abs(np.dot(final.quaternion, goal.quaternion)) > np.cos(0.01 / 2)


@pytest.mark.parametrize("overrides", [
    {"kp_pos": -1.0},
    {"acceleration_regularization": 0.0},
    {"position_weight": 0.0, "orientation_weight": 0.0},
    {"torque_lower": -np.ones(7)},
    {"posture": np.zeros(3)},
])
def test_invalid_impedance_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        ImpedanceQPConfig(-DDQ_MAX, DDQ_MAX, **overrides)


def test_controller_rejects_wrong_joint_count_and_target_type(robot):
    with pytest.raises(ValueError, match="joint count"):
        ImpedanceController(robot, ImpedanceQPConfig(-np.ones(3), np.ones(3)))
    with pytest.raises(TypeError):
        make_controller(robot).compute(robot.get_pose("grasp"), 0.002)


def test_nullspace_torque_produces_no_task_acceleration(robot):
    q_null = np.array([0.4, 0.0, -0.6, -1.5, 0.0, 1.885, 0.0])
    controller = make_controller(robot, posture_kp=0.0, posture_kd=0.0, nullspace_posture=q_null)
    robot.data.qvel[:7] = np.linspace(-0.3, 0.3, 7)
    mujoco.mj_forward(robot.model, robot.data)
    jacobian, mass = robot.get_jacobian("grasp"), robot.get_mass_matrix()
    state = robot.get_state()
    tau_null = controller.nullspace_torque(jacobian, mass, state.positions, state.velocities)
    assert np.linalg.norm(tau_null) > 1.0
    np.testing.assert_allclose(jacobian @ np.linalg.solve(mass, tau_null), 0, atol=1e-9)


def test_nullspace_posture_moves_redundancy_while_holding_the_hand(robot):
    q_null = np.array([0.4, 0.0, -0.6, -1.5, 0.0, 1.885, 0.0])
    controller = make_controller(robot, posture_kp=0.0, posture_kd=0.0, nullspace_posture=q_null)
    target = TaskSpaceTarget("grasp", robot.get_pose("grasp"))
    q_start = robot.get_state().positions
    for _ in range(1500):
        robot.apply_command(controller.compute(target, 0.002))
        mujoco.mj_step(robot.model, robot.data)
    motion = robot.get_state().positions - q_start
    assert abs(motion[0]) > 0.2 and abs(motion[2]) > 0.2  # Elbow swivel through joints 1 and 3.
    np.testing.assert_allclose(robot.get_pose("grasp").position, target.pose.position, atol=1e-4)
    np.testing.assert_allclose(controller.last_solution.nullspace_torques.shape, (7,))
