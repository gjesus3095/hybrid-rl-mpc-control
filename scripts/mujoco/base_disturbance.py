"""IK/QP tracking under simultaneous base disturbance and changing arm targets.

Builds a mobile-base Panda by adding two prismatic (slide) joints to link0
in world X and Y. Each waypoint simultaneously commands a new random base position
AND a new random world-frame end-effector target. The base joints are outside the IK
— the arm must reach the world-frame target purely through arm joint motion despite
the base moving underfoot.

Both base waypoints and arm targets are always in the world frame.

Run:
    uv run python scripts/mujoco/base_disturbance.py
Set SHOW_VIEWER = True for the interactive view.
"""

from contextlib import nullcontext
from importlib.resources import files
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hybrid_control.sim_interfaces.mujoco.adapter import MujocoRobotModel
from hybrid_control.sim_interfaces.mujoco.modifier import MujocoModifier
from hybrid_control.controllers.base import Controller
from hybrid_control.controllers.ik_qp.controller import (
    IKQPConfig,
    InverseKinematicsController,
    PoseTarget,
)
from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import JointPositionCommand, Pose
from hybrid_control.utils.logger import logger


# ==================================================================================================
# 1. CONFIGURATION & CONSTANTS — edit experiment parameters here
# ==================================================================================================
MODEL_DIR = files("hybrid_control.sim_interfaces.mujoco.models.panda")
XML_FILE = str(MODEL_DIR / "scene.xml")
SHOW_VIEWER = True
COM_MARKER_DIAMETER = 0.05  # metres; viewer-only sphere, no mass or collisions.

# --- Base motion ---
BASE_MOTION_RADIUS = 0.2        # m, max XY command per axis; scene limits are ±2 m.
BASE_YAW_RADIUS = np.pi / 6     # rad, max yaw command; scene limits are ±π.
N_BASE_POSITIONS = 10           # random base waypoints to test.
BASE_SERVO_KP = 5000.0          # N/m; stiff position servo for X/Y.
BASE_YAW_SERVO_KP = 500.0       # N·m/rad; position servo for yaw.
BASE_RANGE = 2.0                # m; hard joint limit in the model.
BASE_YAW_RANGE = np.pi          # rad; hard joint limit in the model.

# --- Control parameters ---
JOINT_ACTUATOR_PAIRS = tuple((f"joint{i}", f"actuator{i}") for i in range(1, 8))
N_DOFS = len(JOINT_ACTUATOR_PAIRS)
FRAME_CONTROL = "grasp"
DT_CTRL = 0.02
TRACKING_SECONDS = 5.0
DQ_MAX = np.full(N_DOFS, 20.0)
DQ_MIN = -DQ_MAX
KP_POS = 80.0
KP_ORI = 60.0
JOINT_VEL_WEIGHT = 1e5
POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 1.0

# --- Initial arm configuration ---
Q_INIT = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])
FINGER_POSITION_INIT = 0.04
GRIPPER_COMMAND_INIT = 255

# --- Arm targets (world frame, offset from initial hand position) ---
TARGET_RADIUS = 0.05         # m, XYZ sampling box half-width around initial hand pos.
ROTATION_ORDER = "zyx"       # Extrinsic rotations about fixed world Z, Y, X axes.
ORIENTATION_CENTER_DEG = np.array([-90.0, 0.0, 180.0])
ORIENTATION_RANGE_DEG = np.array([40.0, 30.0, 40.0])

# --- Random seeds ---
RANDOM_SEED = 7

_modifier = MujocoModifier(COM_MARKER_DIAMETER)

# --- Acceptance thresholds ---
POSITION_TOLERANCE = 0.005
ORIENTATION_TOLERANCE = 0.03
VELOCITY_SLACK = 2e-3
COMMAND_SLACK = 1e-4
JOINT_SLACK = 1e-3


# ==================================================================================================
# 2. SIMULATION SETUP
# ==================================================================================================
def create_simulation() -> MujocoRobotModel:
    """Compile mobile base model and initialize arm and base to home positions."""
    spec = MujocoModifier.build_mobile_base_spec(
        XML_FILE, base_range=BASE_RANGE, servo_kp=BASE_SERVO_KP,
        yaw_range=BASE_YAW_RANGE, yaw_servo_kp=BASE_YAW_SERVO_KP,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    robot = MujocoRobotModel(
        model, data, JOINT_ACTUATOR_PAIRS,
        base_actuator_names=("base_x_servo", "base_y_servo", "base_yaw_servo"),
    )
    robot.initialize(
        Q_INIT,
        extra_joint_positions={
            "base_x": 0.0,
            "base_y": 0.0,
            "base_yaw": 0.0,
            "finger_joint1": FINGER_POSITION_INIT,
            "finger_joint2": FINGER_POSITION_INIT,
        },
        extra_actuator_controls={
            "actuator8": GRIPPER_COMMAND_INIT,
            "base_x_servo": 0.0,
            "base_y_servo": 0.0,
            "base_yaw_servo": 0.0,
        },
    )
    return robot


def create_controller(robot: RobotModel) -> Controller[PoseTarget, JointPositionCommand]:
    config = IKQPConfig(
        velocity_lower=DQ_MIN, velocity_upper=DQ_MAX,
        kp_pos=KP_POS, kp_ori=KP_ORI, damping=JOINT_VEL_WEIGHT,
        position_weight=POSITION_WEIGHT, orientation_weight=ORIENTATION_WEIGHT,
    )
    return InverseKinematicsController(robot, config)


def set_base_command(robot: MujocoRobotModel, x: float, y: float, yaw: float):
    """Write a clamped base position command [x, y, yaw] via the robot model's base actuators."""
    robot.apply_base_command([
        np.clip(x,   -BASE_MOTION_RADIUS, BASE_MOTION_RADIUS),
        np.clip(y,   -BASE_MOTION_RADIUS, BASE_MOTION_RADIUS),
        np.clip(yaw, -BASE_YAW_RADIUS,    BASE_YAW_RADIUS),
    ])


# ==================================================================================================
# 4. TRACKING & ASSERTIONS
# ==================================================================================================
def assert_within_limits(values, lower, upper, *, slack, label, context):
    values = np.asarray(values)
    lower = np.broadcast_to(lower, values.shape) - slack
    upper = np.broadcast_to(upper, values.shape) + slack
    invalid = ~np.isfinite(values) | (values < lower) | (values > upper)
    details = "; ".join(
        f"joint {i + 1}: {values[i]:.6f}, allowed [{lower[i]:.6f}, {upper[i]:.6f}]"
        for i in np.flatnonzero(invalid)
    )
    assert not np.any(invalid), f"{context}: {label}: {details}"


def compute_checked_command(robot, controller, target, dt, context):
    state = robot.get_state()
    lower, upper = robot.position_limits
    command = controller.compute(target, dt)
    assert command.joint_names == robot.joint_names, (
        f"{context}: command joint order {command.joint_names} "
        f"does not match model joint order {robot.joint_names}"
    )
    requested_velocity = (command.positions - state.positions) / dt
    assert_within_limits(
        requested_velocity, DQ_MIN, DQ_MAX, slack=VELOCITY_SLACK,
        label="IK velocity (rad/s)", context=context,
    )
    assert_within_limits(
        command.positions, lower, upper, slack=COMMAND_SLACK,
        label="commanded position (rad)", context=context,
    )
    return command


def track_reference(robot, controller, target, handle, context):
    model, data = robot.model, robot.data
    q_min, q_max = robot.position_limits
    arm_dofs = np.array([model.joint(name).dofadr[0] for name in robot.joint_names])
    substeps = max(1, round(DT_CTRL / model.opt.timestep))
    dt = substeps * model.opt.timestep
    steps = int(np.ceil(TRACKING_SECONDS / dt))
    start_time = data.time

    for _ in range(steps):
        started = time.perf_counter()
        if handle is not None and not handle.is_running():
            raise RuntimeError("Viewer closed before tracking completed")
        step_context = f"{context}, t={data.time - start_time:.3f} s"
        command = compute_checked_command(robot, controller, target, dt, step_context)
        robot.apply_command(command)
        for _ in range(substeps):
            data.qfrc_applied[arm_dofs] = data.qfrc_bias[arm_dofs]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            step_context = f"{context}, t={data.time - start_time:.3f} s"
            assert np.all(np.isfinite(data.qpos)), (
                f"{step_context}: nonfinite simulation position: {data.qpos}"
            )
            assert_within_limits(
                robot.get_state().positions, q_min, q_max, slack=JOINT_SLACK,
                label="simulated position (rad)", context=step_context,
            )
        if handle is not None:
            _modifier.update_overlay(handle, robot, target.pose)
            handle.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - started)))
    return data.time - start_time


def report_tracking_error(robot, target, context):
    """Report world-frame pose error; target was recorded before any base motion."""
    actual_pose = robot.get_pose(target.frame_name)
    reference = target.pose.position
    position = actual_pose.position
    position_error = float(np.linalg.norm(reference - position))
    cosine = np.clip(abs(np.dot(target.pose.quaternion, actual_pose.quaternion)), 0.0, 1.0)
    orientation_error = float(2 * np.arccos(cosine))
    pos_ok = position_error < POSITION_TOLERANCE
    ori_ok = orientation_error < ORIENTATION_TOLERANCE
    status = "PASS" if (pos_ok and ori_ok) else "FAIL"
    logger.info(
        "[%s] pos=%.2f mm (tol=%.0f mm), ori=%.2f deg (tol=%.1f deg)",
        status,
        position_error * 1000.0,
        POSITION_TOLERANCE * 1000.0,
        np.degrees(orientation_error),
        np.degrees(ORIENTATION_TOLERANCE),
    )
    if not pos_ok:
        logger.warning("Reference=%s, actual=%s", np.round(reference, 4), np.round(position, 4))
    return position_error, orientation_error, pos_ok and ori_ok


# ==================================================================================================
# 5. EXPERIMENT
# ==================================================================================================
def run_base_disturbance(*, viewer=False):
    """Per waypoint: move base to a new random XY position and track a new random world-frame arm target."""
    logger.info(
        "Starting mobile-base disturbance experiment: viewer=%s, waypoints=%s, seed=%s",
        viewer,
        N_BASE_POSITIONS,
        RANDOM_SEED,
    )
    robot = create_simulation()
    controller = create_controller(robot)
    controller.reset()

    initial_hand_pos = robot.get_pose(FRAME_CONTROL).position
    orientation_center = np.asarray(ORIENTATION_CENTER_DEG, dtype=float)
    orientation_range = np.asarray(ORIENTATION_RANGE_DEG, dtype=float)

    rng = np.random.default_rng(RANDOM_SEED)
    base_xy = rng.uniform(-BASE_MOTION_RADIUS, BASE_MOTION_RADIUS, size=(N_BASE_POSITIONS, 2))
    base_yaws = rng.uniform(-BASE_YAW_RADIUS, BASE_YAW_RADIUS, size=N_BASE_POSITIONS)
    base_waypoints = np.column_stack([base_xy, base_yaws])
    # Arm targets: XYZ offsets around the initial world-frame hand position.
    arm_positions = initial_hand_pos + rng.uniform(-TARGET_RADIUS, TARGET_RADIUS, size=(N_BASE_POSITIONS, 3))
    # Orientations: random Euler angles around the center; first and last equal the center.
    angles = np.tile(orientation_center, (N_BASE_POSITIONS, 1))
    angles[1:-1] += rng.uniform(-orientation_range, orientation_range, size=(N_BASE_POSITIONS - 2, 3))
    arm_quaternions = Rotation.from_euler(ROTATION_ORDER, angles, degrees=True).as_quat()

    viewer_context = nullcontext(None)
    if viewer:
        from mujoco import viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(robot.model, robot.data)

    results = []
    with viewer_context as handle:
        for idx, ((wx, wy, wyaw), pos, quat) in enumerate(
            zip(base_waypoints, arm_positions, arm_quaternions), start=1
        ):
            target = PoseTarget(FRAME_CONTROL, Pose(pos, quat))
            context = f"Waypoint {idx}/{N_BASE_POSITIONS} (seed={RANDOM_SEED})"
            logger.info(
                "%s: base=[%.3f, %.3f] m, yaw=%.1f deg, hand=%s m (world frame)",
                context,
                wx,
                wy,
                np.degrees(wyaw),
                np.round(pos, 4),
            )
            set_base_command(robot, wx, wy, wyaw)
            _modifier.update_overlay(handle, robot, target.pose)
            elapsed = track_reference(robot, controller, target, handle, context)
            results.append(report_tracking_error(
                robot, target,
                context=f"{context}, after {elapsed:.3f} s",
            ))

    passed = sum(1 for *_, ok in results if ok)
    logger.info(
        "Summary: %s/%s waypoints within tolerance (pos<%.0f mm, ori<%.1f deg)",
        passed,
        len(results),
        POSITION_TOLERANCE * 1000.0,
        np.degrees(ORIENTATION_TOLERANCE),
    )
    return results


if __name__ == "__main__":
    run_base_disturbance(viewer=SHOW_VIEWER)
