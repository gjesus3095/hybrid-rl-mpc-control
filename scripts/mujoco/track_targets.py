"""Track seeded Cartesian pose targets with the Panda arm in MuJoCo.

Edit the configuration below, then run:
    uv run python scripts/mujoco/track_targets.py
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
# The MuJoCo model is loaded from the packaged Panda assets; the XML is the single source of truth.
MODEL_DIR = files("hybrid_control.sim_interfaces.mujoco.models.panda")
XML_FILE = str(MODEL_DIR / "scene.xml")
SHOW_VIEWER = True  # Set True for the interactive view.
COM_MARKER_DIAMETER = 0.05  # metres; viewer-only marker, not part of the dynamics model.

# --- Control parameters ---
# We drive the 7 arm joints directly by matching each MuJoCo joint with its corresponding actuator.
JOINT_ACTUATOR_PAIRS = tuple((f"joint{i}", f"actuator{i}") for i in range(1, 8))
N_DOFS = len(JOINT_ACTUATOR_PAIRS)
FRAME_CONTROL = "grasp"  # End-effector frame used for Cartesian tracking.
DT_CTRL = 0.02  # Requested controller period (s), rounded to whole physics steps.
TRACKING_SECONDS = 5.0  # Simulated time spent tracking each target before validation.
DQ_MAX = np.full(N_DOFS, 20.0)  # rad/s; conservative upper joint-rate bound.
DQ_MIN = -DQ_MAX
KP_POS = 80.0  # Positional CLIK gain in the Cartesian error term.
KP_ORI = 60.0  # Orientation CLIK gain.
JOINT_VEL_WEIGHT = 1e5  # QP regularization strength (damping / lambda-like term).
POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 1.0

# --- Initial configuration ---
# Start from a standard Panda home pose; the gripper is opened before targeting begins.
Q_INIT = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])
FINGER_POSITION_INIT = 0.04  # metres; finger spread for a neutral open gripper.
GRIPPER_COMMAND_INIT = 255  # Fully open; actuator range is 0–255.

# --- Position and orientation references ---
# Targets are generated around the current end-effector pose, with a small randomized orientation field.
RANDOM_SEED = 7
TARGET_RADIUS = 0.15  # XYZ sampling box half-width per world axis (m).
ROTATION_ORDER = "zyx"  # Extrinsic rotations about fixed world axes: Z, Y, X.
N_TARGETS = 10  # Includes the start and end orientation references.
ORIENTATION_CENTER_DEG = np.array([-90.0, 0.0, 180.0])  # Absolute Z, Y, X angles.
ORIENTATION_RANGE_DEG = np.array([40.0, 30.0, 40.0])  # Uniform ±offset per Euler angle.

# --- Acceptance thresholds ---
# These tolerances are intentionally strict enough to catch tracking or convergence issues.
POSITION_TOLERANCE = 0.005  # metres
ORIENTATION_TOLERANCE = 0.03  # radians
VELOCITY_SLACK = 2e-3  # rad/s; accounts for OSQP solver tolerance.
COMMAND_SLACK = 1e-4  # radians
JOINT_SLACK = 1e-3  # radians; simulated actuator overshoot.

_modifier = MujocoModifier(COM_MARKER_DIAMETER)


# ==================================================================================================
# 2. SIMULATION SETUP
# ==================================================================================================
def create_simulation() -> MujocoRobotModel:
    """Load the robot and set a consistent initial state for the tracked arm."""
    # The adapter validates the arm mapping before we start stepping the MuJoCo model.
    robot = MujocoRobotModel.from_xml_path(XML_FILE, JOINT_ACTUATOR_PAIRS)
    robot.initialize(
        Q_INIT,
        extra_joint_positions={
            "finger_joint1": FINGER_POSITION_INIT,
            "finger_joint2": FINGER_POSITION_INIT,
        },
        extra_actuator_controls={"actuator8": GRIPPER_COMMAND_INIT},
    )
    return robot


def create_controller(robot: RobotModel) -> Controller[PoseTarget, JointPositionCommand]:
    """Build the IK/QP controller from the experiment tuning and the shared model API."""
    # The model supplies joint order and limits; the config only sets task-space gains and bounds.
    config = IKQPConfig(
        velocity_lower=DQ_MIN, velocity_upper=DQ_MAX,
        kp_pos=KP_POS, kp_ori=KP_ORI, damping=JOINT_VEL_WEIGHT,
        position_weight=POSITION_WEIGHT, orientation_weight=ORIENTATION_WEIGHT,
    )
    return InverseKinematicsController(robot, config)


# ==================================================================================================
# 3. TRACKING & ASSERTIONS
# ==================================================================================================
def assert_within_limits(values, lower, upper, *, slack, label, context):
    """Check a vector against given bounds and print the first offending entries."""
    # The slack lets us account for numerical tolerances from the solver and physics stepping.
    values = np.asarray(values)
    lower = np.broadcast_to(lower, values.shape) - slack
    upper = np.broadcast_to(upper, values.shape) + slack
    invalid = ~np.isfinite(values) | (values < lower) | (values > upper)
    details = "; ".join(
        f"joint {i + 1}: {values[i]:.6f}, allowed [{lower[i]:.6f}, {upper[i]:.6f}]"
        for i in np.flatnonzero(invalid)
    )
    assert not np.any(invalid), f"{context}: {label}: {details}"


def compute_checked_command(
    robot: RobotModel,
    controller: Controller[PoseTarget, JointPositionCommand],
    target: PoseTarget,
    dt: float,
    context: str,
) -> JointPositionCommand:
    """Compute a valid IK target and reject invalid commands before they hit MuJoCo."""
    # The solver should produce a feasible joint update; this assertion layer prevents silent drift.
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


def track_reference(
    robot: MujocoRobotModel,
    controller: Controller[PoseTarget, JointPositionCommand],
    target: PoseTarget,
    handle,
    context: str,
):
    """Track a target for the configured duration while checking command and state validity."""
    # The simulation is stepped in fixed substeps so the control period matches the requested dt.
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
            # Bias the arm torques to keep the controlled joints near the commanded pose at each step.
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


def assert_tracking_error(robot: RobotModel, target: PoseTarget, context: str):
    """Measure final Cartesian tracking error and fail if the target was not reached."""
    # Report both translational and rotational error in a compact, human-friendly form.
    actual_pose = robot.get_pose(target.frame_name)
    reference = target.pose.position
    position = actual_pose.position
    position_error = float(np.linalg.norm(reference - position))
    cosine = np.clip(abs(np.dot(target.pose.quaternion, actual_pose.quaternion)), 0.0, 1.0)
    orientation_error = float(2 * np.arccos(cosine))
    logger.info(
        "Tracking error: %.2f mm, %.2f deg",
        position_error * 1000.0,
        np.degrees(orientation_error),
    )
    failures = []
    if not position_error < POSITION_TOLERANCE:
        failures.append(
            f"position error {position_error:.6f} m must be < "
            f"{POSITION_TOLERANCE:.6f} m "
            f"(reference={reference.round(6)}, actual={position.round(6)})"
        )
    if not orientation_error < ORIENTATION_TOLERANCE:
        failures.append(
            f"orientation error {orientation_error:.6f} rad must be < "
            f"{ORIENTATION_TOLERANCE:.6f} rad"
        )
    assert not failures, f"{context}: " + "; ".join(failures)
    return position_error, orientation_error


# ==================================================================================================
# 4. EXPERIMENT
# ==================================================================================================
def run_tracking(*, viewer=False, change_orientation=True):
    """Generate target poses, then track each one while validating final accuracy."""
    logger.info(
        "Starting Panda IK/QP target-tracking run: viewer=%s, targets=%s, seed=%s",
        viewer,
        N_TARGETS,
        RANDOM_SEED,
    )
    # Fail early on impossible settings so the experiment is self-checking before the sim starts.
    if not np.isfinite(TRACKING_SECONDS) or TRACKING_SECONDS <= 0:
        raise ValueError("TRACKING_SECONDS must be finite and positive")
    if not np.isfinite(DT_CTRL) or DT_CTRL <= 0:
        raise ValueError("DT_CTRL must be finite and positive")
    if not np.isfinite(TARGET_RADIUS) or TARGET_RADIUS < 0:
        raise ValueError("TARGET_RADIUS must be finite and nonnegative")
    if not isinstance(N_TARGETS, int) or N_TARGETS < 2:
        raise ValueError("N_TARGETS must be an integer >= 2")
    orientation_center = np.asarray(ORIENTATION_CENTER_DEG, dtype=float)
    orientation_range = np.asarray(ORIENTATION_RANGE_DEG, dtype=float)
    for name, values in (
        ("ORIENTATION_CENTER_DEG", orientation_center),
        ("ORIENTATION_RANGE_DEG", orientation_range),
    ):
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain three finite angles")
    if np.any(orientation_range < 0):
        raise ValueError("ORIENTATION_RANGE_DEG must be nonnegative")

    robot = create_simulation()
    model, data = robot.model, robot.data
    controller = create_controller(robot)
    controller.reset()
    initial_pose = robot.get_pose(FRAME_CONTROL)
    center = initial_pose.position
    rng = np.random.default_rng(RANDOM_SEED)
    references = center + rng.uniform(-TARGET_RADIUS, TARGET_RADIUS, size=(N_TARGETS, 3))
    angles = np.tile(orientation_center, (N_TARGETS, 1))
    angles[1:-1] += rng.uniform(-orientation_range, orientation_range, size=(N_TARGETS - 2, 3))
    if change_orientation:
        quaternions = Rotation.from_euler(ROTATION_ORDER, angles, degrees=True).as_quat()
    else:
        quaternions = np.tile(initial_pose.quaternion, (N_TARGETS, 1))
    targets = [
        PoseTarget(FRAME_CONTROL, Pose(position, quaternion))
        for position, quaternion in zip(references, quaternions)
    ]

    logger.info(
        "Experiment seed=%s; initial joint positions (rad): %s",
        RANDOM_SEED,
        np.round(robot.get_state().positions, 4),
    )
    results = []
    viewer_context = nullcontext(None)
    if viewer:
        from mujoco import viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, data)

    with viewer_context as handle:
        for index, target in enumerate(targets, start=1):
            # Each target is tracked independently to keep runtime and failure diagnostics readable.
            context = f"Target {index}/{len(targets)} (seed={RANDOM_SEED})"
            orientation_label = (
                f"{ROTATION_ORDER}={angles[index - 1].round(2)} deg"
                if change_orientation else "initial orientation"
            )
            context = f"{context}, {orientation_label}"
            logger.info("%s: target position=%s m", context, np.round(target.pose.position, 4))
            _modifier.update_overlay(handle, robot, target.pose)
            elapsed = track_reference(robot, controller, target, handle, context)
            results.append(assert_tracking_error(
                robot, target,
                context=f"{context}, after {elapsed:.3f} s of tracking",
            ))
    return results


if __name__ == "__main__":
    run_tracking(viewer=SHOW_VIEWER, change_orientation=True)
