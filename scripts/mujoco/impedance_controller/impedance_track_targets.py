"""Track seeded Cartesian pose targets with the torque-controlled Panda arm in MuJoCo.

Same target generation, acceptance criteria, and assertion structure as
qp_controller/track_targets.py. Replaces the IK/QP position-servo loop
with an impedance (inverse-dynamics QP) torque controller.

Edit the configuration below, then run:
    uv run python scripts/mujoco/impedance_controller/impedance_track_targets.py
Set SHOW_VIEWER = True for the interactive view.
"""

from contextlib import nullcontext
from importlib.resources import files
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hybrid_control.controllers.impedance.impedance_control import (
    ImpedanceController,
    ImpedanceQPConfig,
    TaskSpaceTarget,
)
from hybrid_control.robots.state import Pose
from hybrid_control.sim_interfaces.mujoco.adapter import MujocoTorqueRobotModel
from hybrid_control.sim_interfaces.mujoco.modifier import MujocoModifier
from hybrid_control.utils.logger import logger


# ==================================================================================================
# 1. CONFIGURATION & CONSTANTS — edit experiment parameters here
# ==================================================================================================
MODEL_DIR = files("hybrid_control.sim_interfaces.mujoco.models.panda")
XML_FILE = str(MODEL_DIR / "scene.xml")
SHOW_VIEWER = True
COM_MARKER_DIAMETER = 0.05

# --- Control parameters ---
JOINT_ACTUATOR_PAIRS = tuple((f"joint{i}", f"actuator{i}") for i in range(1, 8))
N_DOFS = len(JOINT_ACTUATOR_PAIRS)
FRAME_CONTROL = "grasp"
DT_CTRL = 0.002
TRACKING_SECONDS = 5.0

KP_POS = 400.0
KP_ORI = 200.0
KD_POS = None  # None → critical damping 2*sqrt(KP)
KD_ORI = None
POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 1.0
ACCELERATION_REGULARIZATION = 1e-4
TORQUE_REGULARIZATION = 1e-6
POSTURE_KP = 10.0
USE_NULLSPACE = True
Q_MID = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.885, 0.0])
P_NULLSPACE = np.full(N_DOFS, 10.0)
D_NULLSPACE = np.full(N_DOFS, 1.0)
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
DDQ_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])

# --- Initial configuration ---
Q_INIT = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])
FINGER_POSITION_INIT = 0.04
GRIPPER_COMMAND_INIT = 255

# --- Position and orientation references ---
RANDOM_SEED = 7
TARGET_RADIUS = 0.15
ROTATION_ORDER = "zyx"
N_TARGETS = 10
ORIENTATION_CENTER_DEG = np.array([-90.0, 0.0, 180.0])
ORIENTATION_RANGE_DEG = np.array([40.0, 30.0, 40.0])

# --- Acceptance thresholds ---
POSITION_TOLERANCE = 0.005   # metres
ORIENTATION_TOLERANCE = 0.03  # radians
TORQUE_SLACK = 1.0            # N·m; accounts for QP solver tolerance at bounds
JOINT_SLACK = 1e-3            # radians; simulated actuator overshoot

_modifier = MujocoModifier(COM_MARKER_DIAMETER)


# ==================================================================================================
# 2. SIMULATION SETUP
# ==================================================================================================
def create_simulation() -> MujocoTorqueRobotModel:
    spec = MujocoModifier.build_torque_spec(XML_FILE, [actuator for _, actuator in JOINT_ACTUATOR_PAIRS])
    model = spec.compile()
    data = mujoco.MjData(model)
    robot = MujocoTorqueRobotModel(model, data, JOINT_ACTUATOR_PAIRS)
    robot.initialize(
        Q_INIT,
        extra_joint_positions={
            "finger_joint1": FINGER_POSITION_INIT,
            "finger_joint2": FINGER_POSITION_INIT,
        },
        extra_actuator_controls={"actuator8": GRIPPER_COMMAND_INIT},
    )
    return robot


def create_controller(robot: MujocoTorqueRobotModel) -> ImpedanceController:
    config = ImpedanceQPConfig(
        acceleration_lower=-DDQ_MAX, acceleration_upper=DDQ_MAX,
        torque_lower=-TAU_MAX, torque_upper=TAU_MAX,
        kp_pos=KP_POS, kd_pos=KD_POS, kp_ori=KP_ORI, kd_ori=KD_ORI,
        position_weight=POSITION_WEIGHT, orientation_weight=ORIENTATION_WEIGHT,
        acceleration_regularization=ACCELERATION_REGULARIZATION,
        torque_regularization=TORQUE_REGULARIZATION,
        posture_kp=0.0 if USE_NULLSPACE else POSTURE_KP,
        posture_kd=0.0 if USE_NULLSPACE else None,
        posture=Q_INIT,
        nullspace_posture=Q_MID if USE_NULLSPACE else None,
        nullspace_stiffness=P_NULLSPACE, nullspace_damping=D_NULLSPACE,
    )
    return ImpedanceController(robot, config)


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


def compute_checked_command(robot, controller, target, dt, context):
    """Compute a torque command and reject nonfinite or out-of-bounds results."""
    # The QP bounds torques to ±TAU_MAX; this assertion layer catches any solver divergence.
    command = controller.compute(target, dt)
    assert command.joint_names == robot.joint_names, (
        f"{context}: command joint order {command.joint_names} "
        f"does not match model joint order {robot.joint_names}"
    )
    assert_within_limits(
        command.torques, -TAU_MAX, TAU_MAX, slack=TORQUE_SLACK,
        label="commanded torque (N·m)", context=context,
    )
    return command


def track_reference(robot, controller, target: TaskSpaceTarget, handle, context: str):
    """Track a fixed regulation target for TRACKING_SECONDS while checking validity."""
    # The impedance controller provides full gravity compensation; no qfrc_applied bias needed.
    model, data = robot.model, robot.data
    q_min, q_max = robot.position_limits
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
            mujoco.mj_step(model, data)
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


def assert_tracking_error(robot, target: TaskSpaceTarget, context: str):
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
        "Starting Panda impedance target-tracking run: viewer=%s, targets=%s, seed=%s",
        viewer, N_TARGETS, RANDOM_SEED,
    )
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
    controller = create_controller(robot)
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
        TaskSpaceTarget(FRAME_CONTROL, Pose(position, quaternion))
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
        viewer_context = mujoco_viewer.launch_passive(robot.model, robot.data)

    with viewer_context as handle:
        for index, target in enumerate(targets, start=1):
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
