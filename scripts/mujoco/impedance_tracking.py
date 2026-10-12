"""Track a Cartesian trajectory with the torque-controlled Panda arm in MuJoCo.

The Panda's position servos are converted to torque motors, and the impedance
(inverse-dynamics QP) controller sends joint torques every physics step.

Edit the configuration below, then run:
    uv run python scripts/mujoco/impedance_tracking.py
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
from hybrid_control.utils.trajectories import QuinticTrajectory, SinusoidalTrajectory


# ==================================================================================================
# 1. CONFIGURATION & CONSTANTS — edit experiment parameters here
# ==================================================================================================
MODEL_DIR = files("hybrid_control.sim_interfaces.mujoco.models.panda")
XML_FILE = str(MODEL_DIR / "scene.xml")
SHOW_VIEWER = True  # Set True for the interactive view; a red sphere marks the reference.
SAVE_PLOT = None  # e.g. "impedance_tracking.png" to save reference vs. actual position.

# --- Control parameters ---
JOINT_ACTUATOR_PAIRS = tuple((f"joint{i}", f"actuator{i}") for i in range(1, 8))
N_DOFS = len(JOINT_ACTUATOR_PAIRS)
FRAME_CONTROL = "grasp"
DT_CTRL = 0.002  # Controller period (s); one torque update per 500 Hz physics step.
KP_POS = 400.0  # 1/s²; translational stiffness of the unit-mass impedance.
KP_ORI = 200.0  # 1/s²; rotational stiffness.
KD_POS = None  # 1/s; None selects critical damping 2*sqrt(KP).
KD_ORI = None
POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 1.0
ACCELERATION_REGULARIZATION = 1e-4  # Tikhonov weight on ddq (towards the posture reference).
TORQUE_REGULARIZATION = 1e-6  # Tikhonov weight on tau.
POSTURE_KP = 10.0  # 1/s²; QP posture pull towards Q_INIT (ignored when USE_NULLSPACE is True).
# --- Null-space posture torque, added after the QP: tau += N (P (q_mid - q) - D dq) ---
# N = I - Jᵀ Λ J M⁻¹ with Λ = (J M⁻¹ Jᵀ)⁻¹, so the torque only drives the redundant DoF.
USE_NULLSPACE = True
Q_MID = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.885, 0.0])  # rad; null-space posture target.
P_NULLSPACE = np.full(N_DOFS, 10.0)  # N·m/rad
D_NULLSPACE = np.full(N_DOFS, 1.0)  # N·m·s/rad
# Franka Panda datasheet limits: joint torque (N·m) and joint acceleration (rad/s²).
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
DDQ_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])

# --- Initial configuration ---
Q_INIT = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])
FINGER_POSITION_INIT = 0.04
GRIPPER_COMMAND_INIT = 255

# --- Reference trajectory ---
TRAJECTORY = "quintic"  # "quintic" (seeded waypoints) or "sine" (per-axis oscillation).
RANDOM_SEED = 7
# Quintic: rest-to-rest segments between random waypoints around the start pose.
N_WAYPOINTS = 5
WAYPOINT_RADIUS = 0.10  # m; XYZ sampling half-width.
WAYPOINT_ROTATION_DEG = 20.0  # Uniform ±offset per world axis (rotation vector, degrees).
SEGMENT_SECONDS = 2.0  # Quintic duration per segment.
SETTLE_SECONDS = 0.5  # Hold at each waypoint before checking the final error.
# Sine: p(t) = start + A sin(2π f t), orientation held.
SINE_AMPLITUDE = np.array([0.0, 0.10, 0.05])  # m
SINE_FREQUENCY_HZ = np.array([0.0, 0.5, 1.0])
SINE_SECONDS = 6.0

# --- Acceptance thresholds ---
POSITION_TOLERANCE = 0.005  # m, at the end of each settle phase.
ORIENTATION_TOLERANCE = 0.03  # rad
TRACKING_TOLERANCE = 0.02  # m, maximum position error while moving.


# ==================================================================================================
# 2. SIMULATION SETUP
# ==================================================================================================
def create_simulation() -> MujocoTorqueRobotModel:
    """Load the Panda with torque motors on the arm joints and set the initial state."""
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
        # The projected torque replaces the QP's posture term, so the two do not fight.
        posture_kp=0.0 if USE_NULLSPACE else POSTURE_KP,
        posture_kd=0.0 if USE_NULLSPACE else None,
        posture=Q_INIT,
        nullspace_posture=Q_MID if USE_NULLSPACE else None,
        nullspace_stiffness=P_NULLSPACE, nullspace_damping=D_NULLSPACE,
    )
    return ImpedanceController(robot, config)


def build_segments(start: Pose):
    """Return a list of (trajectory, duration) segments for the configured reference."""
    if TRAJECTORY == "sine":
        return [(SinusoidalTrajectory(start, SINE_AMPLITUDE, SINE_FREQUENCY_HZ), SINE_SECONDS)]
    if TRAJECTORY != "quintic":
        raise ValueError("TRAJECTORY must be 'quintic' or 'sine'")
    rng = np.random.default_rng(RANDOM_SEED)
    start_rotation = Rotation.from_quat(start.quaternion)
    waypoints = [start]
    for _ in range(N_WAYPOINTS):
        offset = rng.uniform(-WAYPOINT_RADIUS, WAYPOINT_RADIUS, 3)
        rotvec = np.radians(rng.uniform(-WAYPOINT_ROTATION_DEG, WAYPOINT_ROTATION_DEG, 3))
        rotation = Rotation.from_rotvec(rotvec) * start_rotation
        waypoints.append(Pose(start.position + offset, rotation.as_quat()))
    waypoints.append(start)  # Return home.
    return [
        (QuinticTrajectory(a, b, SEGMENT_SECONDS), SEGMENT_SECONDS + SETTLE_SECONDS)
        for a, b in zip(waypoints[:-1], waypoints[1:])
    ]


# ==================================================================================================
# 3. TRACKING
# ==================================================================================================
def pose_error(robot, pose: Pose) -> tuple[float, float]:
    actual = robot.get_pose(FRAME_CONTROL)
    position_error = float(np.linalg.norm(pose.position - actual.position))
    cosine = np.clip(abs(np.dot(pose.quaternion, actual.quaternion)), 0.0, 1.0)
    return position_error, float(2 * np.arccos(cosine))


def draw_reference(handle, position) -> None:
    with handle.lock():
        handle.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            handle.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, 0.012), position, np.eye(3).ravel(),
            np.array([1.0, 0.1, 0.1, 0.8], dtype=np.float32),
        )


def run_tracking(*, viewer=False):
    robot = create_simulation()
    model, data = robot.model, robot.data
    controller = create_controller(robot)
    substeps = max(1, round(DT_CTRL / model.opt.timestep))
    dt = substeps * model.opt.timestep
    segments = build_segments(robot.get_pose(FRAME_CONTROL))
    logger.info(
        "Starting Panda impedance run: trajectory=%s, segments=%d, control %.0f Hz, seed=%s",
        TRAJECTORY, len(segments), 1 / dt, RANDOM_SEED,
    )

    log = {"time": [], "reference": [], "actual": [], "torque": []}
    failures = []
    max_torque_ratio = 0.0
    max_nullspace_torque = 0.0
    viewer_context = nullcontext(None)
    if viewer:
        from mujoco import viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, data)

    with viewer_context as handle:
        for index, (trajectory, duration) in enumerate(segments, start=1):
            context = f"Segment {index}/{len(segments)}"
            max_tracking_error = 0.0
            for step in range(int(np.ceil(duration / dt))):
                started = time.perf_counter()
                if handle is not None and not handle.is_running():
                    raise RuntimeError("Viewer closed before tracking completed")
                sample = trajectory.sample(step * dt)
                command = controller.compute(TaskSpaceTarget.from_sample(FRAME_CONTROL, sample), dt)
                robot.apply_command(command)
                for _ in range(substeps):
                    mujoco.mj_step(model, data)

                max_torque_ratio = max(max_torque_ratio, float(np.max(np.abs(command.torques) / TAU_MAX)))
                if controller.last_solution.nullspace_torques is not None:
                    max_nullspace_torque = max(
                        max_nullspace_torque, float(np.max(np.abs(controller.last_solution.nullspace_torques))),
                    )
                position_error, _ = pose_error(robot, sample.pose)
                max_tracking_error = max(max_tracking_error, position_error)
                log["time"].append(data.time)
                log["reference"].append(sample.pose.position)
                log["actual"].append(robot.get_pose(FRAME_CONTROL).position)
                log["torque"].append(command.torques)
                if handle is not None:
                    draw_reference(handle, sample.pose.position)
                    handle.sync()
                    time.sleep(max(0.0, dt - (time.perf_counter() - started)))

            final_position, final_orientation = pose_error(robot, sample.pose)
            ok = (final_position < POSITION_TOLERANCE and final_orientation < ORIENTATION_TOLERANCE
                  and max_tracking_error < TRACKING_TOLERANCE)
            logger.info(
                "[%s] %s: final %.2f mm, %.2f deg; max while moving %.2f mm",
                "PASS" if ok else "FAIL", context, final_position * 1000,
                np.degrees(final_orientation), max_tracking_error * 1000,
            )
            if not ok:
                failures.append(context)

    logger.info("Peak |tau| / tau_max = %.2f; peak |tau_null| = %.2f N·m (null space %s); "
                "%d/%d segments passed", max_torque_ratio, max_nullspace_torque,
                "on" if USE_NULLSPACE else "off", len(segments) - len(failures), len(segments))
    q_final = robot.get_state().positions
    logger.info("Final q = %s; distance to %s = %.3f rad", np.round(q_final, 3),
                "Q_MID" if USE_NULLSPACE else "Q_INIT",
                np.linalg.norm(q_final - (Q_MID if USE_NULLSPACE else Q_INIT)))
    if SAVE_PLOT:
        save_plot(log, SAVE_PLOT)
    assert not failures, f"Tracking failed for: {', '.join(failures)}"
    return log


def save_plot(log, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(log["time"])
    reference, actual, torque = (np.array(log[key]) for key in ("reference", "actual", "torque"))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    for i, axis in enumerate("xyz"):
        axes[0].plot(t, reference[:, i], "--", color=f"C{i}", label=f"{axis} ref")
        axes[0].plot(t, actual[:, i], color=f"C{i}", label=f"{axis}")
    axes[0].set_ylabel("position (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(t, torque)
    axes[1].set_ylabel("torque (N·m)")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    logger.info("Saved plot to %s", path)


if __name__ == "__main__":
    run_tracking(viewer=SHOW_VIEWER)
