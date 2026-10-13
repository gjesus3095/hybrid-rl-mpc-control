"""Drive the torque-controlled Panda into its stretched-arm (workspace-boundary) singularity.

A quintic reference moves the hand in a straight line from the home pose to a
point beyond the arm's reach, holds it there, and comes back. Near full extension
the Jacobian loses rank in the reach direction, so the impedance QP can no longer
realize the desired task acceleration. The run logs how close the arm gets to the
singularity (smallest singular value and condition number of J), the tracking
error, and how hard the joint acceleration and torque bounds are hit.

Because of the Panda's 82.5 mm elbow offsets, the arm is geometrically straight
(maximum shoulder-wrist distance) at q4 ≈ -0.467 rad, not at q4 = 0. Joint angles
between that and the -0.07 rad limit bend the elbow the other way, so the arm can
pass through the singularity into a flipped-elbow branch. The QP has no joint
position limits, so in that branch it may push joint 4 into its limit and stall.

The robot setup is shared with ``impedance_tracking.py``. Edit the configuration
below, then run:
    uv run python scripts/mujoco/impedance_singularity.py
"""

from contextlib import nullcontext
import time

import mujoco
import numpy as np

from hybrid_control.controllers.impedance.impedance_control import (
    ImpedanceController,
    ImpedanceQPConfig,
    TaskSpaceTarget,
)
from hybrid_control.robots.state import Pose
from hybrid_control.utils.logger import logger
from hybrid_control.utils.trajectories import QuinticTrajectory
from impedance_tracking import (
    DDQ_MAX,
    FRAME_CONTROL,
    Q_INIT,
    TAU_MAX,
    create_simulation,
    draw_reference,
    pose_error,
)


# ==================================================================================================
# 1. CONFIGURATION & CONSTANTS — edit experiment parameters here
# ==================================================================================================
SHOW_VIEWER = False  # Set True for the interactive view; a red sphere marks the reference.
SAVE_PLOT = "impedance_singularity.png"  # None disables the plot.
DT_CTRL = 0.002

# --- Reference: home -> REACH_TARGET (beyond reach) -> hold -> home ---
REACH_TARGET = np.array([1.10, 0.0, 0.40])  # m, world frame; > ~0.85 m from the shoulder is unreachable.
MOVE_SECONDS = 3.0  # Quintic duration out and back.
HOLD_SECONDS = 2.0  # Time spent pulling at the unreachable target.

# --- Controller (same structure as impedance_tracking.py) ---
KP_POS = 400.0
KP_ORI = 200.0
POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 1.0
# The Tikhonov weight on ddq is what keeps the QP well-posed at the singularity.
# Try 1e-6 (almost none) vs. 1e-2 (strong damping) to see its effect.
ACCELERATION_REGULARIZATION = 1e-4
TORQUE_REGULARIZATION = 1e-6
POSTURE_KP = 10.0

NEAR_SINGULAR_CONDITION = 100.0  # Report time spent with cond(J) above this.
ELBOW_STRAIGHT_Q4 = -0.467  # rad; q4 where shoulder, elbow and wrist are collinear.


# ==================================================================================================
# 2. EXPERIMENT
# ==================================================================================================
def create_controller(robot) -> ImpedanceController:
    return ImpedanceController(robot, ImpedanceQPConfig(
        acceleration_lower=-DDQ_MAX, acceleration_upper=DDQ_MAX,
        torque_lower=-TAU_MAX, torque_upper=TAU_MAX,
        kp_pos=KP_POS, kp_ori=KP_ORI,
        position_weight=POSITION_WEIGHT, orientation_weight=ORIENTATION_WEIGHT,
        acceleration_regularization=ACCELERATION_REGULARIZATION,
        torque_regularization=TORQUE_REGULARIZATION,
        posture_kp=POSTURE_KP, posture=Q_INIT,
    ))


def build_segments(home: Pose):
    """Out to the unreachable target, hold, and back, keeping the home orientation."""
    far = Pose(REACH_TARGET, home.quaternion)
    return [
        ("out", QuinticTrajectory(home, far, MOVE_SECONDS), MOVE_SECONDS),
        ("hold", QuinticTrajectory(far, far, HOLD_SECONDS), HOLD_SECONDS),
        ("back", QuinticTrajectory(far, home, MOVE_SECONDS), MOVE_SECONDS + 1.0),
    ]


def run(*, viewer=False):
    robot = create_simulation()
    model, data = robot.model, robot.data
    controller = create_controller(robot)
    substeps = max(1, round(DT_CTRL / model.opt.timestep))
    dt = substeps * model.opt.timestep
    home = robot.get_pose(FRAME_CONTROL)
    q_lower, q_upper = robot.position_limits
    shoulder = data.xpos[model.body("link2").id].copy()
    logger.info(
        "Singularity run: target %s m is %.3f m from the shoulder; Tikhonov λ_ddq=%g",
        REACH_TARGET, np.linalg.norm(REACH_TARGET - shoulder), ACCELERATION_REGULARIZATION,
    )

    keys = ("time", "reference", "actual", "error", "sigma_min", "condition",
            "torque_ratio", "ddq_ratio", "q4", "q4_margin", "desired_norm", "achieved_norm")
    log = {key: [] for key in keys}
    viewer_context = nullcontext(None)
    if viewer:
        from mujoco import viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, data)

    with viewer_context as handle:
        for name, trajectory, duration in build_segments(home):
            for step in range(int(np.ceil(duration / dt))):
                started = time.perf_counter()
                if handle is not None and not handle.is_running():
                    raise RuntimeError("Viewer closed before the run completed")
                sample = trajectory.sample(step * dt)
                jacobian = robot.get_jacobian(FRAME_CONTROL)
                singular_values = np.linalg.svd(jacobian, compute_uv=False)
                try:
                    command = controller.compute(TaskSpaceTarget.from_sample(FRAME_CONTROL, sample), dt)
                except RuntimeError as error:
                    logger.error("QP failed at t=%.3f s (%s): %s", data.time, name, error)
                    raise
                solution = controller.last_solution
                dq = robot.get_state().velocities
                achieved = jacobian @ solution.joint_accelerations + (
                    robot.get_jacobian_derivative(FRAME_CONTROL) @ dq
                )
                robot.apply_command(command)
                for _ in range(substeps):
                    mujoco.mj_step(model, data)

                q = robot.get_state().positions
                log["time"].append(data.time)
                log["reference"].append(sample.pose.position)
                log["actual"].append(robot.get_pose(FRAME_CONTROL).position)
                log["error"].append(pose_error(robot, sample.pose)[0])
                log["sigma_min"].append(singular_values[-1])
                log["condition"].append(singular_values[0] / singular_values[-1])
                log["torque_ratio"].append(np.max(np.abs(command.torques) / TAU_MAX))
                log["ddq_ratio"].append(np.max(np.abs(solution.joint_accelerations) / DDQ_MAX))
                log["q4"].append(q[3])
                log["q4_margin"].append(q_upper[3] - q[3])
                log["desired_norm"].append(np.linalg.norm(solution.desired_task_acceleration[:3]))
                log["achieved_norm"].append(np.linalg.norm(achieved[:3]))
                if handle is not None:
                    draw_reference(handle, sample.pose.position)
                    handle.sync()
                    time.sleep(max(0.0, dt - (time.perf_counter() - started)))

            index = slice(-int(np.ceil(duration / dt)), None)
            logger.info(
                "%-4s: final error %6.1f mm | min sigma %.4f | max cond %7.1f | "
                "peak |tau|/max %.2f | peak |ddq|/max %.2f | min joint-4 margin %.3f rad",
                name, log["error"][-1] * 1000, min(log["sigma_min"][index]),
                max(log["condition"][index]), max(log["torque_ratio"][index]),
                max(log["ddq_ratio"][index]), min(log["q4_margin"][index]),
            )

    near = np.array(log["condition"]) > NEAR_SINGULAR_CONDITION
    flipped = np.array(log["q4"]) > ELBOW_STRAIGHT_Q4
    flips = np.flatnonzero(np.diff(flipped.astype(int)))
    logger.info(
        "Elbow crossed straight (q4=%.3f rad) %d times, first at t=%s s; ends %s",
        ELBOW_STRAIGHT_Q4, len(flips),
        f"{log['time'][flips[0]]:.2f}" if len(flips) else "-",
        "FLIPPED (q4 > straight)" if flipped[-1] else "in the normal elbow branch",
    )
    logger.info("cond(J) > %.0f for %.2f s; return-to-home error %.2f mm",
                NEAR_SINGULAR_CONDITION, near.sum() * dt, log["error"][-1] * 1000)
    if SAVE_PLOT:
        save_plot(log, SAVE_PLOT)
    return log


def save_plot(log, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(log["time"])
    reference, actual = np.array(log["reference"]), np.array(log["actual"])
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 10))
    for i, axis in enumerate("xyz"):
        axes[0].plot(t, reference[:, i], "--", color=f"C{i}", label=f"{axis} ref")
        axes[0].plot(t, actual[:, i], color=f"C{i}", label=axis)
    axes[0].set_ylabel("hand position (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(t, np.array(log["error"]) * 1000)
    axes[1].set_ylabel("position error (mm)")
    axes[2].semilogy(t, log["condition"], label="cond(J)")
    axes[2].semilogy(t, 1 / np.array(log["sigma_min"]), label="1 / σ_min(J)")
    axes[2].axhline(NEAR_SINGULAR_CONDITION, color="gray", linestyle=":")
    axes[2].set_ylabel("singularity measure")
    axes[2].legend(fontsize=8)
    axes[3].plot(t, log["torque_ratio"], label="max |τ| / τ_max")
    axes[3].plot(t, log["ddq_ratio"], label="max |ddq| / ddq_max")
    axes[3].axhline(1.0, color="gray", linestyle=":")
    axes[3].set_ylabel("bound usage")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    logger.info("Saved plot to %s", path)


if __name__ == "__main__":
    run(viewer=SHOW_VIEWER)
