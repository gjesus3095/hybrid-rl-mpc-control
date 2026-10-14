"""Interactive base disturbance: arrow keys move the base X/Y, space/AltGr rotate about Z, while the impedance controller tracks a fixed world-frame EE target.

Controls:
    ← →      base X  (left / right)
    ↑ ↓      base Y  (forward / back)
    Space    base Yaw CCW (+Z)
    AltGr    base Yaw CW  (-Z)

The EE target is fixed at the initial grasp pose in world frame.
The arm compensates for every base displacement through joint torques.

Run:
    uv run python scripts/mujoco/impedance_controller/impedance_keyboard.py
"""

from importlib.resources import files
import time

import mujoco
import mujoco.viewer as mujoco_viewer
import numpy as np

from hybrid_control.controllers.impedance.impedance_control import ImpedanceController
from hybrid_control.utils.control_utils import ImpedanceQPConfig, TaskSpaceTarget
from hybrid_control.sim_interfaces.mujoco.adapter import MujocoRobotModel
from hybrid_control.sim_interfaces.mujoco.modifier import MujocoModifier
from hybrid_control.utils.logger import logger


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
MODEL_DIR = files("hybrid_control.sim_interfaces.mujoco.models.panda")
XML_FILE = str(MODEL_DIR / "scene.xml")
COM_MARKER_DIAMETER = 0.05

FRAME_CONTROL = "grasp"
DT_CTRL = 0.002
JOINT_ACTUATOR_PAIRS = tuple((f"joint{i}", f"actuator{i}") for i in range(1, 8))
N_DOFS = len(JOINT_ACTUATOR_PAIRS)
KP_POS = 400.0
KP_ORI = 200.0
KD_POS = None
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

Q_INIT = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])
FINGER_POSITION_INIT = 0.04
GRIPPER_COMMAND_INIT = 255

BASE_VEL = 0.8          # m/s reference speed while XY key is held
BASE_YAW_VEL = 1.0     # rad/s reference speed while space is held
BASE_RANGE = 2.0        # hard joint limit in the model
BASE_YAW_RANGE = np.pi  # hard yaw joint limit in the model
BASE_SERVO_KP = 5000.0
BASE_YAW_SERVO_KP = 500.0
STATUS_LOG_PERIOD = 1.0  # seconds; print a compact base/EE status line periodically

# GLFW key codes
_KEY_RIGHT, _KEY_LEFT, _KEY_UP, _KEY_DOWN = 262, 263, 265, 264
_KEY_SPACE = 32
_KEY_ALT_GR = 346
# GLFW fires REPEAT events at ~30 Hz while a key is held.
# Track the last event time per key; treat the key as held while within this window.
_KEY_HELD_TIMEOUT = 0.12  # seconds (~4× the 33 ms repeat interval)

_modifier = MujocoModifier(COM_MARKER_DIAMETER)


# ==================================================================================================
# SETUP
# ==================================================================================================
def _create_simulation() -> MujocoRobotModel:
    spec = MujocoModifier.build_mobile_base_spec(
        XML_FILE, base_range=BASE_RANGE, servo_kp=BASE_SERVO_KP,
        yaw_range=BASE_YAW_RANGE, yaw_servo_kp=BASE_YAW_SERVO_KP,
    )
    MujocoModifier.convert_to_torque_actuators(spec, [actuator for _, actuator in JOINT_ACTUATOR_PAIRS])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = MujocoRobotModel(
        model, data, JOINT_ACTUATOR_PAIRS, control_mode="torque",
        base_actuator_names=("base_x_servo", "base_y_servo", "base_yaw_servo"),
    )
    robot.initialize(
        Q_INIT,
        extra_joint_positions={
            "base_x": 0.0, "base_y": 0.0, "base_yaw": 0.0,
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


def _create_controller(robot: MujocoRobotModel) -> ImpedanceController:
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
# MAIN
# ==================================================================================================
def run():
    logger.info("Starting interactive base disturbance demo with impedance EE tracking.")
    robot = _create_simulation()
    controller = _create_controller(robot)

    # Fixed world-frame EE target at the initial arm pose.
    target = TaskSpaceTarget(FRAME_CONTROL, robot.get_pose(FRAME_CONTROL))

    model, data = robot.model, robot.data
    substeps = max(1, round(DT_CTRL / model.opt.timestep))
    dt = substeps * model.opt.timestep

    base_ref = np.zeros(3)  # [x, y, yaw]
    _nav_keys = (_KEY_RIGHT, _KEY_LEFT, _KEY_UP, _KEY_DOWN, _KEY_SPACE, _KEY_ALT_GR)
    _last_key_time: dict[int, float] = {}
    last_status_log = 0.0

    def key_callback(keycode: int) -> None:
        if keycode in _nav_keys:
            _last_key_time[keycode] = time.monotonic()

    def _held(keycode: int, now: float) -> bool:
        return now - _last_key_time.get(keycode, 0.0) < _KEY_HELD_TIMEOUT

    logger.info("Launching impedance base disturbance — fixed EE target, keyboard-controlled base.")
    logger.info("  ← / →    base X (left / right)  @ %.1f m/s", BASE_VEL)
    logger.info("  ↑ / ↓    base Y (fwd / back)    @ %.1f m/s", BASE_VEL)
    logger.info("  Space    base Yaw CCW (+Z)       @ %.1f rad/s", BASE_YAW_VEL)
    logger.info("  AltGr    base Yaw CW  (-Z)       @ %.1f rad/s", BASE_YAW_VEL)
    logger.info("Close viewer window to exit.")

    with mujoco_viewer.launch_passive(model, data, key_callback=key_callback) as handle:
        while handle.is_running():
            started = time.perf_counter()
            now = time.monotonic()

            if _held(_KEY_RIGHT, now):
                base_ref[0] = np.clip(base_ref[0] + BASE_VEL * dt, -BASE_RANGE, BASE_RANGE)
            if _held(_KEY_LEFT, now):
                base_ref[0] = np.clip(base_ref[0] - BASE_VEL * dt, -BASE_RANGE, BASE_RANGE)
            if _held(_KEY_UP, now):
                base_ref[1] = np.clip(base_ref[1] + BASE_VEL * dt, -BASE_RANGE, BASE_RANGE)
            if _held(_KEY_DOWN, now):
                base_ref[1] = np.clip(base_ref[1] - BASE_VEL * dt, -BASE_RANGE, BASE_RANGE)
            if _held(_KEY_SPACE, now):
                base_ref[2] = np.clip(base_ref[2] + BASE_YAW_VEL * dt, -BASE_YAW_RANGE, BASE_YAW_RANGE)
            if _held(_KEY_ALT_GR, now):
                base_ref[2] = np.clip(base_ref[2] - BASE_YAW_VEL * dt, -BASE_YAW_RANGE, BASE_YAW_RANGE)

            robot.apply_base_command(base_ref)
            command = controller.compute(target, dt)
            robot.apply_command(command)

            for _ in range(substeps):
                mujoco.mj_step(model, data)

            now_log = time.monotonic()
            if now_log - last_status_log >= STATUS_LOG_PERIOD:
                actual_pose = robot.get_pose(FRAME_CONTROL)
                bx = data.qpos[model.joint("base_x").qposadr[0]]
                by = data.qpos[model.joint("base_y").qposadr[0]]
                byaw = data.qpos[model.joint("base_yaw").qposadr[0]]
                pos_error = float(np.linalg.norm(target.pose.position - actual_pose.position))
                cosine = np.clip(abs(np.dot(target.pose.quaternion, actual_pose.quaternion)), 0.0, 1.0)
                ori_error = float(2.0 * np.arccos(cosine))
                message = "base=(%.3f, %.3f) m yaw=%.1f° | ee_pos_err=%.4f m | ee_ori_err=%.3f°"
                log = logger.warning if (pos_error > 5e-3 or np.degrees(ori_error) > 1.0) else logger.info
                log(message, bx, by, np.degrees(byaw), pos_error, np.degrees(ori_error))
                last_status_log = now_log

            _modifier.update_overlay(handle, robot, target.pose)
            handle.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - started)))


if __name__ == "__main__":
    run()
