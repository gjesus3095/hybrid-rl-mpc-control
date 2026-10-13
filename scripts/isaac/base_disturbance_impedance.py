"""Isaac Lab: the keyboard moves the Ridgeback base (stand-in for the RL base action) while main's
torque-level impedance QP (``controllers/impedance``) keeps the Franka TCP (``panda_fingertip_centered``)
on a world-frame path, with null-space posture torque and a manipulability floor against singularities.

Per 2 ms step (physics and QP, as main's MuJoCo scripts), for all environments:
    1. read the simulator state and dynamics once (GPU -> CPU) and the base command (keyboard or
       ``--schedule``); every 20th step update the markers and render (and record a ``--record`` frame);
    2. solve each environment's QP on the CPU, with the commanded base twist as feedforward: the
       controller only sees the arm joints, so the base-induced TCP velocity/acceleration go into the target;
    3. write base velocity and arm effort targets together (CPU -> GPU), then step physics once.

The arm's implicit PD gains are zeroed (pure effort control) and gravity is on: the QP compensates it.
Commands to run: ``docs/isaac_base_disturbance.md``.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

MAX_ENVS = 16
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=6, help=f"Number of environments (1-{MAX_ENVS}).")
parser.add_argument("--schedule", action="store_true",
                    help="Replace the keyboard by a fixed key sequence: still, +x, -x, +y, -y, +yaw, -yaw (2 s each).")
parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many sim seconds (0: until closed).")
parser.add_argument("--log", type=str, default="logs/isaac_impedance.csv", help="CSV log path.")
parser.add_argument("--record", action="store_true",
                    help="Record all envs offscreen (works headless); with --schedule, stop after one cycle.")
parser.add_argument("--record_width", type=int, default=640, help="Video width in px (16:9).")
parser.add_argument("--record_path", type=str, default="logs/isaac_impedance.mp4", help="mp4 output path.")
add_launcher_args(parser)
args = parser.parse_args()
if args.record:  # not --video: Isaac's launcher reserves that for its own MoviePy recorder
    args.enable_cameras = True  # offscreen rendering needs the camera extensions
if not 1 <= args.num_envs <= MAX_ENVS:
    parser.error(f"--num_envs must be in [1, {MAX_ENVS}]")

import csv
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import clone, configclass, instantiate, replace
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab_assets import RIDGEBACK_FRANKA_PANDA_CFG

from hybrid_control.controllers.impedance.impedance_control import (
    ImpedanceController, ImpedanceQPConfig, TaskSpaceTarget,
)
from hybrid_control.robots.state import JointTorqueCommand, Pose
from hybrid_control.sim_interfaces.isaac.adapter import IsaacDynamicsBuffer, IsaacTorqueRobotModel
from hybrid_control.utils.logger import logger
from hybrid_control.utils.manipulability import manipulability

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
DT = 1 / 500  # physics step = control period = QP dt (main's DT_CTRL)
RENDER_EVERY = 20  # render, update markers and record every 20th step -> 25 fps
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
BASE_JOINTS = ("dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint")
BASE_BODIES = ("dummy_base_x", "dummy_base_y", "base_link", "arm_mount_link", "front_laser", "rear_laser")
BASE_LINK, HAND, TCP = "base_link", "panda_hand", "panda_fingertip_centered"
Q_INIT = (0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4)  # same arm start as the MuJoCo scripts
# Impedance QP, as main's scripts/mujoco/impedance_controller/impedance_keyboard.py
KP_POS, KP_ORI = 400.0, 200.0  # 1/s², critically damped
DDQ_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])  # rad/s²
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])  # N·m, also the simulator effort limits
Q_MID = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.885, 0.0])  # null-space posture
P_NULLSPACE, D_NULLSPACE = 10.0, 1.0  # N·m/rad, N·m·s/rad
# Singularity avoidance (main's README option 3): barrier floor on w = sqrt(det(J Jᵀ)) + joint-limit barriers
MANIPULABILITY_ROWS, MANIPULABILITY_MIN = "full", 0.03
JOINT_LIMIT_MARGIN = 0.05  # rad
KEY_SENSITIVITY = (0.3, 0.3, 0.5)  # vx, vy (m/s), wz (rad/s) while a key is held
SCHEDULE = [(2.0, (0, 0, 0), "still"), (2.0, (1, 0, 0), "+x"), (2.0, (-1, 0, 0), "-x"), (2.0, (0, 1, 0), "+y"),
            (2.0, (0, -1, 0), "-y"), (2.0, (0, 0, 1), "+yaw"), (2.0, (0, 0, -1), "-yaw")]
ENV_SPACING = 2.5  # m
ARROW_GAIN, ARROW_HEAD = 2.0, 0.12  # arrow length in m per m/s; head length in m
VIDEO_SIZE = (args.record_width, args.record_width * 9 // 16)
STATUS_PERIOD = 1.0  # s, console summary


ROBOT_CFG = replace(clone(RIDGEBACK_FRANKA_PANDA_CFG), prim_path="{ENV_REGEX_NS}/Robot")
ROBOT_CFG.init_state.joint_pos.update(dict(zip(ARM_JOINTS, Q_INIT)))
for _group in ("panda_shoulder", "panda_forearm"):  # pure effort control of the arm
    ROBOT_CFG.actuators[_group].stiffness = 0.0
    ROBOT_CFG.actuators[_group].damping = 0.0


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot = ROBOT_CFG


# ==================================================================================================
# HELPERS
# ==================================================================================================
def tcp_offset() -> Pose:
    """``panda_hand -> panda_fingertip_centered`` from Isaac Lab's Factory Franka (franka_mimic.usd)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/Factory/franka_mimic.usd"))
    cache = UsdGeom.XformCache()
    world = {p.GetName(): cache.GetLocalToWorldTransform(p) for p in stage.Traverse() if p.GetName() in (HAND, TCP)}
    relative = world[TCP] * world[HAND].GetInverse()  # Gf row-vector convention: TCP in hand
    rotation = relative.ExtractRotationQuat()
    return Pose(np.array(relative.ExtractTranslation()), np.array([*rotation.GetImaginary(), rotation.GetReal()]))


def yaw(pose: Pose) -> float:
    R = Rotation.from_quat(np.array(pose.quaternion)).as_matrix()
    return float(np.arctan2(R[1, 0], R[0, 0]))


def base_command_world(command: np.ndarray, base_yaw: float) -> np.ndarray:
    """Base-frame (vx, vy, wz) -> base joint velocities (world vx, vy, wz)."""
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    return np.array([c * command[0] - s * command[1], s * command[0] + c * command[1], command[2]])


def base_feedforward(robot: IsaacTorqueRobotModel, base_vel: np.ndarray, target: Pose) -> TaskSpaceTarget:
    """Fixed world target expressed for a controller that only sees the arm joints.

    With the base moving at world (vx, vy, wz) about the base_link z axis, the TCP twist is
    J dq + V_b and its acceleration J ddq + dJ dq + A_b, where (r = TCP - base axis, w = wz ez)
    V_b = [v + w x r; w] and, at constant base velocity, A_b = [w x (w x r) + w x (J_lin dq); 0].
    Tracking zero world twist/acceleration therefore means arm references -V_b and -A_b.
    """
    w = np.array([0.0, 0.0, base_vel[2]])
    r = robot.get_pose(TCP).position - robot.get_pose(BASE_LINK).position
    arm_velocity = robot.get_jacobian(TCP)[:3] @ robot.get_state().velocities
    velocity = np.r_[np.r_[base_vel[:2], 0.0] + np.cross(w, r), w]
    acceleration = np.r_[np.cross(w, np.cross(w, r)) + np.cross(w, arm_velocity), np.zeros(3)]
    return TaskSpaceTarget(TCP, target, -velocity, -acceleration)


def pose_error(target: Pose, pose: Pose) -> tuple[float, float]:
    """Position (m) and rotation-angle (rad) error."""
    rotation = Rotation.from_quat(np.array(target.quaternion)) * Rotation.from_quat(np.array(pose.quaternion)).inv()
    return float(np.linalg.norm(target.position - pose.position)), float(rotation.magnitude())


def marker(cfg, path: str, **sphere):
    cfg = clone(cfg)
    if sphere:
        cfg.markers["sphere"].radius = sphere["radius"]
        cfg.markers["sphere"].visual_material.diffuse_color = sphere["color"]
    if "frame" in cfg.markers:
        cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    return VisualizationMarkers(replace(cfg, prim_path=path))


def arrow_marker(path: str, color) -> VisualizationMarkers:
    """Planar arrow from a unit-length x-cylinder (shaft) and an x-cone (head).

    Isaac's ``arrow_x.usd`` markers render flat and pointing up in this install, so they are not used.
    """
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers={
        "shaft": sim_utils.CylinderCfg(radius=0.02, height=1.0, axis="X", visual_material=material),
        "head": sim_utils.ConeCfg(radius=0.05, height=ARROW_HEAD, axis="X", visual_material=material)}))


def arrows(origins: np.ndarray, directions: np.ndarray, lengths: np.ndarray):
    """Shaft + head poses for ``arrow_marker``: from ``origins`` along ``directions`` (N, 2); hidden for zero length."""
    n = len(origins)
    yaws = np.arctan2(directions[:, 1], directions[:, 0])
    unit = np.c_[np.cos(yaws), np.sin(yaws), np.zeros(n)]
    shaft = np.maximum(lengths - ARROW_HEAD, 0.0)
    positions = np.vstack([origins + 0.5 * shaft[:, None] * unit, origins + (shaft + 0.5 * ARROW_HEAD)[:, None] * unit])
    scales = np.ones((2 * n, 3))
    scales[:n, 0] = shaft
    scales[np.tile(lengths < 1e-3, 2)] = 0.0
    return dict(translations=positions, orientations=np.tile(Rotation.from_euler("z", yaws[:, None]).as_quat(), (2, 1)),
                scales=scales, marker_indices=[0] * n + [1] * n)


def make_camera(num_envs: int):
    """Offscreen RTX camera for --record (after Senne's ridgeback_demo.py).

    Isaac Lab sensors in a cloned scene need one camera per env; all get the same overview pose
    and only env 0's image is used.
    """
    from isaaclab.sensors.camera import Camera, CameraCfg  # imports pxr: only after the app started
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    spawn = sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                       clipping_range=(0.1, 1.0e5))
    for env in range(num_envs):  # explicit paths: a regex spawn path also matches prims inside the robots
        spawn.func(f"/World/envs/env_{env}/RecordCam", spawn)
    return Camera(CameraCfg(
        prim_path="/World/envs/env_.*/RecordCam", update_period=0.0, width=VIDEO_SIZE[0], height=VIDEO_SIZE[1],
        # Per-env scene partitioning hides all env geometry from a camera outside /World/envs/env_i
        data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False), spawn=None))


class VideoRecorder:
    """Views all env origins, burns in a status overlay and writes an H.264 mp4 (imageio-ffmpeg)."""

    LEGEND = (("TCP target / measured (axes)", (255, 255, 255)), ("arm CoM", (26, 204, 26)), ("base CoM", (255, 128, 0)),
              ("base velocity cmd", (255, 0, 0)), ("yaw rate cmd", (0, 0, 255)))

    def __init__(self, camera, origins: np.ndarray, path: str):
        try:  # Isaac Lab's OpenCV has no FFmpeg; imageio-ffmpeg bundles its own binary
            import imageio_ffmpeg
        except ModuleNotFoundError as e:
            raise RuntimeError("--record needs imageio-ffmpeg: uv pip install --python ~/IsaacLab/.venv/bin/python "
                               "imageio-ffmpeg") from e

        self.camera = camera
        center = origins.mean(axis=0) + np.array([0.3, 0.0, 0.5])
        distance = 2.6 + 1.2 * np.ptp(origins[:, :2], axis=0).max()  # the env grid is longest along x
        eye = center + distance * np.array([0.2, -0.8, 0.55])
        camera.set_world_poses_from_view(*(torch.tensor(np.tile(v, (len(origins), 1)), dtype=torch.float32,
                                                        device=camera.device) for v in (eye, center)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # yuv420p H.264 plays in VS Code and browsers; block size 8 keeps 16:9 sizes like 640x360 unscaled
        self.writer = imageio_ffmpeg.write_frames(path, VIDEO_SIZE, fps=1 / (RENDER_EVERY * DT), codec="libx264",
                                                  pix_fmt_out="yuv420p", macro_block_size=8)
        self.writer.send(None)  # start ffmpeg
        self.path = path
        self.frames = 0

    def capture(self, lines: list[str], t: float):
        from PIL import Image, ImageDraw, ImageFont

        self.camera.update(RENDER_EVERY * DT)
        image = Image.fromarray(self.camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8))
        draw = ImageDraw.Draw(image)
        k = VIDEO_SIZE[0] / 1280  # overlay layout is in px at 1280 wide
        font, small = ImageFont.load_default(size=round(24 * k)), ImageFont.load_default(size=round(18 * k))
        outline = dict(stroke_width=max(1, round(2 * k)), stroke_fill=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((16 * k, (14 + 30 * i) * k), line, font=font, fill=(255, 255, 255), **outline)
        for i, (label, color) in enumerate(self.LEGEND):
            y = VIDEO_SIZE[1] - (30 + 26 * (len(self.LEGEND) - i)) * k
            draw.rectangle((VIDEO_SIZE[0] - 330 * k, y + 3 * k, VIDEO_SIZE[0] - 314 * k, y + 19 * k), fill=color, outline=(0, 0, 0))
            draw.text((VIDEO_SIZE[0] - 306 * k, y), label, font=small, fill=(255, 255, 255), **outline)
        draw.text((16 * k, VIDEO_SIZE[1] - 40 * k), f"t = {t:5.2f} s", font=font, fill=(255, 255, 255), **outline)
        self.writer.send(np.ascontiguousarray(image))
        self.frames += 1

    def close(self):
        self.writer.close()
        logger.info("Saved %d frames to %s", self.frames, Path(self.path).resolve())


# ==================================================================================================
# MAIN
# ==================================================================================================
def reset_to_default(sim, scene):
    """Write the configured initial joint state (sim.reset() alone keeps the USD joint state)."""
    articulation = scene["robot"]
    joint_pos = articulation.data.default_joint_pos.torch.clone()
    articulation.write_joint_position_to_sim_index(position=joint_pos)
    articulation.write_joint_velocity_to_sim_index(velocity=articulation.data.default_joint_vel.torch.clone())
    articulation.actuators.target_command.set_position_index(value=joint_pos)
    articulation.reset()
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(DT)


def make_controller(robot: IsaacTorqueRobotModel) -> ImpedanceController:
    return ImpedanceController(robot, ImpedanceQPConfig(
        acceleration_lower=-DDQ_MAX, acceleration_upper=DDQ_MAX, torque_lower=-TAU_MAX, torque_upper=TAU_MAX,
        kp_pos=KP_POS, kp_ori=KP_ORI,
        posture_kp=0.0, posture_kd=0.0,  # the null-space torque handles the posture
        nullspace_posture=Q_MID, nullspace_stiffness=P_NULLSPACE, nullspace_damping=D_NULLSPACE,
        manipulability_rows=MANIPULABILITY_ROWS, manipulability_min=MANIPULABILITY_MIN,
        joint_limit_margin=JOINT_LIMIT_MARGIN,
    ))


def run(sim, scene, camera=None):
    articulation = scene["robot"]
    num_envs = scene.num_envs
    base_ids = [articulation.joint_names.index(n) for n in BASE_JOINTS]

    extra_frames = {TCP: (HAND, tcp_offset())}
    logger.info("TCP offset %s -> %s: %s m", HAND, TCP, np.round(extra_frames[TCP][1].position, 5))
    buffer = IsaacDynamicsBuffer(articulation, ARM_JOINTS, jacobian_bodies=[HAND])
    robots = [IsaacTorqueRobotModel(buffer, env, extra_frames) for env in range(num_envs)]
    controllers = [make_controller(robot) for robot in robots]

    # Fixed world TCP target per env; replace target_fn to track any path
    initial = [r.get_pose(TCP) for r in robots]

    def target_fn(t: float, env: int) -> Pose:
        return initial[env]

    w0 = manipulability(robots[0].get_jacobian(TCP), MANIPULABILITY_ROWS)
    logger.info("Manipulability (%s) at the start: %.4f (floor %.4f); torque limits %s N·m", MANIPULABILITY_ROWS, w0,
                MANIPULABILITY_MIN, np.round(robots[0].torque_limits[1], 1))

    if args.schedule:
        cycle = sum(duration for duration, _, _ in SCHEDULE)

        def read_command(t: float) -> tuple[np.ndarray, str]:
            t, start = t % cycle, 0.0
            for duration, keys, name in SCHEDULE:
                if t < start + duration:
                    break
                start += duration
            return np.multiply(keys, KEY_SENSITIVITY).astype(float), name
    else:
        if not sim.has_gui:
            raise RuntimeError("The keyboard needs the GUI (--viz kit); use --schedule for headless runs")
        keyboard = Se2Keyboard(Se2KeyboardCfg(v_x_sensitivity=KEY_SENSITIVITY[0], v_y_sensitivity=KEY_SENSITIVITY[1],
                                              omega_z_sensitivity=KEY_SENSITIVITY[2]))
        logger.info("%s", keyboard)
        read_command = lambda t: (keyboard.advance().cpu().numpy().astype(float), "keyboard")
    duration = args.duration
    if args.record and args.schedule and duration <= 0:
        duration = cycle  # one cycle of the schedule
    video = VideoRecorder(camera, scene.env_origins.cpu().numpy(), args.record_path) if camera is not None else None

    markers = {
        "target": marker(FRAME_MARKER_CFG, "/Visuals/tcp_target"),
        "tcp": marker(FRAME_MARKER_CFG, "/Visuals/tcp"),
        "arm_com": marker(SPHERE_MARKER_CFG, "/Visuals/arm_com", radius=0.04, color=(0.1, 0.8, 0.1)),
        "base_com": marker(SPHERE_MARKER_CFG, "/Visuals/base_com", radius=0.06, color=(1.0, 0.5, 0.0)),
        "base_vel": arrow_marker("/Visuals/base_vel_cmd", (1.0, 0.0, 0.0)),
        "yaw_rate": arrow_marker("/Visuals/yaw_rate_cmd", (0.0, 0.0, 1.0)),
    }
    arm_bodies = [n for n in articulation.body_names if n not in BASE_BODIES and n != "world"]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["t", "cmd_vx", "cmd_vy", "cmd_wz", "base_x", "base_y", "base_yaw", "pos_err_max_mm",
                  "ori_err_max_mrad", "manipulability_min", "floor_slack_max", "tau_null_max", "saturated_torques",
                  "failed_solves", "t_read_ms", "t_solve_ms", "t_write_ms", "t_sim_ms"])
    logger.info("Logging to %s", log_path.resolve())

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    timings, sim_time, last_status, step = [], 0.0, 0.0, 0
    try:
        while sim.is_running() and (duration <= 0 or sim_time < duration - 1e-9):
            # 1. One GPU -> CPU copy of the state and dynamics that resulted from the previous step
            sync()
            t0 = time.perf_counter()
            buffer.read(sim_time)
            t_read = time.perf_counter() - t0

            tcp = [r.get_pose(TCP) for r in robots]
            errors = np.array([pose_error(target_fn(sim_time, e), tcp[e]) for e in range(num_envs)])
            command, phase = read_command(sim_time)
            yaws = np.array([yaw(robot.get_pose(BASE_LINK)) for robot in robots])
            base_targets = np.array([base_command_world(command, base_yaw) for base_yaw in yaws])

            # 2. The QP of all envs (same sim time for all); a failed env falls back to h (gravity/Coriolis)
            failed, t0 = 0, time.perf_counter()
            for env, (robot, controller) in enumerate(zip(robots, controllers)):
                try:
                    robot.apply_command(controller.compute(base_feedforward(robot, base_targets[env],
                                                                            target_fn(sim_time + DT, env)), DT))
                except RuntimeError:
                    failed += 1
                    robot.apply_command(JointTorqueCommand(ARM_JOINTS, np.clip(
                        robot.get_nonlinear_forces(), *robot.torque_limits)))
            t_solve = time.perf_counter() - t0
            solutions = [c.last_solution for c in controllers]
            w_min = min(s.manipulability for s in solutions if s is not None)
            slack = max(s.manipulability_slack for s in solutions if s is not None)
            tau_null = max(np.abs(s.nullspace_torques).max() for s in solutions if s is not None)
            saturated = int(np.sum(np.abs(buffer.efforts) >= buffer.effort_limits - 1e-6))

            if step % RENDER_EVERY == 0:  # markers show the state that is rendered now
                origins = np.array([robot.get_pose(BASE_LINK).position for robot in robots])
                targets = [target_fn(sim_time, e) for e in range(num_envs)]
                markers["target"].visualize(np.array([p.position for p in targets]),
                                            np.array([p.quaternion for p in targets]))
                markers["tcp"].visualize(np.array([p.position for p in tcp]), np.array([p.quaternion for p in tcp]))
                markers["arm_com"].visualize(np.array([buffer.com(e, arm_bodies) for e in range(num_envs)]))
                markers["base_com"].visualize(np.array([buffer.com(e, BASE_BODIES) for e in range(num_envs)]))
                lift = np.array([0.0, 0.0, 0.35])
                markers["base_vel"].visualize(**arrows(origins + lift, base_targets[:, :2],
                                                       ARROW_GAIN * np.linalg.norm(base_targets[:, :2], axis=1)))
                front = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)  # tangential velocity 0.6 m in front of the base
                tangent = np.sign(base_targets[:, 2:3]) * np.stack([-front[:, 1], front[:, 0]], axis=1)
                markers["yaw_rate"].visualize(**arrows(origins + lift + 0.6 * np.c_[front, np.zeros(num_envs)], tangent,
                                                       ARROW_GAIN * 0.6 * np.abs(base_targets[:, 2])))
                sim.render()
                if video is not None:
                    video.capture([
                        f"Impedance QP (torque, OSQP, {1 / DT:.0f} Hz)   base command: {phase}  "
                        f"[vx {command[0]:+.1f} m/s, vy {command[1]:+.1f} m/s, wz {command[2]:+.1f} rad/s]",
                        f"TCP error, max over {num_envs} envs: {1e3 * errors[:, 0].max():5.1f} mm  "
                        f"{1e3 * errors[:, 1].max():5.1f} mrad",
                        f"QP solve all {num_envs} envs: {1e3 * t_solve:4.2f} ms   manipulability min: {w_min:.3f} "
                        f"(floor {MANIPULABILITY_MIN})   null-space torque max: {tau_null:.2f} N·m"], sim_time)

            # 3. One CPU -> GPU write of all targets, then physics
            sync()
            t0 = time.perf_counter()
            buffer.write()
            articulation.actuators.target_command.set_velocity_index(
                value=torch.as_tensor(base_targets, dtype=torch.float32, device=articulation.device), joint_ids=base_ids)
            sync()
            t_write = time.perf_counter() - t0

            t0 = time.perf_counter()
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(DT)
            sync()
            t_sim = time.perf_counter() - t0

            base0 = robots[0].get_pose(BASE_LINK)
            row = [sim_time, *command, base0.position[0], base0.position[1], yaws[0], 1e3 * errors[:, 0].max(),
                   1e3 * errors[:, 1].max(), w_min, slack, tau_null, saturated, failed,
                   1e3 * t_read, 1e3 * t_solve, 1e3 * t_write, 1e3 * t_sim]
            log.writerow([f"{v:.6g}" for v in row])
            timings.append(row[-4:])
            if sim_time - last_status >= STATUS_PERIOD - 1e-9:
                last = np.array(timings)
                logger.info("t=%.1fs | pos err max %.2f mm, ori %.2f mrad | w min %.3f, slack %.2g, tau_null %.2f, "
                            "saturated %d, failed %d | ms mean/p95: read %.2f/%.2f, solve %.2f/%.2f, write %.2f/%.2f, "
                            "sim %.2f/%.2f", sim_time, *row[7:14],
                            *np.ravel([[last[:, i].mean(), np.percentile(last[:, i], 95)] for i in range(4)]))
                timings, last_status = [], sim_time
            sim_time += DT
            step += 1
    finally:
        log_file.close()
        if video is not None:
            video.close()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=DT, device=args.device)
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([4.0, 4.0, 3.0], [0.0, 0.0, 0.5])
        scene = instantiate(SceneCfg(num_envs=args.num_envs, env_spacing=ENV_SPACING))
        camera = make_camera(args.num_envs) if args.record else None  # must exist before sim.reset()
        sim.reset()
        reset_to_default(sim, scene)
        run(sim, scene, camera)


if __name__ == "__main__":
    main()
