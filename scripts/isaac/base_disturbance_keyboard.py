"""Isaac Lab: the keyboard moves the Ridgeback base (stand-in for the RL base action) while the
Pink IK/QP (``controllers/ik_qp``) keeps the Franka TCP (``panda_fingertip_centered``) on a
world-frame pose target.

Per 50 Hz control step (physics 200 Hz, 4 substeps), for all environments:
    1. read the simulator state once (GPU -> CPU) and the base command (keyboard or ``--schedule``);
       update the markers and render every step; record a ``--record`` frame at 25 fps (every 2nd step);
    2. predict each ``panda_link0`` pose after the commanded base motion (the sim is not stepped yet);
    3. solve each environment's QP against the predicted base pose (on the CPU: sequentially or in
       ``--workers`` processes that each handle a slice of the environments);
    4. write base velocity and arm position (+ velocity feedforward) targets together (CPU -> GPU),
       then step physics 4x without rendering.

Arm gravity is disabled, matching Isaac Lab's high-PD Franka config for IK. Run commands:
``docs/isaac_base_disturbance.md``.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

MAX_ENVS = 512
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=16, help=f"Number of environments (1-{MAX_ENVS}).")
parser.add_argument("--solver", choices=("daqp", "osqp"), default="daqp", help="QP backend.")
parser.add_argument("--workers", type=int, default=0,
                    help="IK/QP worker processes, one per physical core (0: solve sequentially in this process).")
parser.add_argument("--schedule", action="store_true",
                    help="Replace the keyboard by a fixed key sequence: still, +x, -x, +y, -y, +yaw, -yaw (2 s each).")
parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many sim seconds (0: until closed).")
parser.add_argument("--log", type=str, default="logs/isaac_base_disturbance.csv", help="CSV log path.")
parser.add_argument("--record", action="store_true",
                    help="Record env 0 offscreen (works headless); with --schedule, stop after one cycle.")
parser.add_argument("--record_width", type=int, default=640, help="Video width in px (16:9).")
parser.add_argument("--record_path", type=str, default="logs/isaac_base_disturbance.mp4",
                    help="mp4 output path (needs imageio-ffmpeg).")
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
import pinocchio as pin
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
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from hybrid_control.controllers.ik_qp.config import IKQPConfig
from hybrid_control.controllers.ik_qp.ik_qp import DEFAULT_URDF, build_model
from hybrid_control.controllers.ik_qp.parallel import ParallelPinkIK, physical_cpus
from hybrid_control.robots.state import Pose
from hybrid_control.sim_interfaces.isaac.adapter import IsaacArticulationBuffer, IsaacRobotModel
from hybrid_control.utils.logger import logger

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================
PHYSICS_DT = 1 / 200
DECIMATION = 4
DT = PHYSICS_DT * DECIMATION  # 50 Hz control period, also the QP dt
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
BASE_JOINTS = ("dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint")
BASE_BODIES = ("dummy_base_x", "dummy_base_y", "base_link", "arm_mount_link", "front_laser", "rear_laser")
BASE_LINK, ARM_ROOT, TCP = "base_link", "panda_link0", "panda_fingertip_centered"
DQ_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # Franka joint velocity limits, rad/s
KP_POS, KP_ORI = 50.0, 100.0  # Pink gains kp_pos*DT = 0.5*kp_ori*DT = 1: full correction of the predicted error per step
DAMPING = 1e5
Q_INIT = (0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4)  # same arm start as the MuJoCo scripts
KEY_SENSITIVITY = (0.3, 0.3, 0.5)  # vx, vy (m/s), wz (rad/s) while a key is held; 0.8 m/s saturates the arm joints
ARROW_GAIN = 2.0  # arrow length in m per m/s
ARROW_HEAD = 0.12  # m
# Generic key presses (after Senne's ridgeback_demo.py BASE_SCHEDULE): (duration s, key multipliers, name)
SCHEDULE = [(2.0, (0, 0, 0), "still"), (2.0, (1, 0, 0), "+x"), (2.0, (-1, 0, 0), "-x"), (2.0, (0, 1, 0), "+y"),
            (2.0, (0, -1, 0), "-y"), (2.0, (0, 0, 1), "+yaw"), (2.0, (0, 0, -1), "-yaw")]
VIDEO_STEP = 2  # record every 2nd control step -> 25 fps
VIDEO_SIZE = (args.record_width, args.record_width * 9 // 16)
VIDEO_EYE, VIDEO_TARGET = (2.2, -2.2, 1.6), (0.3, 0.0, 0.5)  # relative to the env 0 origin
STATUS_PERIOD = 1.0  # s, console summary


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot = replace(clone(RIDGEBACK_FRANKA_PANDA_CFG), prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.rigid_props = PhysxRigidBodyCfg(disable_gravity=True)
    robot.init_state.joint_pos.update(dict(zip(ARM_JOINTS, Q_INIT)))


# ==================================================================================================
# HELPERS
# ==================================================================================================
def tcp_offset() -> Pose:
    """``panda_hand -> panda_fingertip_centered`` from Isaac Lab's Factory Franka (franka_mimic.usd)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/Factory/franka_mimic.usd"))
    cache = UsdGeom.XformCache()
    world = {p.GetName(): cache.GetLocalToWorldTransform(p) for p in stage.Traverse()
             if p.GetName() in ("panda_hand", TCP)}
    relative = world[TCP] * world["panda_hand"].GetInverse()  # Gf row-vector convention: TCP in hand
    rotation = relative.ExtractRotationQuat()
    return Pose(np.array(relative.ExtractTranslation()), np.array([*rotation.GetImaginary(), rotation.GetReal()]))


def matrix(pose: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_quat(np.array(pose.quaternion)).as_matrix(), pose.position
    return T


def matrices(poses: np.ndarray) -> np.ndarray:
    """(N, 7) poses (position, xyzw) -> (N, 4, 4) homogeneous transforms."""
    T = np.tile(np.eye(4), (len(poses), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = Rotation.from_quat(poses[:, 3:]).as_matrix(), poses[:, :3]
    return T


def compose_batch(parent: np.ndarray, offset: Pose) -> np.ndarray:
    """World poses (N, 7) of ``offset`` expressed in each of the ``parent`` poses (N, 7)."""
    rotation = Rotation.from_quat(parent[:, 3:])
    return np.c_[parent[:, :3] + rotation.apply(np.array(offset.position)),
                 (rotation * Rotation.from_quat(np.array(offset.quaternion))).as_quat()]


def yaws_of(poses: np.ndarray) -> np.ndarray:
    R = Rotation.from_quat(poses[:, 3:]).as_matrix()
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def predict_after_base_motion(base: np.ndarray, frame: np.ndarray, base_vel: np.ndarray) -> np.ndarray:
    """World poses (N, 7) of ``frame`` after the base joints (world x, y, then yaw) move by base_vel*DT.

    T' = Trans(dx, dy) @ B @ Rz(dyaw) @ B^-1 @ T, with B the current base_link pose; base, frame
    (N, 7) world poses (xyzw), base_vel (N, 3).
    """
    B = matrices(base)
    delta = np.tile(np.eye(4), (len(base), 1, 1))
    delta[:, :3, :3] = Rotation.from_euler("z", base_vel[:, 2:3] * DT).as_matrix()
    T = B @ delta @ np.linalg.inv(B) @ matrices(frame)
    T[:, :2, 3] += base_vel[:, :2] * DT
    return np.c_[T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()]


def base_command_world(command: np.ndarray, base_yaws: np.ndarray) -> np.ndarray:
    """Base-frame (vx, vy, wz) -> base joint velocities (N, 3): world vx, vy, wz."""
    c, s = np.cos(base_yaws), np.sin(base_yaws)
    return np.stack([c * command[0] - s * command[1], s * command[0] + c * command[1],
                     np.full_like(base_yaws, command[2])], axis=1)


def pose_errors(target: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Position (m) and rotation-angle (rad) errors (N, 2) between (N, 7) poses."""
    rotation = Rotation.from_quat(target[:, 3:]) * Rotation.from_quat(pose[:, 3:]).inv()
    return np.c_[np.linalg.norm(target[:, :3] - pose[:, :3], axis=1), rotation.magnitude()]


class PinocchioTcp:
    """URDF forward kinematics of the TCP in the ``panda_link0`` frame (prediction/consistency checks)."""

    def __init__(self, extra_frames):
        self.model = build_model(DEFAULT_URDF, extra_frames)
        self.data = self.model.createData()
        self.idx_q = [self.model.joints[self.model.getJointId(n)].idx_q for n in ARM_JOINTS]
        self.root, self.tcp = self.model.getFrameId(ARM_ROOT), self.model.getFrameId(TCP)

    def in_root(self, q: np.ndarray) -> np.ndarray:
        q_pin = pin.neutral(self.model)
        q_pin[self.idx_q] = q
        pin.framesForwardKinematics(self.model, self.data, q_pin)
        return self.data.oMf[self.root].actInv(self.data.oMf[self.tcp]).homogeneous


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


def record_camera_cfg():
    """Offscreen RTX camera for --record (after Senne's ridgeback_demo.py), one per env as a scene sensor:
    Isaac Lab sizes sensors by the env count, so a single /World camera fails with num_envs > 1. Only env 0
    is recorded."""
    from isaaclab.sensors.camera import CameraCfg  # imports pxr: only after the app started
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/RecordCam", update_period=0.0, width=VIDEO_SIZE[0], height=VIDEO_SIZE[1],
        # Per-env scene partitioning hides all env geometry from a camera outside /World/envs/env_i
        data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.1, 1.0e5)))


class VideoRecorder:
    """Grabs camera frames, burns in a status overlay and writes an H.264 mp4 (imageio-ffmpeg)."""

    LEGEND = (("TCP target / measured (axes)", (255, 255, 255)), ("arm CoM", (26, 204, 26)), ("base CoM", (255, 128, 0)),
              ("base velocity cmd", (255, 0, 0)), ("yaw rate cmd", (0, 0, 255)))

    def __init__(self, camera, origins: np.ndarray, path: str):
        try:  # Isaac Lab's OpenCV has no FFmpeg; imageio-ffmpeg bundles its own binary
            import imageio_ffmpeg
        except ModuleNotFoundError as e:
            raise RuntimeError("--record needs imageio-ffmpeg: uv pip install --python ~/IsaacLab/.venv/bin/python "
                               "imageio-ffmpeg") from e

        self.camera = camera
        eye, target = (torch.tensor(origins + np.array(v), dtype=torch.float32, device=camera.device)
                       for v in (VIDEO_EYE, VIDEO_TARGET))  # every env's camera on the same view of its env
        camera.set_world_poses_from_view(eye, target)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # yuv420p H.264 plays in VS Code and browsers; block size 8 keeps 16:9 sizes like 640x360 unscaled
        self.writer = imageio_ffmpeg.write_frames(path, VIDEO_SIZE, fps=1 / (VIDEO_STEP * DT), codec="libx264",
                                                  pix_fmt_out="yuv420p", macro_block_size=8)
        self.writer.send(None)  # start ffmpeg
        self.path = path
        self.frames = 0

    def capture(self, t, phase, command, error_env0, solve_env0, solve_all, num_envs):
        from PIL import Image, ImageDraw, ImageFont

        self.camera.update(VIDEO_STEP * DT)
        image = Image.fromarray(self.camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8))
        draw = ImageDraw.Draw(image)
        k = VIDEO_SIZE[0] / 1280  # overlay layout is in px at 1280 wide
        font, small = ImageFont.load_default(size=round(24 * k)), ImageFont.load_default(size=round(18 * k))
        outline = dict(stroke_width=max(1, round(2 * k)), stroke_fill=(0, 0, 0))
        lines = [f"Pink IK/QP ({args.solver})   base command: {phase}  "
                 f"[vx {command[0]:+.1f} m/s, vy {command[1]:+.1f} m/s, wz {command[2]:+.1f} rad/s]",
                 f"TCP error env 0: {1e3 * error_env0[0]:5.1f} mm  {1e3 * error_env0[1]:5.1f} mrad",
                 f"QP solve: env 0 {1e3 * solve_env0:4.2f} ms   all {num_envs} envs {1e3 * solve_all:4.2f} ms"]
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
    scene.update(PHYSICS_DT)


def run(sim, scene, ik: ParallelPinkIK, camera=None):
    articulation = scene["robot"]
    num_envs = scene.num_envs
    base_ids = [articulation.joint_names.index(n) for n in BASE_JOINTS]

    extra_frames = {TCP: ("panda_hand", tcp_offset())}
    logger.info("TCP offset panda_hand -> %s: %s m", TCP, np.round(extra_frames[TCP][1].position, 5))
    buffer = IsaacArticulationBuffer(articulation, ARM_JOINTS)
    robot0 = IsaacRobotModel(buffer, 0, extra_frames)  # start-up checks only; the loop works on (N, ...) arrays
    body = {name: buffer.body_ids[name] for name in (BASE_LINK, ARM_ROOT, "panda_hand")}
    config = IKQPConfig(-DQ_MAX, DQ_MAX, kp_pos=KP_POS, kp_ori=KP_ORI, damping=DAMPING)
    ik.configure(ARM_JOINTS, config, root_frame=ARM_ROOT, solver=args.solver, extra_frames=extra_frames)
    logger.info("IK/QP: %s", f"{ik.num_workers} worker processes" if ik.num_workers else "sequential")
    fk = PinocchioTcp(extra_frames)

    def world_poses(name: str) -> np.ndarray:
        """(N, 7) world poses (position, xyzw) of a body or the TCP in the current buffer."""
        if name == TCP:
            return compose_batch(buffer.body_pose[:, body["panda_hand"]], extra_frames[TCP][1])
        return buffer.body_pose[:, body[name]].astype(float)

    # Fixed world TCP target per env; replace target_fn to track any path
    initial = world_poses(TCP)

    def target_fn(t: float) -> np.ndarray:
        """(N, 7) world TCP targets at time t."""
        return initial

    # URDF vs simulator kinematics (logged, not enforced)
    T_sim = np.linalg.inv(matrix(robot0.get_pose(ARM_ROOT))) @ matrix(robot0.get_pose(TCP))
    T_urdf = fk.in_root(robot0.get_state().positions)
    logger.info("URDF vs USD TCP in panda_link0 (env 0): %.2f mm, %.2f mrad",
                1e3 * np.linalg.norm(T_sim[:3, 3] - T_urdf[:3, 3]),
                1e3 * Rotation.from_matrix(T_sim[:3, :3].T @ T_urdf[:3, :3]).magnitude())
    q_min, q_max = robot0.position_limits
    logger.info("Simulator arm limits used by the QP: lower %s, upper %s", np.round(q_min, 3), np.round(q_max, 3))

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
    log.writerow(["t", "num_envs", "workers", "cmd_vx", "cmd_vy", "cmd_wz", "base_x", "base_y", "base_yaw",
                  "pos_err_max_mm", "ori_err_max_mrad", "pred_err_max_mm", "active_bounds", "failed_solves",
                  "t_read_ms", "t_solve_ms", "t_solve_worker_max_ms", "t_write_ms", "t_sim_ms", "t_step_ms"])
    logger.info("Logging to %s", log_path.resolve())

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    predicted_tcp = None  # (N, 3) TCP positions predicted by the previous step's QP
    timings, sim_time, last_status, step = [], 0.0, 0.0, 0
    t_solve, solve_env0 = 0.0, 0.0
    t_step_start = time.perf_counter()
    try:
        while sim.is_running() and (duration <= 0 or sim_time < duration):
            # 1. One GPU -> CPU copy of the state that resulted from the previous step
            sync()
            t0 = time.perf_counter()
            buffer.read(sim_time)
            t_read = time.perf_counter() - t0

            # Errors of the previous step: tracking and one-step prediction
            tcp, base, link0 = world_poses(TCP), world_poses(BASE_LINK), world_poses(ARM_ROOT)
            targets = target_fn(sim_time)
            errors = pose_errors(targets, tcp)
            pred_err = np.nan if predicted_tcp is None else np.linalg.norm(tcp[:, :3] - predicted_tcp, axis=1).max()

            command, phase = read_command(sim_time)
            yaws = yaws_of(base)
            base_targets = base_command_world(command, yaws)

            # Markers show the state that is rendered now (before this step's QP and physics)
            origins = base[:, :3]
            markers["target"].visualize(targets[:, :3], targets[:, 3:])
            markers["tcp"].visualize(tcp[:, :3], tcp[:, 3:])
            markers["arm_com"].visualize(buffer.coms(arm_bodies))
            markers["base_com"].visualize(buffer.coms(BASE_BODIES))
            lift = np.array([0.0, 0.0, 0.35])
            markers["base_vel"].visualize(**arrows(origins + lift, base_targets[:, :2],
                                                   ARROW_GAIN * np.linalg.norm(base_targets[:, :2], axis=1)))
            front = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)  # tangential velocity 0.6 m in front of the base
            tangent = np.sign(base_targets[:, 2:3]) * np.stack([-front[:, 1], front[:, 0]], axis=1)
            markers["yaw_rate"].visualize(**arrows(origins + lift + 0.6 * np.c_[front, np.zeros(num_envs)], tangent,
                                                   ARROW_GAIN * 0.6 * np.abs(base_targets[:, 2])))
            sim.render()
            if video is not None and step % VIDEO_STEP == 0:
                video.capture(sim_time, phase, command, errors[0], solve_env0, t_solve, num_envs)

            # 2.-3. Predicted base pose and the QP of all envs (same sim time for all)
            link0_next = predict_after_base_motion(base, link0, base_targets)
            q = buffer.joint_pos[:, buffer.joint_ids]
            t0 = time.perf_counter()
            q_cmd, ok = ik.compute(q, buffer.joint_pos_limits, link0_next, target_fn(sim_time + DT), TCP, DT)
            t_solve = time.perf_counter() - t0
            solve_env0 = t_solve / num_envs
            failed = int(np.sum(~ok))
            q_cmd[~ok] = buffer.targets[~ok]  # hold the previous position targets of failed envs
            buffer.targets[:] = q_cmd
            buffer.velocity_targets[:] = np.where(ok[:, None], (q_cmd - q) / DT, 0.0)
            q_min, q_max = buffer.joint_pos_limits[..., 0], buffer.joint_pos_limits[..., 1]
            active = int(np.sum(np.abs(q_cmd - q) / DT >= DQ_MAX - 1e-6)
                         + np.sum(np.minimum(q_cmd - q_min, q_max - q_cmd) <= 1e-6))
            tcp_in_link0 = np.array([fk.in_root(q_env)[:, 3] for q_env in q_cmd])  # (N, 4) homogeneous
            predicted_tcp = np.einsum("nij,nj->ni", matrices(link0_next), tcp_in_link0)[:, :3]

            # 4. One CPU -> GPU write of all targets, then physics
            sync()
            t0 = time.perf_counter()
            buffer.write()
            articulation.actuators.target_command.set_velocity_index(
                value=torch.as_tensor(base_targets, dtype=torch.float32, device=articulation.device), joint_ids=base_ids)
            sync()
            t_write = time.perf_counter() - t0

            t0 = time.perf_counter()
            for substep in range(DECIMATION):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(PHYSICS_DT)
            sync()
            t_sim = time.perf_counter() - t0

            t_step = time.perf_counter() - t_step_start
            t_step_start = time.perf_counter()
            row = [sim_time, num_envs, ik.num_workers, *command, base[0, 0], base[0, 1], yaws[0],
                   1e3 * errors[:, 0].max(), 1e3 * errors[:, 1].max(), 1e3 * pred_err, active, failed,
                   1e3 * t_read, 1e3 * t_solve, 1e3 * ik.worker_times.max(), 1e3 * t_write, 1e3 * t_sim, 1e3 * t_step]
            log.writerow([f"{v:.6g}" for v in row])
            timings.append(row[-6:-1])
            if sim_time - last_status >= STATUS_PERIOD:
                last = np.array(timings)
                logger.info("t=%.1fs | pos err max %.2f mm, ori %.2f mrad, pred %.2f mm | active %d, failed %d | "
                            "ms mean/p95: read %.2f/%.2f, solve %.2f/%.2f (env max %.2f), write %.2f/%.2f, sim %.2f/%.2f",
                            sim_time, *row[9:14], *np.ravel([[last[:, i].mean(), np.percentile(last[:, i], 95)]
                                                             for i in (0, 1)]), last[:, 2].max(),
                            *np.ravel([[last[:, i].mean(), np.percentile(last[:, i], 95)] for i in (3, 4)]))
                timings, last_status = [], sim_time
            sim_time += DT
            step += 1
    finally:
        log_file.close()
        if video is not None:
            video.close()


def main():
    # Fork the IK workers before Isaac Sim starts (forking the running simulator can deadlock);
    # pin them to physical cores, leaving core 0 for the simulator's main thread
    cpus = physical_cpus()[1:]
    ik = ParallelPinkIK(args.workers, cpus=cpus if len(cpus) >= args.workers else None)
    try:
        _main(ik)
    finally:
        ik.close()


def _main(ik):
    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device)
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([4.0, 4.0, 3.0], [0.0, 0.0, 0.5])
        scene_cfg = SceneCfg(num_envs=args.num_envs, env_spacing=4.0)
        if args.record:  # must exist before sim.reset()
            scene_cfg.record_camera = record_camera_cfg()
        scene = instantiate(scene_cfg)
        camera = scene["record_camera"] if args.record else None
        sim.reset()
        reset_to_default(sim, scene)
        run(sim, scene, ik, camera)


if __name__ == "__main__":
    main()
