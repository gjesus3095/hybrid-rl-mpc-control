"""Ridgeback + Franka demo: drive the mobile base and move the arm (Isaac Lab 3.0 / Isaac Sim 6.x).

Port of our Isaac Lab 2.3 demo (itself a fixed version of Isaac Lab's
``source/isaaclab/test/assets/check_ridgeback_franka.py``). Two lessons carried over:

1. Never address joints by hard-coded index: the simulator orders joints its own way
   (e.g. ``panda_joint2`` can be index 0). Every joint is looked up by *name* here.
2. In Isaac Sim 5.1 the Ridgeback-Franka USD had no fixed joint anchoring the dummy ``world``
   link, so base commands moved the dummy links instead of the base. The 6.x USD that Isaac Lab
   3.0 loads already contains that anchor (``/panda_mobile/world/FixedJoint``), so the stock
   ``RIDGEBACK_FRANKA_PANDA_CFG`` is used unchanged. The world-frame base pose is printed so
   you can verify the base really moves.

Usage (from the Isaac Lab 3.0 environment, in this project folder):

    <IsaacLab>/isaaclab.sh -p scripts/isaac/with_controller/ridgeback_demo.py --viz kit     # Linux, GUI
    <IsaacLab>\\isaaclab.bat -p scripts\\isaac\\with_controller\\ridgeback_demo.py --viz kit  # Windows, GUI
    (drop ``--viz kit`` for a headless run)
"""

# ---------------------------------------------------------------------------
# 1. Parse launcher arguments. The simulator itself is started in main() by launch_simulation().
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Drive a Ridgeback-Franka mobile manipulator.")
parser.add_argument("--base_speed", type=float, default=0.5, help="Base speed [m/s] and turn rate [rad/s].")
parser.add_argument("--video", action="store_true", help="Record one cycle offscreen to an mp4 and exit.")
parser.add_argument("--video_path", type=str, default="ridgeback_demo.mp4", help="Output path of the video.")
parser.add_argument("--video_fps", type=int, default=30, help="Video frame rate [Hz].")
add_launcher_args(parser)  # adds --device, --viz, --physics, ...
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True  # offscreen rendering needs the camera extensions (works headless)

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils import clone

from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG  # isort:skip

# Base motion schedule: (duration [s], vx [m/s], vy [m/s], wz [rad/s], description)
S = args_cli.base_speed
BASE_SCHEDULE = [
    (2.0, 0.0, 0.0, 0.0, "standing still"),
    (2.0, S, 0.0, 0.0, "driving forward (+x)"),
    (2.0, -S, 0.0, 0.0, "driving backward (-x)"),
    (2.0, 0.0, S, 0.0, "sliding left (+y)"),
    (2.0, 0.0, -S, 0.0, "sliding right (-y)"),
    (2.0, 0.0, 0.0, S, "turning counter-clockwise"),
    (2.0, 0.0, 0.0, -S, "turning clockwise"),
]


def design_scene() -> tuple[Articulation, "Camera | None"]:
    """Ground plane, a light, one robot and (with --video) a recording camera."""
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/defaultGroundPlane", ground)
    light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    # Copy Isaac Lab's config (don't modify the shared original).
    robot_cfg = clone(RIDGEBACK_FRANKA_PANDA_CFG)
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=robot_cfg)

    camera = None
    if args_cli.video:
        # Imported here, not at module level: camera.py imports ``pxr``, which must only load
        # after launch_simulation() has started Kit.
        from isaaclab.sensors.camera import Camera, CameraCfg

        camera = Camera(
            cfg=CameraCfg(
                prim_path="/World/RecordCam",
                update_period=1.0 / args_cli.video_fps,
                height=720,
                width=1280,
                data_types=["rgb"],
                renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                ),
            )
        )
    return robot, camera


def run_simulator(sim: SimulationContext, robot: Articulation, camera: "Camera | None" = None):
    # -- Look up joint indices BY NAME (preserve_order keeps the order we ask for).
    base_ids, base_names = robot.find_joints(
        ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"],
        preserve_order=True,
    )
    arm_ids, arm_names = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
    finger_ids, _ = robot.find_joints("panda_finger_joint.*")
    base_body = robot.find_bodies("base_link")[0][0]

    print("[INFO] Joint order inside the simulator:", robot.joint_names)
    print("[INFO] Base joints :", dict(zip(base_names, base_ids)))
    print("[INFO] Arm joints  :", dict(zip(arm_names, arm_ids)))

    sim_dt = sim.get_physics_dt()
    # Isaac Lab 3.0: data fields are ProxyArrays -> `.torch` gives a torch tensor.
    default_arm_pos = robot.data.default_joint_pos.torch[:, arm_ids].clone()
    cycle_time = sum(step[0] for step in BASE_SCHEDULE)
    t = 0.0
    last_phase = None
    frames = []
    step_count = 0
    frame_interval = max(1, round(1.0 / (args_cli.video_fps * sim_dt)))
    if camera is not None:
        # Same view as sim.set_camera_view in main().
        camera.set_world_poses_from_view(
            torch.tensor([[3.5, 3.5, 2.5]], device=robot.device), torch.tensor([[0.0, 0.0, 0.5]], device=robot.device)
        )

    while sim.is_running():
        # -- Reset robot at the start of every cycle.
        if t >= cycle_time and camera is not None:
            break  # --video: one full cycle recorded
        if t >= cycle_time or t == 0.0:
            t = 0.0
            robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
            robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
            robot.reset()
            print("[INFO] Reset robot")

        # -- Base: pick the velocity command for the current phase of the schedule.
        elapsed = 0.0
        for duration, vx, vy, wz, description in BASE_SCHEDULE:
            if t < elapsed + duration:
                break
            elapsed += duration
        if description != last_phase:
            # Where the base REALLY is in the world (not just what the joints report)
            x, y, z = robot.data.body_pos_w.torch[0, base_body].tolist()
            # Isaac Lab 3.0 quaternions are (x, y, z, w) -- was (w, x, y, z) in 2.x
            qx, qy, qz, w = robot.data.body_quat_w.torch[0, base_body].tolist()
            yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
            print(f"[INFO] t={t:5.2f}s  base_link at x={x:+.2f} m, y={y:+.2f} m, z={z:+.2f} m, yaw={yaw:+.2f} rad"
                  f" -> {description}")
            last_phase = description
        base_vel = torch.tensor([[vx, vy, wz]], device=robot.device)
        robot.actuators.target_command.set_velocity_index(value=base_vel, joint_ids=base_ids)

        # -- Arm: gently swing joint 1 (shoulder yaw) and joint 4 (elbow) around the default pose.
        arm_target = default_arm_pos.clone()
        arm_target[:, 0] += 0.5 * math.sin(2 * math.pi * 0.25 * t)  # panda_joint1
        arm_target[:, 3] += 0.3 * math.sin(2 * math.pi * 0.25 * t)  # panda_joint4
        robot.actuators.target_command.set_position_index(value=arm_target, joint_ids=arm_ids)

        # -- Gripper: open for 1 s, closed for 1 s.
        finger_target = 0.04 if int(t) % 2 == 0 else 0.0
        robot.actuators.target_command.set_position_index(
            value=torch.full((robot.num_instances, len(finger_ids)), finger_target, device=robot.device),
            joint_ids=finger_ids,
        )

        # -- Send targets to the physics engine, step, read back the new state.
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        t += sim_dt

        # -- Video: grab one frame every `frame_interval` physics steps (e.g. 120 Hz / 30 fps = 4).
        step_count += 1
        if camera is not None:
            camera.update(sim_dt)
            if step_count % frame_interval == 0:
                frames.append(camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy())

    if frames:
        save_video(frames, args_cli.video_path, args_cli.video_fps)


def save_video(frames: list, path: str, fps: int):
    """Write frames to an mp4 (needs imageio + imageio-ffmpeg), else fall back to a folder of PNGs."""
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps)
        print(f"[INFO] Saved {len(frames)} frames to {path}")
    except Exception as e:  # missing imageio / ffmpeg backend
        import os

        import numpy as np
        from PIL import Image

        folder = os.path.splitext(path)[0] + "_frames"
        os.makedirs(folder, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(np.asarray(f, dtype=np.uint8)).save(f"{folder}/{i:05d}.png")
        print(f"[WARN] mp4 writer unavailable ({e}); saved {len(frames)} PNGs to {folder}")
        print(f"[WARN] Install with: pip install imageio imageio-ffmpeg  (or: ffmpeg -framerate {fps} -i {folder}/%05d.png out.mp4)")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    # Starts Isaac Sim / the physics runtime; closes it again when the block exits.
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([3.5, 3.5, 2.5], [0.0, 0.0, 0.5])
        robot, camera = design_scene()
        sim.reset()  # starts physics; robot buffers are valid only after this
        print("[INFO] Setup complete")
        run_simulator(sim, robot, camera)


if __name__ == "__main__":
    main()
