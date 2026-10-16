"""Render one PNG of the cartoon chicken skin on the free-base Ridgeback-Franka (0.4 x 0.4 m platform).

The arm moves from its folded pose to a "head forward" pose (``Q_CHICKEN``) and holds it, then an offscreen RTX
camera takes the picture. Headless, so it runs over ssh; the PNG can be inspected without Isaac Lab.

Usage (from this project folder):

    DISPLAY= PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/isaac/chicken_snapshot.py
    (--show_robot keeps the Franka visible inside the chicken, --eye/--target move the camera)

Assets: ``uv run python scripts/make_chicken_assets.py`` regenerates the meshes and the overlay URDF.
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

OUT_DIR = Path(__file__).resolve().parents[2] / "out"
parser = argparse.ArgumentParser(description="Single image of the chicken skin.")
parser.add_argument("--out", type=str, default=str(OUT_DIR / "chicken_snapshot.png"), help="Default: <repo>/out/.")
parser.add_argument("--footprint", type=float, nargs=2, default=[0.40, 0.40], metavar=("LENGTH", "WIDTH"))
parser.add_argument("--show_robot", action="store_true", help="Keep the Franka meshes visible (they poke through the neck).")
parser.add_argument("--eye", type=float, nargs=3, default=[1.95, 1.6, 1.3])
parser.add_argument("--target", type=float, nargs=3, default=[0.3, 0.0, 0.6])
parser.add_argument("--res", type=int, nargs=2, default=[1280, 960], metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--move", type=float, default=1.5, help="Seconds to move the arm to the head-forward pose.")
parser.add_argument("--hold", type=float, default=1.0, help="Seconds held before the picture.")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # offscreen rendering (works headless)

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import matrix_from_quat

from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import BaseParams

# Head in front of the mount, beak forward and 10 deg down, comb up (hand z ~ +x, hand y ~ +z); the elbow folds
# back over the body, so the neck makes an S. Found by IK on lula_franka_gen.urdf.
Q_CHICKEN = [0.285, -0.633, -0.263, -2.226, -0.726, 2.972, -0.196]  # panda_joint1..7 [rad]


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        # Imported here: they import pxr, which must load only after Kit has started.
        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        from hybrid_control.sim_interfaces.isaac.ridgeback.chicken import ChickenSkin

        sim = SimulationContext(sim_cfg)
        env = RidgebackDREnv([BaseParams(footprint=tuple(args_cli.footprint), payload=0.0)], sim)
        dome = sim_utils.DomeLightCfg(intensity=600.0, color=(0.85, 0.9, 1.0))
        dome.func("/World/DomeLight", dome)
        skin = ChickenSkin(["/World/envs/env_0/Robot"], hide_robot=not args_cli.show_robot)
        camera = Camera(
            cfg=CameraCfg(
                prim_path="/World/SnapshotCam",
                height=args_cli.res[1],
                width=args_cli.res[0],
                data_types=["rgb"],
                # the camera is outside /World/envs/env_0, so it must see across the env partitions
                renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 1.0e4)
                ),
            )
        )
        env.start()
        device = sim.device
        camera.set_world_poses_from_view(torch.tensor([args_cli.eye], device=device),
                                         torch.tensor([args_cli.target], device=device))

        q_chicken = torch.tensor([Q_CHICKEN], device=device)
        no_cmd = torch.zeros(1, 3, device=device)
        while env.time < args_cli.move + args_cli.hold:
            s = min(env.time / max(args_cli.move, 1e-6), 1.0)
            s = s * s * (3 - 2 * s)  # smoothstep
            env.step(env.q_fold + s * (q_chicken - env.q_fold), no_cmd, arm_effort=env.gravity_compensation())
            skin.update(env.robot)
            camera.update(env.dt)

        robot = env.robot
        hand = robot.find_bodies("panda_hand")[0][0]
        rot = matrix_from_quat(robot.data.body_quat_w.torch[0, hand]).cpu()  # xyzw in Isaac Lab 3.0
        print(f"[INFO] hand at {robot.data.body_pos_w.torch[0, hand].cpu().numpy().round(3)}, "
              f"beak (hand z) {rot[:, 2].numpy().round(2)}, comb (hand y) {rot[:, 1].numpy().round(2)}, "
              f"tipped={bool(env.tipped[0])}")

        from PIL import Image

        frame = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
        Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(args_cli.out)
        print(f"[INFO] Saved {args_cli.out}")


if __name__ == "__main__":
    main()
