"""Domain-randomized Ridgeback-Franka envs side by side: each env has its own footprint, base mass,
COM height and payload, sampled from ``DomainRandomization``.

Scenario (same for every env): settle with the arm folded -> arm swings to a stretched pose -> hold. Envs
whose platform is too small or too light for their payload tip over; the others stay upright. Optionally
the base follows ``--cmd`` once the arm is stretched.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
        $ISAACLAB_PYTHON scripts/isaac/with_controller/ridgeback_dr_demo.py --num_envs 8 --seed 0 --video
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

OUT_DIR = Path(__file__).resolve().parents[3] / "out"  # <repo>/out, not the Isaac Lab folder
parser = argparse.ArgumentParser(description="Domain-randomized free-base Ridgeback-Franka envs.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_cols", type=int, default=4, help="Grid columns.")
parser.add_argument("--spacing", type=float, default=2.5, help="Env spacing [m].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
parser.add_argument("--settle", type=float, default=1.0, help="Seconds with the arm folded.")
parser.add_argument("--swing", type=float, default=3.0, help="Seconds to move the arm to the stretched pose.")
parser.add_argument("--hold", type=float, default=3.0, help="Seconds held stretched.")
parser.add_argument("--video", action="store_true", help="Record offscreen to an mp4.")
parser.add_argument("--video_path", type=str, default=str(OUT_DIR / "ridgeback_dr_envs.mp4"), help="Default: <repo>/out/.")
parser.add_argument("--video_fps", type=int, default=30)
add_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization

CAM_EYE, CAM_TARGET = (0.0, -8.0, 5.5), (0.0, -0.3, 0.3)
FOCAL = 16.0  # [mm]; wide enough to fit the whole env grid
RES = (1600, 900)  # width, height


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(CAM_EYE, CAM_TARGET)
        params = DomainRandomization().sample_many(args_cli.num_envs, seed=args_cli.seed)
        env = RidgebackDREnv(params, sim, spacing=args_cli.spacing, num_cols=args_cli.num_cols)

        camera = markers = None
        if args_cli.video:
            # Imported here: camera.py imports pxr, which must load only after Kit has started.
            from isaaclab.sensors.camera import Camera, CameraCfg
            from isaaclab_physx.renderers import IsaacRtxRendererCfg

            from hybrid_control.sim_interfaces.isaac.ridgeback.viz import ComMarkers, overlay, project

            markers = ComMarkers(env.num_envs, sim.device)
            camera = Camera(
                cfg=CameraCfg(
                    prim_path="/World/RecordCam",
                    update_period=1.0 / args_cli.video_fps,
                    height=RES[1],
                    width=RES[0],
                    data_types=["rgb"],
                    renderer_cfg=IsaacRtxRendererCfg(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=FOCAL, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                    ),
                )
            )
        env.start()
        for i, p in enumerate(params):
            print(f"[INFO] env {i}: footprint {p.footprint[0]:.2f} x {p.footprint[1]:.2f} m, base {p.base_mass:.1f} kg, "
                  f"com_z {p.base_com_z:.2f} m, payload {p.payload:.1f} kg, total {env.mass_total[i]:.1f} kg")

        device = sim.device
        if camera is not None:
            camera.set_world_poses_from_view(
                torch.tensor([CAM_EYE], device=device), torch.tensor([CAM_TARGET], device=device)
            )
            tag_points = env.origins + torch.tensor([0.3, -0.85, 0.02], device=device)
            tag_px, _ = project(tag_points, CAM_EYE, CAM_TARGET, *RES, focal=FOCAL)

        cmd_active = torch.tensor(args_cli.cmd, device=device).expand(env.num_envs, 3)
        t_hold = args_cli.settle + args_cli.swing
        t_end = t_hold + args_cli.hold
        frame_interval = max(1, round(1.0 / (args_cli.video_fps * env.dt)))
        frames, step, reported = [], 0, torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        while env.time < t_end:
            s = min(max((env.time - args_cli.settle) / max(args_cli.swing, 1e-6), 0.0), 1.0)
            s = s * s * (3 - 2 * s)  # smoothstep
            q = env.q_fold + s * (env.q_stretched - env.q_fold)
            cmd = cmd_active if env.time >= t_hold else torch.zeros_like(cmd_active)
            env.step(q, cmd)
            step += 1
            for i in torch.nonzero(env.tipped & ~reported).flatten().tolist():
                print(f"[INFO] t={env.time:5.2f}s env {i} TIPPED")
            reported |= env.tipped
            if camera is not None:
                markers.update(env)
                camera.update(env.dt)
                if step % frame_interval == 0:
                    frame = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
                    frames.append(overlay(frame, env, tag_px, env.time))

        for i in range(env.num_envs):
            print(f"[RESULT] env {i}: tipped={bool(env.tipped[i])}, com_left_footprint={bool(env.left_footprint[i])}, "
                  f"final COM offset {env.com_offset[i]:.3f} m / half-length {env.half_length[i]:.3f} m")
        if frames:
            import imageio.v2 as imageio

            Path(args_cli.video_path).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args_cli.video_path, frames, fps=args_cli.video_fps)
            print(f"[INFO] Saved {len(frames)} frames to {args_cli.video_path}")


if __name__ == "__main__":
    main()
