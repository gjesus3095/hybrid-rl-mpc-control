"""Record several completely different episodes in a row with a base policy: a video you can watch end to end.

Every episode has its own platform (footprint, base mass, payload) and reference trajectory, drawn from the training
distribution; the trajectory kinds cycle through ``--kinds`` (by default fixed point, sinusoidal, quintic), so every kind
is shown. The footprint is fixed when an env is spawned, so each episode gets its own env, far apart from the others;
the cameras cut to the active one and only that env is reset and driven. Tipped robots stay down for ``--hold_tipped``
seconds before the next episode starts.
Four cameras look at the active env from its front, left, right and back (a 2 x 2 grid), next to a side panel with the
episode number and outcome history, trajectory kind and phase (spawn -> A, A -> B, ...), live CoM-offset bar, platform
parameters, QP solver (training and this run) and the marker legend.

Modes for the policy (``--mode``):
  deterministic  the network's mean action (what ``play_base_policy.py`` evaluates)
  sampled        mean action + Gaussian noise (``--noise_std``): how a policy acts while it explores, i.e. an untrained
                 network looks random. Use with an early checkpoint to show an untrained policy.
  none           no policy: zero base command (base parked), only the arm controller acts. The checkpoint is still
                 needed to pick the arm controller it was trained with.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    $ISAACLAB_PYTHON scripts/isaac/record_episodes_video.py \\
        --checkpoint <run>/model_999.pt --title "TRAINED POLICY" --video_path out/trained_episodes.mp4
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from hybrid_control.rl.config import (
    CONTROLLERS, qp_solver_name, saved_controller, saved_observation_cfg, saved_qp_solver, saved_spatial_order,
)
from hybrid_control.rl.policy_with_controller.trajectories import KINDS, TrajectoryCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.params import VIDEO_LIGHT

OUT_DIR = Path(__file__).resolve().parents[2] / "out"
parser = argparse.ArgumentParser(description="Record one env over multiple episodes.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--title", type=str, default="POLICY")
parser.add_argument("--mode", choices=["deterministic", "sampled", "none"], default="deterministic")
parser.add_argument("--noise_std", type=float, default=0.5)
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--kinds", nargs="+", choices=list(KINDS), default=list(KINDS),
                    help="Trajectory kinds, cycled over the episodes.")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--spacing", type=float, default=8.0, help="Distance between the per-episode envs [m].")
parser.add_argument("--hold_tipped", type=float, default=1.5, help="Seconds a tipped robot stays on screen.")
parser.add_argument("--hold_end", type=float, default=0.5, help="Seconds after a completed episode.")
parser.add_argument("--qp_workers", type=int, default=2)
parser.add_argument("--controller", choices=CONTROLLERS, default=None,
                    help="Arm controller; default: the one the checkpoint was trained with (<run>/params/env.yaml).")
parser.add_argument("--spatial_order", type=int, default=None,
                    help="Bezier order of the quintic paths; default: the checkpoint's (<run>/params/env.yaml), 1 for old runs.")
parser.add_argument("--video_path", type=str, default=str(OUT_DIR / "episodes.mp4"))
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument("--view", choices=["all", "front", "left", "right", "back"], default="all",
                    help="One camera view (full size) instead of the 2 x 2 grid.")
parser.add_argument("--elevation", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="Fixed-point/sine centre elevation above the shoulder [deg]; default: the training range.")
parser.add_argument("--height_clearance", type=float, default=None,
                    help="Quintic paths stay this far above the platform top [m]; default: the training value.")
parser.add_argument("--head_forward", action="store_true",
                    help="Targets with the TCP pointing forward (10 deg down) and the hand y axis up, i.e. the chicken looks "
                         "ahead with its comb up (quintic and sine targets; a fixed point keeps the spawn orientation). Out of the "
                         "training distribution (TCP pointing down).")
parser.add_argument("--light", type=float, default=VIDEO_LIGHT,
                    help=f"Distant light intensity (default {VIDEO_LIGHT:g}; training and the viewer scripts use 3000, "
                         "which overexposes the light ground in the camera images).")
parser.add_argument("--chicken", action="store_true", help="Dress the robot as a cartoon chicken (visual only).")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.controller = args_cli.controller or saved_controller(args_cli.checkpoint)
if args_cli.spatial_order is None:
    args_cli.spatial_order = saved_spatial_order(args_cli.checkpoint)
args_cli.enable_cameras = True

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.policy_with_controller.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization
from hybrid_control.sim_interfaces.isaac.ridgeback.viz import ComMarkers, ReferenceMarkers, _legend

TILE = (800, 450)  # one camera view; the views form a 2 x 2 grid
PANEL_W = 480  # side panel with the text overlays (frame width 480 + 1600 = 2080, a multiple of 16)
FOCAL = 16.0  # [mm]
LOOK_OFFSET, CAM_DIST, CAM_HEIGHT = (0.45, 0.0, 0.55), 3.3, 1.15  # look point (env-local), camera distance and height
# (name, horizontal direction from the look point to the camera) in grid order; the robot faces +x at spawn
VIEWS = [("front", (1.0, 0.0)), ("left", (0.0, 1.0)), ("right", (0.0, -1.0)), ("back", (-1.0, 0.0))]
if args_cli.view != "all":  # a single view fills the whole grid area
    VIEWS = [v for v in VIEWS if v[0] == args_cli.view]
    TILE = (2 * TILE[0], 2 * TILE[1])


def compose(tiles, base_env, k, status, t_ep, episode, n_episodes, outcomes, title, rho, qp_lines, kind, phase):
    """Side panel + 2 x 2 grid of the camera views (or the single view), with all text overlays."""
    from PIL import Image, ImageDraw, ImageFont

    if len(tiles) == 1:
        grid = tiles[0]
    else:
        grid = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
    h = grid.shape[0]
    frame = np.concatenate([np.full((h, PANEL_W, 3), 25, dtype=np.uint8), grid.astype(np.uint8)], axis=1)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    big, font, tag = ImageFont.load_default(size=38), ImageFont.load_default(size=26), ImageFont.load_default(size=21)
    kw = dict(stroke_width=2, stroke_fill=(0, 0, 0))
    white, grey, accent = (255, 255, 255), (180, 180, 180), (255, 215, 90)
    # camera labels and grid lines
    for i, (name, _) in enumerate(VIEWS):
        x, y = PANEL_W + (i % 2) * TILE[0], (i // 2) * TILE[1]
        draw.text((x + 14, y + 10), f"{name} view", font=tag, fill=white, **kw)
    draw.line((PANEL_W, 0, PANEL_W, h), fill=white, width=3)
    if len(VIEWS) > 1:
        draw.line((PANEL_W + TILE[0], 0, PANEL_W + TILE[0], h), fill=white, width=2)
        draw.line((PANEL_W, TILE[1], frame.shape[1], TILE[1]), fill=white, width=2)
    # side panel
    draw.text((20, 14), title, font=big, fill=white, **kw)
    draw.text((20, 62), f"episode {episode} / {n_episodes}   t = {t_ep:4.1f} s", font=font, fill=white, **kw)
    draw.text((20, 98), status, font=font, fill=(255, 90, 90) if status.startswith("TIPPED") else (130, 255, 130), **kw)
    draw.text((20, 150), "trajectory", font=tag, fill=grey, **kw)
    draw.text((20, 174), kind, font=font, fill=accent, **kw)
    draw.text((20, 214), "phase", font=tag, fill=grey, **kw)
    draw.text((20, 238), phase, font=font, fill=accent, **kw)
    p = base_env.params[k]
    draw.text((20, 290), "platform", font=tag, fill=grey, **kw)
    draw.text((20, 314), f"{p.footprint[0]:.2f} x {p.footprint[1]:.2f} m, {p.base_mass:.0f} kg", font=tag, fill=white, **kw)
    draw.text((20, 340), f"payload {p.payload:.1f} kg", font=tag, fill=white, **kw)
    draw.text((20, 384), "QP solver", font=tag, fill=grey, **kw)
    for j, line in enumerate(qp_lines):
        draw.text((20, 408 + 26 * j), line, font=tag, fill=white, **kw)
    _legend(draw, PANEL_W, 20, tag, kw, y0=486)
    # live CoM offset bar
    frac = min(max(rho, 0.0), 1.0)
    bar = (int(255 * min(2 * frac, 1.0)), int(255 * min(2 * (1 - frac), 1.0)), 40)
    x0, y0 = 20, h - 130
    draw.text((x0, y0 - 54), f"CoM offset {100 * rho:.0f}% of the half footprint", font=tag, fill=white, **kw)
    draw.text((x0, y0 - 30), "(100% = at the edge)", font=tag, fill=grey, **kw)
    draw.rectangle((x0, y0, x0 + 400, y0 + 22), fill=(30, 30, 30), outline=white, width=2)
    draw.rectangle((x0, y0, x0 + int(400 * frac), y0 + 22), fill=bar)
    # outcome history
    for i, ok in enumerate(outcomes):
        draw.text((20 + 50 * i, h - 52), "OK" if ok else "X", font=font, fill=(130, 255, 130) if ok else (255, 90, 90), **kw)
    return np.asarray(image)


def trajectory_cfg() -> TrajectoryCfg:
    """The training trajectory distribution, with the recording overrides from the command line."""
    overrides = {}
    if args_cli.elevation is not None:
        overrides["elevation_deg"] = tuple(args_cli.elevation)
    if args_cli.height_clearance is not None:
        overrides["height_clearance"] = args_cli.height_clearance
    if args_cli.head_forward:
        # pitch the TCP z axis to the front, then yaw -90 deg about it so the hand y axis (the comb) points up; targets
        # in front only, or the neck twists to look ahead while the head sits beside the body
        overrides.update(pitch_deg=80.0, tilt_max_deg=10.0, yaw_centre_deg=-90.0, yaw_range_deg=10.0, azimuth_deg=30.0)
    return TrajectoryCfg(spatial_order=args_cli.spatial_order, **overrides)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        k = args_cli.episodes
        scenarios = DomainRandomization().sample_many(k, seed=args_cli.seed)  # same seed -> same episodes in every video
        # Env 0 is an unused dummy (the first env of a scene is not reliably rendered); episode e runs in env e.
        base_env = RidgebackDREnv([scenarios[0]] + scenarios, sim, spacing=args_cli.spacing, num_cols=k + 1,
                                  light_intensity=args_cli.light)
        n = base_env.num_envs

        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        skin = None
        if args_cli.chicken:
            from hybrid_control.sim_interfaces.isaac.ridgeback.chicken import ChickenSkin

            skin = ChickenSkin([f"/World/envs/env_{i}/Robot" for i in range(n)])
        markers = ComMarkers(n, sim.device)
        ref_markers = ReferenceMarkers(n, sim.device, len(saved_observation_cfg(args_cli.checkpoint).lookahead))
        cameras = [
            Camera(
                cfg=CameraCfg(
                    prim_path=f"/World/RecordCam_{name}", update_period=0.0, height=TILE[1], width=TILE[0],
                    data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=FOCAL, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                    ),
                )
            )
            for name, _ in VIEWS
        ]
        base_env.start()
        env = BaseStabilizationEnv(base_env, StabilizationEnvCfg(controller=args_cli.controller, qp_workers=args_cli.qp_workers, observation=saved_observation_cfg(args_cli.checkpoint), trajectory=trajectory_cfg()), seed=args_cli.seed)
        env.auto_reset = False
        qp_lines = [f"training: {saved_qp_solver(args_cli.checkpoint)}",
                    f"evaluation: {qp_solver_name(env.scfg.controller, env.scfg.ik.qp_solver)}"]
        dev = env.device
        runner = OnPolicyRunner(env, make_runner_cfg(), log_dir=None, device=str(dev))
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(dev))

        def new_episode(e):
            """Reset env ``e`` (fresh dynamics, a trajectory of the episode's kind), cut the cameras to it, return obs."""
            kind = KINDS[args_cli.kinds[(e - 1) % len(args_cli.kinds)]]
            env._reset_idx(torch.tensor([e], device=dev), kinds=torch.tensor([kind], device=dev))
            target = base_env.origins[e] + torch.tensor(LOOK_OFFSET, device=dev)
            for cam, (_, (dx, dy)) in zip(cameras, VIEWS):
                eye = target + torch.tensor([CAM_DIST * dx, CAM_DIST * dy, CAM_HEIGHT], device=dev)
                cam.set_world_poses_from_view(eye[None], target[None])
            return env._compute_obs()

        import imageio.v2 as imageio

        Path(args_cli.video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args_cli.video_path, fps=args_cli.video_fps)
        outcomes, episode, hold, finished, t_tip = [], 1, 0, False, None
        obs = new_episode(episode)
        gen = torch.Generator(device=dev).manual_seed(args_cli.seed)
        # close the writer even on a crash or Ctrl-C: an unclosed MP4 has no index (moov atom) and does not play
        # (imageio's own ``with writer`` skips the close when an exception is raised)
        try:
            with torch.inference_mode():
                while episode <= k:
                    a = torch.zeros(n, 3, device=dev)
                    if args_cli.mode != "none":
                        a[episode] = policy(obs[episode : episode + 1])[0]
                    if args_cli.mode == "sampled":
                        a[episode] += args_cli.noise_std * torch.randn(3, device=dev, generator=gen)
                    obs, _, dones, extras = env.step(a)
                    t_ep = float(env.episode_length_buf[episode]) * env.dt
                    if not finished and bool(dones[episode]):
                        failed = bool(extras["failed"][episode])
                        outcomes.append(not failed)
                        finished, t_tip = True, t_ep
                        hold = int((args_cli.hold_tipped if failed else args_cli.hold_end) / env.dt)
                    status = f"TIPPED at {t_tip:.1f} s" if finished and not outcomes[-1] else ("completed" if finished else "upright")
                    markers.update(base_env)
                    if skin is not None:
                        skin.update(base_env.robot)
                    ref_markers.update(env)
                    for cam in cameras:
                        cam.update(env.dt)
                    tiles = [cam.data.output["rgb"].torch[0, ..., :3].cpu().numpy() for cam in cameras]
                    rho = float(base_env.com_offset[episode] / base_env.half_length[episode])
                    kind, phase = env.traj.kind_name(episode), env.traj.phase_name(episode, t_ep)
                    writer.append_data(compose(tiles, base_env, episode, status, t_ep, episode, k, outcomes, args_cli.title,
                                               rho, qp_lines, kind, phase))
                    if finished:
                        hold -= 1
                        if hold <= 0:
                            episode, finished = episode + 1, False
                            if episode <= k:
                                obs = new_episode(episode)
        finally:
            writer.close()
        print(f"[RESULT] {sum(outcomes)} / {len(outcomes)} episodes completed without tipping: {['OK' if o else 'X' for o in outcomes]}")
        print(f"[INFO] Saved video to {args_cli.video_path}")
        env.close()


if __name__ == "__main__":
    main()
