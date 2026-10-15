"""Record several completely different episodes in a row with a base policy: a video you can watch end to end.

Every episode has its own platform (footprint, base mass, payload) and reference trajectory, all drawn from the training
distribution. The footprint is fixed when an env is spawned, so each episode gets its own env, far apart from the others;
the camera cuts to the active one and only that env is reset and driven. Tipped robots stay down for ``--hold_tipped``
seconds before the next episode starts.
Overlays: episode number and outcome history, live CoM-offset bar, platform parameters, reference path and target.

Modes for the policy (``--mode``):
  deterministic  the network's mean action (what ``play_base_policy.py`` evaluates)
  sampled        mean action + Gaussian noise (``--noise_std``): how a policy acts while it explores, i.e. an untrained
                 network looks random. Use with an early checkpoint to show an untrained policy.
  none           no policy: zero base command (base parked), only the arm controller acts. The checkpoint is still
                 needed to pick the arm controller it was trained with.

Usage (from the Isaac Lab folder):

    PYTHONPATH=~/hybrid-rl-mpc-control/src .venv/bin/python ~/hybrid-rl-mpc-control/scripts/isaac/record_episodes_video.py \
        --checkpoint <run>/model_999.pt --title "TRAINED POLICY" --video_path out/trained_episodes.mp4
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from hybrid_control.rl.config import CONTROLLERS, saved_controller, saved_observation_cfg

OUT_DIR = Path(__file__).resolve().parents[2] / "out"
parser = argparse.ArgumentParser(description="Record one env over multiple episodes.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--title", type=str, default="POLICY")
parser.add_argument("--mode", choices=["deterministic", "sampled", "none"], default="deterministic")
parser.add_argument("--noise_std", type=float, default=0.5)
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--spacing", type=float, default=8.0, help="Distance between the per-episode envs [m].")
parser.add_argument("--hold_tipped", type=float, default=1.5, help="Seconds a tipped robot stays on screen.")
parser.add_argument("--hold_end", type=float, default=0.5, help="Seconds after a completed episode.")
parser.add_argument("--qp_workers", type=int, default=2)
parser.add_argument("--controller", choices=CONTROLLERS, default=None,
                    help="Arm controller; default: the one the checkpoint was trained with (<run>/params/env.yaml).")
parser.add_argument("--video_path", type=str, default=str(OUT_DIR / "episodes.mp4"))
parser.add_argument("--video_fps", type=int, default=30)
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.controller = args_cli.controller or saved_controller(args_cli.checkpoint)
args_cli.enable_cameras = True

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization
from hybrid_control.sim_interfaces.isaac.ridgeback.viz import ComMarkers, ReferenceMarkers, _legend, project

RES = (1600, 900)
FOCAL = 16.0  # [mm]
CAM_OFFSET, LOOK_OFFSET = (0.45, -3.3, 1.7), (0.45, 0.0, 0.55)  # relative to the real env's origin


def overlay(frame, base_env, k, status, t_ep, episode, n_episodes, outcomes, title, tag_px, rho):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    big, font, tag = ImageFont.load_default(size=38), ImageFont.load_default(size=26), ImageFont.load_default(size=21)
    w, h = image.size
    kw = dict(stroke_width=2, stroke_fill=(0, 0, 0))
    draw.text((20, 14), title, font=big, fill=(255, 255, 255), **kw)
    draw.text((20, 62), f"episode {episode} / {n_episodes}   t = {t_ep:4.1f} s", font=font, fill=(255, 255, 255), **kw)
    colour = (255, 90, 90) if status.startswith("TIPPED") else (130, 255, 130)
    draw.text((20, 98), status, font=font, fill=colour, **kw)
    p = base_env.params[k]
    draw.text((20, 134), f"platform {p.footprint[0]:.2f} x {p.footprint[1]:.2f} m, {p.base_mass:.0f} kg, payload {p.payload:.1f} kg",
              font=tag, fill=(255, 255, 255), **kw)
    # outcome history
    for i, ok in enumerate(outcomes):
        draw.text((20 + 34 * i, h - 52), "OK" if ok else "X", font=font, fill=(130, 255, 130) if ok else (255, 90, 90), **kw)
    # live CoM offset bar
    frac = min(max(rho, 0.0), 1.0)
    bar = (int(255 * min(2 * frac, 1.0)), int(255 * min(2 * (1 - frac), 1.0)), 40)
    x0, y0 = w // 2 - 160, h - 70
    draw.text((w // 2, y0 - 16), f"CoM offset {100 * rho:.0f}% of the half footprint (100% = at the edge)", font=tag, fill=(255, 255, 255), anchor="mm", **kw)
    draw.rectangle((x0, y0, x0 + 320, y0 + 22), fill=(30, 30, 30), outline=(255, 255, 255), width=2)
    draw.rectangle((x0, y0, x0 + int(320 * frac), y0 + 22), fill=bar)
    _legend(draw, w, 20, tag, kw, y0=20)
    return np.asarray(image)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        k = args_cli.episodes
        scenarios = DomainRandomization().sample_many(k, seed=args_cli.seed)  # same seed -> same episodes in every video
        # Env 0 is an unused dummy (the first env of a scene is not reliably rendered); episode e runs in env e.
        base_env = RidgebackDREnv([scenarios[0]] + scenarios, sim, spacing=args_cli.spacing, num_cols=k + 1)
        n = base_env.num_envs

        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        markers = ComMarkers(n, sim.device)
        ref_markers = ReferenceMarkers(n, sim.device)
        camera = Camera(
            cfg=CameraCfg(
                prim_path="/World/RecordCam", update_period=0.0, height=RES[1], width=RES[0], data_types=["rgb"],
                renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(focal_length=FOCAL, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
            )
        )
        base_env.start()
        env = BaseStabilizationEnv(base_env, StabilizationEnvCfg(controller=args_cli.controller, qp_workers=args_cli.qp_workers, observation=saved_observation_cfg(args_cli.checkpoint)), seed=args_cli.seed)
        env.auto_reset = False
        dev = env.device
        runner = OnPolicyRunner(env, make_runner_cfg(), log_dir=None, device=str(dev))
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(dev))
        horizon = torch.linspace(0.0, env.scfg.episode_length_s, ref_markers.n_path, device=dev)

        def new_episode(e):
            """Reset env ``e`` (fresh dynamics and trajectory), cut the camera to it, return (obs, label pixel)."""
            env._reset_idx(torch.tensor([e], device=dev))
            ref_markers.set_path(env.traj.lookahead(torch.zeros(n, device=dev), horizon) + base_env.origins[:, None])
            origin = base_env.origins[e]
            eye, target = origin + torch.tensor(CAM_OFFSET, device=dev), origin + torch.tensor(LOOK_OFFSET, device=dev)
            camera.set_world_poses_from_view(eye[None], target[None])
            px, _ = project((origin + torch.tensor([0.3, -0.5, 0.0], device=dev))[None], tuple(eye.tolist()), tuple(target.tolist()), *RES, focal=FOCAL)
            return env._compute_obs(), px

        import imageio.v2 as imageio

        Path(args_cli.video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args_cli.video_path, fps=args_cli.video_fps)
        outcomes, episode, hold, finished, t_tip = [], 1, 0, False, None
        obs, tag_px = new_episode(episode)
        gen = torch.Generator(device=dev).manual_seed(args_cli.seed)
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
                ref_markers.set_target(env.traj.position(env.episode_length_buf.float() * env.dt) + base_env.origins)
                camera.update(env.dt)
                frame = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
                rho = float(base_env.com_offset[episode] / base_env.half_length[episode])
                writer.append_data(overlay(frame, base_env, episode, status, t_ep, episode, k, outcomes, args_cli.title, tag_px, rho))
                if finished:
                    hold -= 1
                    if hold <= 0:
                        episode, finished = episode + 1, False
                        if episode <= k:
                            obs, tag_px = new_episode(episode)
        writer.close()
        print(f"[RESULT] {sum(outcomes)} / {len(outcomes)} episodes completed without tipping: {['OK' if o else 'X' for o in outcomes]}")
        print(f"[INFO] Saved video to {args_cli.video_path}")
        env.close()


if __name__ == "__main__":
    main()
