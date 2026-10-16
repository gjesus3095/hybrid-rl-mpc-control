"""Record the trained base policy next to an untrained one on identical scenarios (same platform, dynamics and
reference trajectory), in one scene: the left half of the row runs the trained policy, the right half the untrained.

"Untrained" is a checkpoint saved after the first PPO update (``model_0.pt``), i.e. practically the initial network.
The scenarios are picked from a pool of randomized platforms to include small/light ones (see ``--hard_ranks``); their
trajectory kinds cycle through ``--kinds`` (by default fixed point, sinusoidal, quintic). Each env's tag shows its
trajectory kind and current phase (spawn -> A, A -> B, ...).

Usage (from the repo root, after the Isaac Lab setup in the root README):

    $ISAACLAB_PYTHON scripts/isaac/compare_policies_video.py \
        --trained <run>/model_149.pt --untrained <run>/model_0.pt
"""

import argparse
import copy
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from hybrid_control.rl.config import (
    CONTROLLERS, qp_solver_name, saved_controller, saved_observation_cfg, saved_qp_solver, saved_spatial_order,
)
from hybrid_control.rl.policy_with_controller.trajectories import KINDS, TrajectoryCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.params import VIDEO_LIGHT

OUT_DIR = Path(__file__).resolve().parents[2] / "out"
parser = argparse.ArgumentParser(description="Side-by-side video of a trained and an untrained base policy.")
parser.add_argument("--trained", type=str, required=True)
parser.add_argument("--untrained", type=str, required=True)
parser.add_argument("--scenarios", type=int, default=4)
parser.add_argument("--hard_ranks", type=int, nargs="+", default=[0, 20, 40, 60],
                    help="Pick scenarios at these ranks of a pool sorted from least to most stable platform.")
parser.add_argument("--pool", type=int, default=80)
parser.add_argument("--kinds", nargs="+", choices=list(KINDS), default=list(KINDS),
                    help="Trajectory kinds, cycled over the scenarios.")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--spacing", type=float, default=2.4)
parser.add_argument("--qp_workers", type=int, default=4)
parser.add_argument("--controller", choices=CONTROLLERS, default=None,
                    help="Arm controller; default: the one --trained was trained with (<run>/params/env.yaml).")
parser.add_argument("--spatial_order", type=int, default=None,
                    help="Bezier order of the quintic paths; default: the trained checkpoint's, 1 for old runs.")
parser.add_argument("--video_path", type=str, default=str(OUT_DIR / "policy_comparison.mp4"))
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument("--light", type=float, default=VIDEO_LIGHT,
                    help=f"Distant light intensity (default {VIDEO_LIGHT:g}; training and the viewer scripts use 3000, "
                         "which overexposes the light ground in the camera images).")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.controller = args_cli.controller or saved_controller(args_cli.trained)
if args_cli.spatial_order is None:
    # Both halves replay the same trajectories, so one order is used: the trained policy's.
    args_cli.spatial_order = saved_spatial_order(args_cli.trained)
    untrained_order = saved_spatial_order(args_cli.untrained)
    if untrained_order != args_cli.spatial_order:
        print(f"[WARN] the untrained checkpoint was trained with quintic path order {untrained_order}; both halves use "
              f"the trained policy's order {args_cli.spatial_order} (override with --spatial_order)")
args_cli.enable_cameras = True

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.policy_with_controller.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization

FOCAL = 12.0  # [mm]
VIEW = (1280, 720)  # one camera per group; the video is the two views side by side
CAM_DIST, CAM_HEIGHT, CAM_LOOK_Z = 6.2, 3.2, 0.5


def pick_scenarios(n: int, ranks: list[int], pool: int, seed: int):
    """Scenarios spread over a stability ranking (smallest footprint x lightest base first)."""
    params = DomainRandomization().sample_many(pool, seed=seed)
    params.sort(key=lambda p: min(p.footprint) * p.base_mass)
    return [params[r] for r in ranks[:n]]


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        k = args_cli.scenarios
        scenarios = pick_scenarios(k, args_cli.hard_ranks, args_cli.pool, args_cli.seed)
        # Env 0 is a sacrificial dummy outside both views: in this setup the first env is not rendered (cause not found),
        # so the real scenarios live in envs 1..2k: 1..k trained, k+1..2k untrained, same scenario at i and i + k.
        all_params = [scenarios[0]] + scenarios + scenarios
        base_env = RidgebackDREnv(all_params, sim, spacing=args_cli.spacing, num_cols=2 * k + 1,
                                  light_intensity=args_cli.light)

        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        from hybrid_control.sim_interfaces.isaac.ridgeback.viz import ComMarkers, ReferenceMarkers, overlay_compare, project

        markers = ComMarkers(base_env.num_envs, sim.device)
        ref_markers = ReferenceMarkers(base_env.num_envs, sim.device, len(saved_observation_cfg(args_cli.trained).lookahead))
        cameras = [
            Camera(
                cfg=CameraCfg(
                    prim_path=f"/World/RecordCam{side}", update_period=0.0, height=VIEW[1], width=VIEW[0],
                    data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=FOCAL, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                    ),
                )
            )
            for side in ("L", "R")
        ]
        base_env.start()
        env = BaseStabilizationEnv(base_env, StabilizationEnvCfg(controller=args_cli.controller, qp_workers=args_cli.qp_workers, observation=saved_observation_cfg(args_cli.trained), trajectory=TrajectoryCfg(spatial_order=args_cli.spatial_order)), seed=args_cli.seed)
        n, dev = env.num_envs, env.device

        # Identical scenarios in both halves: same dynamics (the footprint is already equal) and same trajectory.
        env.auto_reset = False
        kinds = [KINDS[args_cli.kinds[(i - 1) % len(args_cli.kinds)]] for i in range(n)]  # scenario s: env 1 + s
        env._reset_idx(torch.arange(n, device=dev), dynamics=all_params, kinds=torch.tensor(kinds, device=dev))
        for name in env.traj.STATE:
            tensor = getattr(env.traj, name)
            tensor[k + 1 :] = tensor[1 : k + 1]
        env._obs = env._compute_obs()

        runner = OnPolicyRunner(env, make_runner_cfg(), log_dir=None, device=str(dev))
        runner.load(args_cli.trained)
        trained = copy.deepcopy(runner.get_inference_policy(device=str(dev)))
        runner.load(args_cli.untrained)
        untrained = copy.deepcopy(runner.get_inference_policy(device=str(dev)))

        # one camera per group, looking at the centre of its row of envs from the front
        trained_ids, untrained_ids = slice(1, k + 1), slice(k + 1, 2 * k + 1)
        centres = [base_env.origins[trained_ids].mean(0), base_env.origins[untrained_ids].mean(0)]
        views = []
        for cam, centre in zip(cameras, centres):
            target = (centre + torch.tensor([0.4, 0.0, CAM_LOOK_Z], device=dev))
            eye = target + torch.tensor([0.0, -CAM_DIST, CAM_HEIGHT - CAM_LOOK_Z], device=dev)
            cam.set_world_poses_from_view(eye[None], target[None])
            views.append((tuple(eye.tolist()), tuple(target.tolist())))
        tag_px = torch.zeros(n, 2, device=dev)
        for g, (eye, target) in enumerate(views):
            ids = (trained_ids, untrained_ids)[g]
            px, _ = project(base_env.origins[ids] + torch.tensor([0.3, -0.7, 0.02], device=dev), eye, target, *VIEW, focal=FOCAL)
            tag_px[ids] = px + torch.tensor([g * VIEW[0], 0.0], device=dev)
        titles_px = torch.tensor([[VIEW[0] / 2, 0.0], [1.5 * VIEW[0], 0.0]])
        divider_px = float(VIEW[0])
        titles = [("TRAINED POLICY", (130, 255, 130)), ("UNTRAINED POLICY", (255, 170, 90))]
        eval_solver = qp_solver_name(env.scfg.controller, env.scfg.ik.qp_solver)
        subtitles = [f"QP solver: training {saved_qp_solver(c)}, evaluation {eval_solver}"
                     for c in (args_cli.trained, args_cli.untrained)]

        tip_times = [math.nan] * n
        frames = []
        obs = env.get_observations()
        steps = int(round(args_cli.seconds / env.dt))
        with torch.inference_mode():
            for step in range(steps):
                actions = torch.zeros(n, 3, device=dev)
                actions[trained_ids] = trained(obs[trained_ids])
                actions[untrained_ids] = untrained(obs[untrained_ids])
                obs, _, _, _ = env.step(actions)
                t = (step + 1) * env.dt
                for i in torch.nonzero(base_env.tipped).flatten().tolist():
                    if math.isnan(tip_times[i]):
                        tip_times[i] = t
                markers.update(base_env)
                ref_markers.update(env)
                for cam in cameras:
                    cam.update(env.dt)
                frame = np.concatenate(
                    [cam.data.output["rgb"].torch[0, ..., :3].cpu().numpy() for cam in cameras], axis=1
                ).copy()
                rho = (base_env.com_offset / base_env.half_length).cpu().numpy()
                frames.append(overlay_compare(frame, base_env, tag_px, t, titles, titles_px, divider_px, tip_times, rho, subtitles,
                                              traj=env.traj))

        for s in range(k):
            p = scenarios[s]
            a, b = 1 + s, 1 + k + s
            print(f"[SCENARIO {s}] {p.footprint[0]:.2f} x {p.footprint[1]:.2f} m, base {p.base_mass:.1f} kg, payload {p.payload:.1f} kg, "
                  f"com_z {p.base_com_z:.2f} m | trained: {'TIPPED at %.1f s' % tip_times[a] if base_env.tipped[a] else 'upright'}, "
                  f"untrained: {'TIPPED at %.1f s' % tip_times[b] if base_env.tipped[b] else 'upright'}")
        import imageio.v2 as imageio

        Path(args_cli.video_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args_cli.video_path, frames, fps=args_cli.video_fps)
        print(f"[INFO] Saved {len(frames)} frames to {args_cli.video_path}")
        env.close()


if __name__ == "__main__":
    main()
