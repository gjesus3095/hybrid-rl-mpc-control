"""Evaluate moving-mass policies or a zero-command baseline on seeded episodes.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
        $ISAACLAB_PYTHON scripts/isaac/no_controller/play_moving_mass_policy.py \
        --checkpoint logs/moving_mass_policy/<run>/model_499.pt

Add ``--video`` to record an mp4 to out/play_moving_mass.mp4 (default path).
"""
import argparse
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation

REPO_ROOT = Path(__file__).resolve().parents[3]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--config', default=None, help='Environment YAML for a baseline without a checkpoint.')
parser.add_argument('--zero_policy', action='store_true')
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--episodes', type=int, default=5, help='Exact episode count per environment.')
parser.add_argument('--spacing', type=float, default=2.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--video', action='store_true')
parser.add_argument('--video_path', type=str, default=str(REPO_ROOT / 'out' / 'play_moving_mass.mp4'))
parser.add_argument('--video_fps', type=int, default=30)
parser.add_argument('--no_markers', action='store_true')
add_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint and not args.zero_policy:
    parser.error('--checkpoint is required unless --zero_policy is selected')
if args.episodes < 1:
    parser.error('--episodes must be positive')
if args.video:
    args.enable_cameras = True
args.scenario = 'play'

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner
from hybrid_control.rl.policy_no_controller.config import create_environment, load_configuration
from hybrid_control.rl.policy_no_controller.observations import to_base_frame


def main():
    cfg, runner_cfg = load_configuration(args.config, args.checkpoint)
    torch.manual_seed(args.seed)
    sim_cfg = sim_utils.SimulationCfg(dt=cfg.physics_dt, device=args.device)
    with launch_simulation(sim_cfg, args):
        if args.video:
            from hybrid_control.sim_interfaces.isaac.moving_mass.env import MovingMassEnv
            from hybrid_control.rl.policy_no_controller.env import BaseStabilizationEnv
            from hybrid_control.sim_interfaces.isaac.moving_mass.viz import DemoView
            base = MovingMassEnv(cfg.domain_randomization.sample_many(args.num_envs, args.seed),
                                 SimulationContext(sim_cfg), spacing=args.spacing,
                                 friction=cfg.friction, tau=cfg.base_time_constant,
                                 max_force=cfg.max_base_force, mechanism=cfg.mechanism)
            view = DemoView(base, args)
            base.start()
            env = BaseStabilizationEnv(base, cfg, args.seed)
        else:
            env = create_environment(SimulationContext(sim_cfg), cfg, args.num_envs, args.spacing, args.seed)
            base = env.env
            view = None

        env.auto_reset = False
        policy = None
        try:
            if args.video:
                view.start()
            if not args.zero_policy:
                runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=str(env.device))
                runner.load(args.checkpoint)
                policy = runner.get_inference_policy(device=str(env.device))
            counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            failures, samples, total_length = 0, 0, 0.0
            sums = dict(com_rho=0.0, tilt=0.0, footprint_excursion=0.0, speed=0.0)
            obs = env.get_observations()
            labels = [f'env {i}' for i in range(env.num_envs)]
            t = 0.0
            with torch.inference_mode():
                while (counts < args.episodes).any():
                    active = counts < args.episodes
                    action = torch.zeros(env.num_envs, 3, device=env.device) if policy is None else policy(obs)
                    action[~active] = 0
                    obs, _, dones, extras = env.step(action)
                    t += env.dt
                    if view is not None:
                        view.update(t, labels)
                    speed = to_base_frame(base.yaw, base.robot.data.body_com_lin_vel_w.torch[:, base.base_id])[:, :2].norm(dim=-1)
                    values = dict(com_rho=base.com_offset/base.half_length,
                                  tilt=torch.maximum(base.roll.abs(), base.pitch.abs()),
                                  footprint_excursion=base.left_footprint.float(), speed=speed)
                    for name, value in values.items():
                        sums[name] += float(value[active].sum())
                    samples += int(active.sum())
                    finished = dones.bool() & active
                    failures += int(extras['failed'][finished].sum())
                    total_length += float(env.episode_length_buf[finished].sum()) * env.dt
                    counts[finished] += 1
                    ids = finished.nonzero(as_tuple=False).flatten()
                    if len(ids):
                        env._reset_idx(ids)
                        obs = env._compute_obs()
            total = args.episodes * env.num_envs
            print(f'[RESULT] {total} episodes, {failures} failed, survival {1-failures/total:.3f}, mean duration {total_length/total:.2f}s')
            print('[METRICS]', {k: v/max(samples, 1) for k, v in sums.items()})
            if args.video:
                print(f'[INFO] Video: {args.video_path}')
        finally:
            env.close()
            if view is not None:
                view.close()


if __name__ == '__main__':
    main()
