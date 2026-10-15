"""Train/resume PPO for the controller-free moving-mass platform.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
        $ISAACLAB_PYTHON scripts/isaac/no_controller/train_moving_mass_policy.py --num_envs 64 --max_iterations 500

Logs and checkpoints go to <repo>/logs/moving_mass_policy/<timestamp>/ (view with ``tensorboard --logdir logs``).
"""
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--max_iterations', type=int, default=500)
parser.add_argument('--num_steps_per_env', type=int, default=None)
parser.add_argument('--spacing', type=float, default=2.5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config', help='Environment YAML; see the RL README.')
parser.add_argument('--resume', help='Moving-mass model checkpoint.')
parser.add_argument('--log_dir', default=None)
add_launcher_args(parser)
args = parser.parse_args()

import torch
import yaml
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner
from hybrid_control.rl.policy_no_controller.config import create_environment, load_configuration


def main():
    cfg, runner_cfg = load_configuration(args.config, args.resume)
    if args.num_steps_per_env is not None:
        runner_cfg['num_steps_per_env'] = args.num_steps_per_env
    torch.manual_seed(args.seed)
    sim_cfg = sim_utils.SimulationCfg(dt=cfg.physics_dt, device=args.device)
    with launch_simulation(sim_cfg, args):
        env = create_environment(SimulationContext(sim_cfg), cfg, args.num_envs, args.spacing, args.seed)
        directory = Path(args.log_dir or Path(__file__).resolve().parents[3]/'logs'/'moving_mass_policy'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        (directory/'params').mkdir(parents=True, exist_ok=True)
        (directory/'params'/'env.yaml').write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
        (directory/'params'/'agent.yaml').write_text(yaml.safe_dump(runner_cfg, sort_keys=False))
        (directory/'params'/'run.yaml').write_text(yaml.safe_dump({'seed': args.seed, 'num_envs': args.num_envs}))
        try:
            runner = OnPolicyRunner(env, runner_cfg, log_dir=str(directory), device=str(env.device))
            if args.resume:
                runner.load(args.resume)
            print(f'[INFO] moving mass: {env.num_envs} envs, {1/env.dt:.0f} Hz policy; logs {directory}')
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        finally:
            env.close()


if __name__ == '__main__':
    main()
