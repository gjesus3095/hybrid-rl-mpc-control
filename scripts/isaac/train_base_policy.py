"""Train the RL base policy (PPO, RSL-RL) with the IK/QP arm in the loop, on domain-randomized Ridgeback-Franka envs.

The policy outputs base velocity commands (vx, vy, wz); the QP tracks a world-frame arm reference trajectory and knows
nothing about the base. Reward: keep the whole-robot COM centred over the platform and stay near the trajectory centre.

Usage (from the Isaac Lab folder):

    PYTHONPATH=~/hybrid-rl-mpc-control/src uv run --extra isaacsim python \
        ~/hybrid-rl-mpc-control/scripts/isaac/train_base_policy.py --num_envs 64 --max_iterations 500

Logs and checkpoints go to <repo>/logs/base_policy/<timestamp>/ (view with ``tensorboard --logdir logs``).
"""

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Train the RL base policy.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--num_steps_per_env", type=int, default=24, help="Rollout length per PPO iteration.")
parser.add_argument("--spacing", type=float, default=2.5, help="Env spacing [m].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--difficulty", type=float, default=1.0, help="Trajectory amplitude/speed scale in [0, 1].")
parser.add_argument("--ik_every", type=int, default=2, help="Physics steps per QP update (must divide 4).")
parser.add_argument("--qp_workers", type=int, default=16, help="Worker processes for the per-env QPs (0 = serial).")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint (model_*.pt) to resume from.")
parser.add_argument("--log_dir", type=str, default=None, help="Default: <repo>/logs/base_policy/<timestamp>.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import dataclasses

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.rl.trajectories import TrajectoryCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        torch.manual_seed(args_cli.seed)
        sim = SimulationContext(sim_cfg)
        env_cfg = StabilizationEnvCfg(
            ik_every=args_cli.ik_every, qp_workers=args_cli.qp_workers, trajectory=TrajectoryCfg(difficulty=args_cli.difficulty)
        )
        # The footprint is fixed per env slot at spawn; mass, COM height and payload are re-sampled at every reset.
        params = DomainRandomization().sample_many(args_cli.num_envs, seed=args_cli.seed)
        base_env = RidgebackDREnv(params, sim, spacing=args_cli.spacing)
        base_env.start()
        env = BaseStabilizationEnv(base_env, env_cfg, seed=args_cli.seed)

        log_dir = args_cli.log_dir or str(REPO / "logs" / "base_policy" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        runner_cfg = make_runner_cfg(num_steps_per_env=args_cli.num_steps_per_env)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=str(env.device))
        if args_cli.resume:
            runner.load(args_cli.resume)
        print(f"[INFO] {env.num_envs} envs, policy {1 / env.dt:.0f} Hz, episode {env.max_episode_length} steps, logs: {log_dir}")
        try:
            runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
