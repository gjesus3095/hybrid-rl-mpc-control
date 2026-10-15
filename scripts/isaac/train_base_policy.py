"""Train the RL base policy (PPO, RSL-RL) with a QP arm controller in the loop, on domain-randomized Ridgeback-Franka envs.

The policy outputs base velocity commands (vx, vy, wz); the arm controller (``--controller``: ``ik_qp`` = Pink IK/QP,
``impedance`` = torque-level impedance QP, the folder names under ``controllers/``) tracks a world-frame arm reference
trajectory. Reward: keep the whole-robot COM centred over the platform and stay near the trajectory centre.

Usage (from the repo root, with ISAACLAB_PYTHON set as in the README):

    $ISAACLAB_PYTHON scripts/isaac/train_base_policy.py --controller ik_qp --num_envs 64 --max_iterations 500

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
parser.add_argument("--controller", choices=("ik_qp", "impedance"), default="ik_qp",
                    help="Arm controller: ik_qp (Pink IK/QP, joint position targets) or impedance (impedance QP, torques).")
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
import yaml
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg, saved_observation_cfg
from hybrid_control.rl.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.rl.trajectories import TrajectoryCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization


def _plain(value):
    """Tuples -> lists, so the YAML files hold plain lists."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        torch.manual_seed(args_cli.seed)
        sim = SimulationContext(sim_cfg)
        env_cfg = StabilizationEnvCfg(
            controller=args_cli.controller, ik_every=args_cli.ik_every, qp_workers=args_cli.qp_workers,
            trajectory=TrajectoryCfg(difficulty=args_cli.difficulty),
            # a resumed run keeps the observations (network input size) of the checkpoint it continues
            **({"observation": saved_observation_cfg(args_cli.resume)} if args_cli.resume else {}),
        )
        # The footprint is fixed per env slot at spawn; mass, COM height and payload are re-sampled at every reset.
        params = DomainRandomization().sample_many(args_cli.num_envs, seed=args_cli.seed)
        base_env = RidgebackDREnv(params, sim, spacing=args_cli.spacing)
        base_env.start()
        env = BaseStabilizationEnv(base_env, env_cfg, seed=args_cli.seed)

        log_dir = args_cli.log_dir or str(REPO / "logs" / "base_policy" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        runner_cfg = make_runner_cfg(num_steps_per_env=args_cli.num_steps_per_env)
        # Same as Isaac Lab's rsl_rl train.py: keep the configs next to the checkpoints
        dump_yaml(str(Path(log_dir) / "params" / "env.yaml"), _plain(dataclasses.asdict(env_cfg)))
        dump_yaml(str(Path(log_dir) / "params" / "agent.yaml"), _plain(runner_cfg))
        print("[INFO] PPO configuration:\n" + yaml.dump(_plain(runner_cfg), sort_keys=False, default_flow_style=False))
        runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=str(env.device))
        if args_cli.resume:
            runner.load(args_cli.resume)
        print(f"[INFO] {env.num_envs} envs, arm controller {args_cli.controller}, policy {1 / env.dt:.0f} Hz, episode {env.max_episode_length} steps, logs: {log_dir}")
        try:
            runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
