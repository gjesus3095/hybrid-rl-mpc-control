"""Run a trained base policy (deterministic) with its QP arm controller in the loop and report how many episodes survive.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    $ISAACLAB_PYTHON scripts/isaac/with_controller/play_base_policy.py --checkpoint logs/base_policy/<run>/model_499.pt
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from hybrid_control.rl.config import CONTROLLERS, saved_controller, saved_observation_cfg

parser = argparse.ArgumentParser(description="Evaluate a trained base policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=2, help="Episodes per env to run.")
parser.add_argument("--spacing", type=float, default=2.5)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--difficulty", type=float, default=1.0)
parser.add_argument("--qp_workers", type=int, default=8, help="Worker processes for the per-env QPs (0 = serial).")
parser.add_argument("--controller", choices=CONTROLLERS, default=None,
                    help="Arm controller; default: the one the checkpoint was trained with (<run>/params/env.yaml).")
parser.add_argument("--zero_policy", action="store_true", help="Baseline: send zero base commands instead of the policy.")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.controller = args_cli.controller or saved_controller(args_cli.checkpoint)

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.policy_with_controller.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.rl.policy_with_controller.trajectories import TrajectoryCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        params = DomainRandomization().sample_many(args_cli.num_envs, seed=args_cli.seed)
        base_env = RidgebackDREnv(params, sim, spacing=args_cli.spacing)
        base_env.start()
        env = BaseStabilizationEnv(
            base_env, StabilizationEnvCfg(controller=args_cli.controller, qp_workers=args_cli.qp_workers, trajectory=TrajectoryCfg(difficulty=args_cli.difficulty), observation=saved_observation_cfg(args_cli.checkpoint)), seed=args_cli.seed
        )
        runner = OnPolicyRunner(env, make_runner_cfg(), log_dir=None, device=str(env.device))
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(env.device))

        obs = env.get_observations()
        finished = failed = steps = 0
        rho_sum = 0.0
        base = env.env
        motion = {"speed": 0.0, "yaw_rate": 0.0, "yaw": 0.0, "dist": 0.0, "dact": 0.0}
        prev_action = torch.zeros(env.num_envs, 3, device=env.device)
        target = args_cli.episodes * env.num_envs
        with torch.inference_mode():
            while finished < target:
                actions = torch.zeros(env.num_envs, 3, device=env.device) if args_cli.zero_policy else policy(obs)
                obs, _, dones, extras = env.step(actions)
                steps += 1
                rho_sum += float(extras["log"]["Metrics/com_rho"])
                up = ~base.tipped  # base-motion statistics over envs that are still upright
                if up.any():
                    vel = base.robot.data.body_com_lin_vel_w.torch[:, base.base_id, :2].norm(dim=-1)
                    motion["speed"] += float(vel[up].mean())
                    motion["yaw_rate"] += float(base.robot.data.body_com_ang_vel_w.torch[up, base.base_id, 2].abs().mean())
                    motion["yaw"] += float(base.yaw[up].abs().mean())
                    motion["dist"] += float(base.base_pos[up, :2].norm(dim=-1).mean())
                    motion["dact"] += float((actions - prev_action)[up].norm(dim=-1).mean())
                prev_action = actions
                finished += int(dones.sum())
                failed += int(extras["failed"].sum())
        env.close()
        print(f"[RESULT] {finished} episodes, {failed} failed ({100.0 * failed / finished:.1f}%), "
              f"mean COM offset {rho_sum / steps:.3f} of the half footprint")
        print(f"[MOTION] base speed {motion['speed'] / steps:.3f} m/s, |yaw rate| {motion['yaw_rate'] / steps:.3f} rad/s, "
              f"|yaw| {motion['yaw'] / steps:.3f} rad, distance from spawn {motion['dist'] / steps:.2f} m, "
              f"|action change| {motion['dact'] / steps:.3f} per step (upright envs)")


if __name__ == "__main__":
    main()
