"""Integration check of the RL env with the QP in the loop (no training).

1. Jacobian: ``J @ qd`` from the adapter must match the finite-difference end-effector velocity.
2. QP tracking: with zero base commands the end-effector follows the world-frame reference.
3. Partial reset: resetting some envs must not disturb the others.
4. Throughput: wall-clock cost of the QP loop and of whole policy steps.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    $ISAACLAB_PYTHON scripts/isaac/check_stabilization_env.py --num_envs 16
"""

import argparse
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Check the base-stabilization RL env.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--qp_workers", type=int, default=0, help="QP worker processes (0 = serial).")
parser.add_argument("--kp_pos", type=float, default=None, help="Override the QP position gain.")
parser.add_argument("--steps", type=int, default=240, help="Policy steps for the tracking/throughput runs.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from hybrid_control.rl.policy_with_controller.env import BaseStabilizationEnv, IKCfg, StabilizationEnvCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        params = DomainRandomization().sample_many(args_cli.num_envs, seed=args_cli.seed)
        base_env = RidgebackDREnv(params, sim, spacing=2.5)
        base_env.start()
        env = BaseStabilizationEnv(base_env, StabilizationEnvCfg(qp_workers=args_cli.qp_workers, **({"ik": IKCfg(kp_pos=args_cli.kp_pos)} if args_cli.kp_pos else {})), seed=args_cli.seed)
        n, dev = env.num_envs, env.device
        zero = torch.zeros(n, 3, device=dev)
        print(f"[INFO] obs: actor {tuple(env.get_observations()['actor'].shape)}, critic {tuple(env.get_observations()['critic'].shape)}")

        # 1. Jacobian vs finite differences, while the arm moves under the QP
        errs = []
        for _ in range(20):
            env.step(zero)
        arm = env.arm
        for _ in range(10):
            env._update_arm_command(0.0)
            arm.refresh()
            p0, J, qd = arm.ee_pos.copy(), arm.jacobian.copy(), arm.qd.copy()
            base_env.step(env.q_cmd, zero)
            arm.refresh()
            v_fd = (arm.ee_pos - p0) / base_env.dt
            v_j = np.einsum("nij,nj->ni", J[:, :3], qd)
            errs.append(np.linalg.norm(v_fd - v_j, axis=1) / (np.linalg.norm(v_fd, axis=1) + 1e-2))
        print(f"[CHECK 1] Jacobian vs finite-difference EE velocity: median rel. error {np.median(errs):.3f}, 95th pct {np.percentile(errs, 95):.3f}")

        # 2. Tracking with a static base: every reference is reachable from the spawn pose, so the QP alone should track
        # it. Report the end-effector error of the envs that have not tipped (zero base command, no resets).
        env.auto_reset = False
        env._reset_idx(torch.arange(n, device=dev))
        t0 = time.time()
        for step in range(args_cli.steps):
            env.step(zero)
        wall = time.time() - t0
        t_now = env.episode_length_buf.float() * env.dt
        err = (env.arm.ee_pos_t() - base_env.origins - env.traj.position(t_now)).norm(dim=-1)
        ok = ~base_env.tipped
        e = err[ok].cpu().numpy()
        print(f"[CHECK 2] static-base EE tracking error after {float(t_now[0]):.1f} s, {int(ok.sum())}/{n} envs upright: "
              f"median {np.median(e):.3f} m, 90th pct {np.percentile(e, 90):.3f} m, max {e.max():.3f} m, "
              f"{100 * float((e < 0.05).mean()):.0f}% within 5 cm; QP failures: {env.ik_failures}")
        sps = args_cli.steps * n / wall
        print(f"[CHECK 4] {args_cli.steps} policy steps x {n} envs in {wall:.1f}s -> {sps:.0f} env-steps/s "
              f"({1e3 * wall / args_cli.steps:.0f} ms per policy step)")
        env.auto_reset = True

        # 3. Partial reset leaves other envs untouched
        env.step(zero)
        before = base_env.base_pos.clone(), base_env.robot.data.joint_pos.torch.clone()
        ids = torch.arange(0, n, 2, device=dev)
        base_env.reset(ids)
        after = base_env.base_pos.clone(), base_env.robot.data.joint_pos.torch.clone()
        keep = torch.arange(1, n, 2, device=dev)
        same = torch.allclose(before[0][keep], after[0][keep]) and torch.allclose(before[1][keep], after[1][keep])
        moved = (before[1][ids] - after[1][ids]).abs().max() > 1e-4 or (before[0][ids] - after[0][ids]).abs().max() > 1e-4
        print(f"[CHECK 3] partial reset: untouched envs unchanged={same}, reset envs changed={bool(moved)}")

        # random actions: no NaNs, resets work, episodes terminate
        done_total = 0
        for _ in range(100):
            obs, rew, dones, extras = env.step(2 * torch.rand(n, 3, device=dev) - 1)
            done_total += int(dones.sum())
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(rew).all()
        env.close()
        print(f"[CHECK 5] 100 random-action steps: {done_total} episode ends, all observations/rewards finite")


if __name__ == "__main__":
    main()
