"""Write the text at several pen-plane heights in parallel (one env per height, no rendering) and compare the tracking.

For every height the same text path (``hybrid_control.rl.evaluation.text_path``, ``plane="table"``) is played with the
trained policy on the same platform. Besides the pen tracking error (x-y error, shape error to the planned stroke, height
error, orientation error) it logs how folded the arm is: the TCP distance from the shoulder, the margin of the arm
joints to their limits, and the Yoshikawa manipulability of the arm Jacobian (the impedance controller keeps it above a
floor, ``ImpedanceCfg.manipulability_min``).

Usage (from the repo root, after the Isaac Lab setup in the root README):

    $ISAACLAB_PYTHON scripts/isaac/test_text_heights.py --heights 0.40 0.55 0.70
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from hybrid_control.rl.config import CONTROLLERS, saved_controller, saved_observation_cfg

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Compare pen-plane heights for the writing task.")
parser.add_argument("--checkpoint", type=str, default=str(ROOT / "logs/base_policy/impedance_quintic_run1/model_1200.pt"))
parser.add_argument("--heights", type=float, nargs="+", default=[0.38, 0.42, 0.46, 0.50, 0.55, 0.65, 0.75, 0.85])
parser.add_argument("--x_table", type=float, default=0.58, help="x of the middle of the letters [m].")
parser.add_argument("--text", type=str, default="MPCRL2026")
parser.add_argument("--controller", choices=CONTROLLERS, default=None)
parser.add_argument("--footprint", type=float, nargs=2, default=(0.45, 0.40))
parser.add_argument("--base_mass", type=float, default=30.0)
parser.add_argument("--payload", type=float, default=1.5)
parser.add_argument("--write_speed", type=float, default=0.15)
parser.add_argument("--spacing", type=float, default=8.0)
parser.add_argument("--qp_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out_dir", type=str, default=str(ROOT / "out/mpcrl2026/height_sweep"))
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.controller = args_cli.controller or saved_controller(args_cli.checkpoint)

import json
import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.evaluation.scenarios import REF_DT, ScenarioSet
from hybrid_control.rl.evaluation.scripted import ScriptedTrajectories
from hybrid_control.rl.evaluation.text_path import TextCfg, build_text_path
from hybrid_control.rl.policy_with_controller.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import BaseParams
from hybrid_control.utils.torch_rotations import quat_angle

SHOULDER_Z = 0.613  # arm shoulder height at spawn [m]
MANIPULABILITY_FLOOR = 0.03  # ImpedanceCfg.manipulability_min


def main():
    heights = list(args_cli.heights)
    paths = [build_text_path(TextCfg(controller=args_cli.controller, text=args_cli.text, plane="table", z_table=h,
                                     x_table=args_cli.x_table, write_speed=args_cli.write_speed)) for h in heights]
    n, t_len = len(paths), max(len(p.poses) for p in paths)
    poses = np.stack([np.concatenate([p.poses, np.repeat(p.poses[-1:], t_len - len(p.poses), axis=0)]) for p in paths])
    params = BaseParams(tuple(args_cli.footprint), args_cli.base_mass, 0.10, args_cli.payload)
    scenario = ScenarioSet("heights", poses, np.array([p.duration for p in paths], np.float32), np.zeros((n, 3), np.float32),
                           np.array([[params.base_mass, params.base_com_z, params.payload]] * n, np.float32))
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        base_env = RidgebackDREnv([params] * n, sim, spacing=args_cli.spacing, num_cols=math.ceil(math.sqrt(n)))
        base_env.start()
        env = BaseStabilizationEnv(base_env, StabilizationEnvCfg(controller=args_cli.controller, qp_workers=args_cli.qp_workers,
                                                                 episode_length_s=3600.0, observation=saved_observation_cfg(args_cli.checkpoint)),
                                   seed=args_cli.seed)
        env.auto_reset = False
        env.traj = ScriptedTrajectories(n, env.device, env.scfg.trajectory)
        env.traj.load(scenario)
        runner = OnPolicyRunner(env, make_runner_cfg(), log_dir=None, device=str(env.device))
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(env.device))
        limits = env.arm.limits_array  # (7, 2)
        steps = int(math.ceil(max(p.duration for p in paths) / env.dt))
        keys = ("t", "ref", "tip", "ori", "rho", "failed", "base", "yaw", "q", "manip")
        log = {k: [] for k in keys}
        with torch.inference_mode():
            env._reset_idx(torch.arange(n, device=env.device), dynamics=[params] * n)
            env._obs = env._compute_obs()
            obs = env.get_observations()
            for step in range(steps):
                obs, _, _, _ = env.step(policy(obs))
                t = env.episode_length_buf.float() * env.dt
                tip = env.arm.ee_pose_t()[0] - base_env.origins
                ref, ref_quat = env.traj.pose(t)
                jac = env.arm.jacobian  # (n, 6, 7) at the last QP update
                log["t"].append(t.cpu().numpy())
                log["ref"].append(ref.cpu().numpy()), log["tip"].append(tip.cpu().numpy())
                log["ori"].append(quat_angle(env.arm.ee_pose_t()[1], ref_quat).cpu().numpy())
                log["rho"].append((base_env.com_offset / base_env.half_length).cpu().numpy())
                log["failed"].append((base_env.tipped | base_env.left_footprint).cpu().numpy())
                log["base"].append(base_env.base_pos[:, :2].cpu().numpy()), log["yaw"].append(base_env.yaw.cpu().numpy())
                log["q"].append(env.arm.q.copy())
                log["manip"].append(np.sqrt(np.maximum(np.linalg.det(jac @ jac.transpose(0, 2, 1)), 0.0)))
                if step % 600 == 0:
                    print(f"[HEIGHTS] step {step}/{steps}", flush=True)
        env.close()

        L = {k: np.stack(v, axis=1) for k, v in log.items()}  # (n, T, ...)
        np.savez_compressed(out_dir / "log.npz", heights=np.array(heights), **L)
        results = []
        for i, (h, path) in enumerate(zip(heights, paths)):
            idx = np.minimum(np.round(L["t"][i] / REF_DT).astype(int), len(path.poses) - 1)
            ref, tip = L["ref"][i], L["tip"][i]
            speed = np.linalg.norm(np.gradient(ref[:, :2], 1 / 30, axis=0), axis=1)
            m = path.pen_down[idx] & (speed > 0.02)
            # shape error: distance of the written point to the planned stroke of its letter
            shape = np.full(len(m), np.nan)
            for k, st in enumerate(path.strokes):
                plan = np.stack([st["yz"][:, 1], st["yz"][:, 0]], axis=1)  # table plane: x is the second in-plane coordinate
                sel = np.flatnonzero(m & (path.letter[idx] == k))
                if len(sel):
                    shape[sel] = np.sqrt(((tip[sel, None, :2] - plan[None]) ** 2).sum(-1)).min(1)
            c, s = np.cos(L["yaw"][i]), np.sin(L["yaw"][i])
            mount = L["base"][i] + 0.30 * np.stack([c, s], axis=1)
            d = tip[:, :2] - mount
            fwd = c * d[:, 0] + s * d[:, 1]
            dist = np.sqrt((d**2).sum(1) + (tip[:, 2] - SHOULDER_Z) ** 2)
            q = L["q"][i]
            margin_per_joint = np.minimum(q - limits[:, 0], limits[:, 1] - q)  # (T, 7)
            margin = margin_per_joint.min(1)
            exy = np.linalg.norm((tip - ref)[:, :2], axis=1)
            failed = L["failed"][i].any()
            results.append({
                "height_m": h, "failed": bool(failed),
                "xy_err_cm": {"rms": float(100 * np.sqrt(np.mean(exy[m] ** 2))), "mean": float(100 * exy[m].mean()),
                              "p95": float(100 * np.percentile(exy[m], 95)), "max": float(100 * exy[m].max())},
                "shape_err_cm": {"rms": float(100 * np.sqrt(np.nanmean(shape[m] ** 2))), "max": float(100 * np.nanmax(shape[m]))},
                "z_err_cm_rms": float(100 * np.sqrt(np.mean((tip - ref)[m, 2] ** 2))),
                "ori_err_deg_median": float(np.degrees(np.median(L["ori"][i][m]))),
                "rho_max": float(L["rho"][i].max()),
                "pen_dist_from_shoulder_m": {"median": float(np.median(dist[m])), "p10": float(np.percentile(dist[m], 10))},
                "pen_forward_of_mount_m_median": float(np.median(fwd[m])),
                "base_x_range_m": [float(L["base"][i][m, 0].min()), float(L["base"][i][m, 0].max())],
                "joint_margin_rad": {"median": float(np.median(margin[m])), "p5": float(np.percentile(margin[m], 5)),
                                     "share_below_0.1": float(np.mean(margin[m] < 0.1)),
                                     "closest_joint_counts": np.bincount(margin_per_joint[m].argmin(1), minlength=7).tolist()},
                "manipulability": {"median": float(np.median(L["manip"][i][m])), "p5": float(np.percentile(L["manip"][i][m], 5)),
                                   "share_below_floor": float(np.mean(L["manip"][i][m] < MANIPULABILITY_FLOOR))},
            })
        (out_dir / "summary.json").write_text(json.dumps({"x_table": args_cli.x_table, "text": args_cli.text, "results": results}, indent=2))
        head = (f"{'z [m]':>6s} {'fail':>4s} {'xy rms':>7s} {'xy p95':>7s} {'xy max':>7s} {'shape rms':>9s} {'z rms':>6s} {'ori med':>7s} "
                f"{'dist shoulder':>13s} {'jlim p5':>8s} {'jlim<0.1':>8s} {'manip p5':>8s} {'manip<floor':>11s}")
        print("\n[RESULT] text '%s' at x = %.2f m, pen heights compared (errors in cm unless noted)\n" % (args_cli.text, args_cli.x_table) + head)
        for r in results:
            print(f"{r['height_m']:6.2f} {'YES' if r['failed'] else 'no':>4s} {r['xy_err_cm']['rms']:7.2f} {r['xy_err_cm']['p95']:7.2f} "
                  f"{r['xy_err_cm']['max']:7.2f} {r['shape_err_cm']['rms']:9.2f} {r['z_err_cm_rms']:6.2f} {r['ori_err_deg_median']:7.1f} "
                  f"{r['pen_dist_from_shoulder_m']['median']:13.2f} {r['joint_margin_rad']['p5']:8.2f} {100 * r['joint_margin_rad']['share_below_0.1']:7.0f}% "
                  f"{r['manipulability']['p5']:8.3f} {100 * r['manipulability']['share_below_floor']:10.0f}%", flush=True)


if __name__ == "__main__":
    main()
