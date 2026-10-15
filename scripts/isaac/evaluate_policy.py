"""Play a policy (with its QP arm controller in the loop) on the deterministic evaluation benchmark and log everything.

The benchmark (``hybrid_control.rl.evaluation.scenarios``) is the same for every policy: five sets of 100 scenarios
(fixed points, sines, quintic A -> B, quintic with B outside the training reach, rigid straight / bent paths), each with
its own reference, base mass, COM height and payload. Env slot ``j`` has footprint ``j`` in every set. Each set is played
once, all scenarios in parallel, until each scenario's own duration ends or the platform fails (tips / COM leaves the
footprint). Per-step logs (``LOG_FIELDS`` in ``metrics.py``: reference, TCP pose and errors, base state, COM and its
margin to the footprint edge, arm joints, actions) and summary statistics are written to ``--out_dir``:

    <out_dir>/<set>.npz   per-step logs (S, T, dim) + valid mask, failure step, durations, dynamics, footprints
    <out_dir>/summary.json   RMS / std / percentiles of the errors, COM margin, failures, base motion per set

Usage (from the repo root, after the Isaac Lab setup in the root README):

    $ISAACLAB_PYTHON scripts/isaac/evaluate_policy.py --checkpoint logs/base_policy/<run>/model_999.pt
    $ISAACLAB_PYTHON scripts/isaac/evaluate_policy.py --zero_policy               # baseline: base commanded to stand still

Replays differ only by the nondeterminism of the GPU physics (add ``--deterministic`` to the launcher arguments to reduce it).
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from hybrid_control.rl.config import CONTROLLERS, saved_controller, saved_observation_cfg

parser = argparse.ArgumentParser(description="Evaluate a policy on the deterministic benchmark.")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--zero_policy", action="store_true", help="Baseline: send zero base commands instead of the policy.")
parser.add_argument("--benchmark", type=str, default=None,
                    help="Benchmark .npz from scripts/generate_eval_set.py (for the same --controller); default: generate it in memory.")
parser.add_argument("--sets", nargs="+", default=None, help="Sets to run (default: all in the benchmark).")
parser.add_argument("--num_scenarios", type=int, default=None,
                    help="Only the first N scenarios of every set, e.g. for a smoke test (same scenarios as in the full run).")
parser.add_argument("--spacing", type=float, default=8.0,
                    help="Env spacing [m]; the far-B sets drive the base up to ~2 m, so keep neighbours well apart.")
parser.add_argument("--qp_workers", type=int, default=16, help="Worker processes for the per-env QPs (0 = serial).")
parser.add_argument("--controller", choices=CONTROLLERS, default=None,
                    help="Arm controller; default: the one the checkpoint was trained with (<run>/params/env.yaml).")
parser.add_argument("--current_observation", action="store_true",
                    help="Use the current ObservationCfg defaults instead of <run>/params/env.yaml (for checkpoints without one).")
parser.add_argument("--payloads", type=float, nargs="+", default=None,
                    help="Extra mass at the gripper [kg]: run everything once per value with this payload in every scenario "
                         "(other dynamics unchanged) -> <out_dir>/payload_<kg>kg/. Default: each scenario's own payload.")
parser.add_argument("--out_dir", type=str, default=None, help="Default: out/eval/<name>.")
parser.add_argument("--name", type=str, default=None, help="Run name; default: <run dir>_<checkpoint> or zero_policy.")
parser.add_argument("--seed", type=int, default=1)
add_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.checkpoint and not args_cli.zero_policy:
    parser.error("give --checkpoint or --zero_policy")
args_cli.controller = args_cli.controller or (saved_controller(args_cli.checkpoint) if args_cli.checkpoint else "ik_qp")

import dataclasses
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.evaluation.metrics import LOG_FIELDS, failure_steps, format_table, summarize, valid_mask
from hybrid_control.rl.evaluation.scenarios import SET_NAMES, Benchmark, BenchmarkCfg, generate_benchmark
from hybrid_control.rl.evaluation.scripted import ScriptedTrajectories
from hybrid_control.rl.policy_with_controller.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.rl.policy_with_controller.observations import ObservationCfg, to_base_frame
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.utils.torch_rotations import quat_angle


def snapshot(env: BaseStabilizationEnv, action: torch.Tensor) -> torch.Tensor:
    """(N, D) row of every ``LOG_FIELDS`` quantity after the last step, concatenated in ``LOG_FIELDS`` order."""
    b = env.env
    t = env.episode_length_buf.float() * env.dt
    ref_pos, ref_quat = env.traj.pose(t)
    ee_pos, ee_quat = env.arm.ee_pose_t()
    ee_pos = ee_pos - b.origins
    err = ee_pos - ref_pos
    c, s = torch.cos(b.yaw), torch.sin(b.yaw)
    com_total = b.com_total - b.origins
    rel = com_total[:, :2] - env._centre_xy()
    dx, dy = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]  # platform-frame COM offset
    hl, hw = b.half_length, b.half_width
    robot = b.robot
    fields = {
        "t": t[:, None], "ref_pos": ref_pos, "ref_quat": ref_quat, "ee_pos": ee_pos, "ee_quat": ee_quat,
        "pos_err_vec": err, "pos_err": err.norm(dim=-1, keepdim=True), "ori_err": quat_angle(ee_quat, ref_quat)[:, None],
        "base_pos": b.base_pos[:, :2], "yaw": b.yaw[:, None], "roll": b.roll[:, None], "pitch": b.pitch[:, None],
        "base_vel": to_base_frame(b.yaw, robot.data.body_com_lin_vel_w.torch[:, b.base_id])[:, :2],
        "yaw_rate": robot.data.body_com_ang_vel_w.torch[:, b.base_id, 2:3],
        "com_total": com_total, "com_arm": b.com_arm - b.origins, "com_xy_platform": torch.stack([dx, dy], dim=-1),
        "half_size": torch.stack([hl, hw], dim=-1),
        "rho": torch.maximum(dx.abs() / hl, dy.abs() / hw)[:, None],
        "margin_m": torch.minimum(hl - dx.abs(), hw - dy.abs())[:, None],
        "tipped": b.tipped.float()[:, None], "left_footprint": b.left_footprint.float()[:, None],
        "arm_q": robot.data.joint_pos.torch[:, b.arm_ids], "action": action,
    }
    for name, dim in LOG_FIELDS.items():
        assert fields[name].shape[-1] == dim, (name, fields[name].shape)
    return torch.cat([fields[name].float() for name in LOG_FIELDS], dim=-1)


@torch.inference_mode()  # the env keeps inference tensors (as in play_base_policy.py), so resets must run in it too
def run_set(env, policy, benchmark: Benchmark, name: str, slots: int, zero_policy: bool, payload: float | None = None):
    """Play set ``name`` once (every scenario with ``payload`` kg at the gripper if given); returns the per-step logs
    (S, T, dim), the valid mask and the first failure step per scenario."""
    scenario = benchmark.sets[name]
    env.traj.load(scenario)
    params = benchmark.params(name, slots)
    if payload is not None:
        params = [dataclasses.replace(p, payload=payload) for p in params]
    env._reset_idx(torch.arange(env.num_envs, device=env.device), dynamics=params)
    env._obs = env._compute_obs()
    n_steps = np.floor(scenario.duration[:slots] / env.dt + 1e-6).astype(int)  # same count as the 5 s = 150 steps of training
    obs, rows, ik_before = env.get_observations(), [], env.ik_failures
    for _ in range(int(n_steps.max())):
        actions = torch.zeros(env.num_envs, 3, device=env.device) if zero_policy else policy(obs)
        obs, _, _, _ = env.step(actions)
        rows.append(snapshot(env, actions.clamp(-1.0, 1.0)))
    stacked = torch.stack(rows, dim=1).cpu().numpy()  # (S, T, D)
    log, start = {}, 0
    for field, dim in LOG_FIELDS.items():
        log[field] = stacked[..., start:start + dim]
        start += dim
    fail_step = failure_steps((log["tipped"][..., 0] > 0) | (log["left_footprint"][..., 0] > 0), n_steps)
    valid = valid_mask(n_steps, fail_step, stacked.shape[1])
    return log, valid, fail_step, n_steps, env.ik_failures - ik_before


def git_state() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
        return commit + ("+dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def main():
    benchmark = (Benchmark.load(args_cli.benchmark) if args_cli.benchmark
                 else generate_benchmark(BenchmarkCfg(controller=args_cli.controller)))
    if benchmark.cfg.controller != args_cli.controller:
        raise SystemExit(f"benchmark was generated for the {benchmark.cfg.controller!r} controller (TCP spawn pose), "
                         f"not {args_cli.controller!r}")
    names = args_cli.sets or [n for n in SET_NAMES if n in benchmark.sets]
    slots = args_cli.num_scenarios or benchmark.cfg.n_per_set
    ckpt = args_cli.checkpoint
    name = args_cli.name or (f"{Path(ckpt).resolve().parent.name}_{Path(ckpt).stem}" if ckpt else "zero_policy")
    out_dir = Path(args_cli.out_dir or Path(__file__).resolve().parents[2] / "out" / "eval" / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EVAL] benchmark {benchmark.fingerprint()}, sets {names}, {slots} scenarios each -> {out_dir}", flush=True)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        base_env = RidgebackDREnv(benchmark.params(names[0], slots), sim, spacing=args_cli.spacing)
        base_env.start()
        env = BaseStabilizationEnv(
            base_env,
            StabilizationEnvCfg(controller=args_cli.controller, qp_workers=args_cli.qp_workers, episode_length_s=3600.0,
                                observation=saved_observation_cfg(ckpt) if ckpt and not args_cli.current_observation else ObservationCfg()),
            seed=args_cli.seed,
        )
        env.auto_reset = False  # the runner owns the episodes: each set is one pass over all slots
        env.traj = ScriptedTrajectories(env.num_envs, env.device, env.scfg.trajectory)
        policy = None
        if ckpt:
            runner = OnPolicyRunner(env, make_runner_cfg(), log_dir=None, device=str(env.device))
            runner.load(ckpt)
            policy = runner.get_inference_policy(device=str(env.device))

        for payload in args_cli.payloads or [None]:
            run_dir = out_dir if payload is None else out_dir / f"payload_{payload:g}kg"
            run_dir.mkdir(parents=True, exist_ok=True)
            label = "" if payload is None else f" at {payload:g} kg payload"
            summaries = {}
            for set_name in names:
                log, valid, fail_step, n_steps, ik_failures = run_set(env, policy, benchmark, set_name, slots,
                                                                      args_cli.zero_policy, payload)
                scenario = benchmark.sets[set_name]
                summaries[set_name] = summarize(log, valid, fail_step, env.dt, n_steps=n_steps)
                summaries[set_name]["ik_failures"] = int(ik_failures)
                dynamics = scenario.dynamics[:slots].copy()
                if payload is not None:
                    dynamics[:, 2] = payload
                np.savez_compressed(
                    run_dir / f"{set_name}.npz", **log, valid=valid, fail_step=fail_step, n_steps=n_steps, dt=env.dt,
                    duration=scenario.duration[:slots], dynamics=dynamics, footprints=benchmark.footprints[:slots],
                )
                print(f"[EVAL]{label} {set_name}: {int((fail_step >= 0).sum())}/{slots} failed", flush=True)

            # still inside launch_simulation: leaving it shuts Kit (and the process) down
            meta = {
                "name": name, "checkpoint": ckpt, "zero_policy": args_cli.zero_policy, "controller": args_cli.controller,
                "payload_kg": payload, "benchmark_seed": benchmark.cfg.seed, "scenarios_per_set": slots,
                "policy_dt": env.dt, "git": git_state(), "date": datetime.now().isoformat(timespec="seconds"),
            }
            merged, fingerprints = dict(summaries), {n: benchmark.fingerprint([n]) for n in summaries}
            summary_path = run_dir / "summary.json"
            if summary_path.exists():  # add to an earlier run of the same policy (e.g. a set added to the benchmark later)
                before = json.loads(summary_path.read_text())
                keys = ("checkpoint", "zero_policy", "controller", "scenarios_per_set", "payload_kg")
                if all(before["meta"].get(k) == meta[k] for k in keys):
                    merged = {**before["sets"], **summaries}
                    fingerprints = {**before["meta"].get("set_fingerprints", {}), **fingerprints}
            merged = {n: merged[n] for n in SET_NAMES if n in merged}
            meta["benchmark_fingerprint"] = benchmark.fingerprint(list(merged))  # over the sets in this summary
            meta["set_fingerprints"] = fingerprints
            summary_path.write_text(json.dumps({"meta": meta, "sets": merged}, indent=2))
            print(f"\n[RESULT] {name}{label} (benchmark {meta['benchmark_fingerprint']}, {slots} scenarios per set)\n"
                  + format_table(merged), flush=True)
        env.close()


if __name__ == "__main__":
    main()
