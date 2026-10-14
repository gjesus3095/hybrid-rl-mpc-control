"""Time ParallelPinkIK (Pink + DAQP in CPU worker processes) against the sequential loop.

No simulator: each environment gets a random arm configuration near Q_INIT on a random base pose,
and a TCP target displaced by about one control step of base motion (as in the Isaac base
disturbance script), plus a fraction of larger jumps that activate velocity bounds.

    PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/benchmark_parallel_ik.py
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from hybrid_control.utils.control_utils import IKQPConfig
from hybrid_control.controllers.ik_qp.ik_qp import DEFAULT_URDF, build_model
from hybrid_control.controllers.ik_qp.parallel import ParallelPinkIK, physical_cpus
from hybrid_control.robots.state import Pose

ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
ROOT, TCP = "panda_link0", "panda_fingertip_centered"
TCP_OFFSET = Pose(np.array([0.0, 0.0, 0.1121]), np.array([0.0, 0.0, 0.0, 1.0]))
DT = 0.02  # 50 Hz control, as in scripts/isaac/base_disturbance_keyboard.py
DQ_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
CONFIG = IKQPConfig(-DQ_MAX, DQ_MAX, kp_pos=50.0, kp_ori=100.0, damping=1e5)
Q_INIT = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])


def make_batch(num_envs: int, rng: np.random.Generator):
    model = build_model(DEFAULT_URDF, {TCP: ("panda_hand", TCP_OFFSET)})
    data = model.createData()
    idx_q = [model.joints[model.getJointId(n)].idx_q for n in ARM_JOINTS]
    limits = np.stack([model.lowerPositionLimit[idx_q], model.upperPositionLimit[idx_q]], axis=1)
    q = np.clip(Q_INIT + rng.uniform(-0.3, 0.3, (num_envs, 7)), limits[:, 0] + 1e-3, limits[:, 1] - 1e-3)
    root, target = np.empty((num_envs, 7)), np.empty((num_envs, 7))
    for env in range(num_envs):
        base = pin.SE3(Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix(),
                       np.r_[rng.uniform(-5, 5, 2), 0.4])
        q_pin = pin.neutral(model)
        q_pin[idx_q] = q[env]
        pin.framesForwardKinematics(model, data, q_pin)
        tcp = base * data.oMf[model.getFrameId(ROOT)].actInv(data.oMf[model.getFrameId(TCP)])
        scale = 0.05 if rng.random() < 0.2 else 0.3 * DT  # 20% large jumps -> active velocity bounds
        position = tcp.translation + rng.normal(size=3) * scale
        rotation = Rotation.from_rotvec(rng.normal(size=3) * scale) * Rotation.from_matrix(tcp.rotation)
        root[env] = np.r_[base.translation, Rotation.from_matrix(base.rotation).as_quat()]
        target[env] = np.r_[position, rotation.as_quat()]
    return q, np.broadcast_to(limits, (num_envs, 7, 2)).copy(), root, target


def time_calls(pool, batch, steps):
    pool.compute(*batch, TCP, DT)  # warm-up
    wall, worker_max = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        result = pool.compute(*batch, TCP, DT)
        wall.append(time.perf_counter() - t0)
        worker_max.append(pool.worker_times.max())
    return np.array(wall), np.array(worker_max), result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 16, 50, 100, 200, 300, 500])
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 24, 32, 48])
    parser.add_argument("--steps", type=int, default=30, help="Timed calls per configuration.")
    parser.add_argument("--no-pin", action="store_true", help="Do not pin workers to physical cores.")
    parser.add_argument("--out", default="logs/benchmark_parallel_ik.csv")
    args = parser.parse_args()

    cpus = None if args.no_pin else physical_cpus()
    max_workers = max(args.workers)
    if cpus is not None and len(cpus) < max_workers:
        parser.error(f"Only {len(cpus)} physical cores for {max_workers} workers; use --no-pin")
    batches = {n: make_batch(n, np.random.default_rng(n)) for n in args.envs}
    reference = {}
    rows = []
    print(f"{'envs':>5} {'workers':>7} {'median ms':>10} {'p95 ms':>8} {'per env us':>11} "
          f"{'slowest worker ms':>18} {'speedup':>8} {'failed':>6} {'max |dq diff|':>14}")
    for workers in args.workers:
        # Workers fork before any configuration, as they must before Isaac Sim starts
        with ParallelPinkIK(workers, cpus=None if cpus is None else cpus[1:workers + 1] or cpus) as pool:
            pool.configure(ARM_JOINTS, CONFIG, root_frame=ROOT, solver="daqp",
                           extra_frames={TCP: ("panda_hand", TCP_OFFSET)})
            for n in args.envs:
                if workers > n:
                    continue
                wall, worker_max, (q_cmd, ok) = time_calls(pool, batches[n], args.steps)
                if workers == 0:
                    reference[n] = (q_cmd, np.median(wall))
                diff = np.abs(q_cmd - reference[n][0]).max() / DT if n in reference else np.nan
                speedup = reference[n][1] / np.median(wall) if n in reference else np.nan
                row = dict(envs=n, workers=workers, median_ms=1e3 * np.median(wall),
                           p95_ms=1e3 * np.percentile(wall, 95), per_env_us=1e6 * np.median(wall) / n,
                           worker_max_ms=1e3 * np.median(worker_max), speedup=speedup,
                           failed=int((~ok).sum()), max_dq_diff=diff)
                rows.append(row)
                print(f"{n:5d} {workers:7d} {row['median_ms']:10.2f} {row['p95_ms']:8.2f} {row['per_env_us']:11.1f} "
                      f"{row['worker_max_ms']:18.2f} {speedup:8.1f} {row['failed']:6d} {diff:14.2e}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
