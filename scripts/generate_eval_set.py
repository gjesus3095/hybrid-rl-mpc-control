"""Generate (and optionally plot) the deterministic evaluation benchmark. No Isaac needed, only torch / numpy.

    python scripts/generate_eval_set.py                 # prints the fingerprint and per-set statistics
    python scripts/generate_eval_set.py --save out/eval/benchmark.npz --plot out/eval/benchmark.png
    python scripts/generate_eval_set.py --spatial_order 3     # poly / poly_far along cubic Bezier paths

The same seed always gives the same benchmark (same fingerprint). ``scripts/isaac/evaluate_policy.py`` regenerates it in
memory by default, so saving is only needed to share or archive the exact scenarios (about 6 MB).
"""

import argparse
from pathlib import Path

import numpy as np

from hybrid_control.rl.evaluation.scenarios import REF_DT, RIGID_TYPES, SET_NAMES, SPAWN_POSES, BenchmarkCfg, generate_benchmark

parser = argparse.ArgumentParser(description="Generate the deterministic evaluation benchmark.")
parser.add_argument("--seed", type=int, default=BenchmarkCfg.seed)
parser.add_argument("--controller", choices=sorted(SPAWN_POSES), default="ik_qp", help="Arm controller (fixes the TCP spawn pose).")
parser.add_argument("--n_per_set", type=int, default=BenchmarkCfg.n_per_set)
parser.add_argument("--spatial_order", type=int, default=BenchmarkCfg().trajectory.spatial_order,
                    help="Bezier order of the poly / poly_far paths under the quintic time scaling (1 = straight segments).")
parser.add_argument("--save", type=str, default=None, help="Write the benchmark to this .npz.")
parser.add_argument("--plot", type=str, default=None, help="Write a top-view / side-view overview to this .png.")
parser.add_argument("--plot_count", type=int, default=25, help="Scenarios per set drawn in the plot (the first N).")
args = parser.parse_args()

benchmark = generate_benchmark(BenchmarkCfg.with_spatial_order(args.spatial_order, seed=args.seed, n_per_set=args.n_per_set,
                                                               controller=args.controller))
print(f"benchmark fingerprint {benchmark.fingerprint()} (seed {args.seed}, {args.controller} spawn pose, {args.n_per_set} scenarios "
      f"per set, poly paths of Bezier order {args.spatial_order})")
arm_base = np.array(benchmark.cfg.trajectory.arm_base)
print(f"{'set':9s} {'duration [s]':>22s} {'max |p-arm base|_xy [m]':>24s} {'peak speed [m/s]':>18s}")
for name in SET_NAMES:
    s = benchmark.sets[name]
    pos = s.poses[..., :3]
    speed = np.linalg.norm(np.diff(pos, axis=1), axis=-1).max(1) / REF_DT
    reach = np.linalg.norm(pos[..., :2] - arm_base[:2], axis=-1).max(1)
    print(f"{name:9s} {s.duration.min():6.2f} / {s.duration.mean():5.2f} / {s.duration.max():5.2f} "
          f"{reach.mean():10.2f} / {reach.max():5.2f} {speed.mean():10.2f} / {speed.max():5.2f}")
rigid = benchmark.sets["rigid"].info["path_type"]
print("rigid paths: " + ", ".join(f"{(rigid == k).sum()} {label}" for k, label in enumerate(RIGID_TYPES)))

if args.save:
    benchmark.save(args.save)
    print(f"saved {args.save}")
if args.plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(SET_NAMES), figsize=(4 * len(SET_NAMES), 8))
    theta = np.linspace(0, 2 * np.pi, 200)
    for col, name in enumerate(SET_NAMES):
        pos = benchmark.sets[name].poses[..., :3]
        for p in pos[:args.plot_count]:
            axes[0, col].plot(p[:, 0], p[:, 1], lw=0.7)
            axes[1, col].plot(np.linalg.norm(p[:, :2] - arm_base[:2], axis=-1), p[:, 2], lw=0.7)
        for radius, style in ((benchmark.cfg.trajectory.outer_radius, "k--"), (benchmark.cfg.far_max_radius_xy, "k:")):
            axes[0, col].plot(arm_base[0] + radius * np.cos(theta), arm_base[1] + radius * np.sin(theta), style, lw=0.8)
        axes[0, col].set(title=f"{name} (first {min(args.plot_count, len(pos))} of {len(pos)})", xlabel="x [m]", ylabel="y [m]", aspect="equal")
        axes[1, col].set(xlabel="x-y distance from the arm base [m]", ylabel="z [m]")
    fig.suptitle("dashed: 0.8 m training reach around the arm base, dotted: 2 m far-B limit")
    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=110)
    print(f"saved {args.plot}")
