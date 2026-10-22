"""Plot sampled reference trajectories (world frame, env origin = where the base spawns) with the arm's reach limits.

Top view (x-y) and side view (x-z): shoulder, reach sphere/min radius, minimum height, the base footprint at spawn,
every trajectory (fixed points as squares, the start of each sine marked) and the quintic spawn -> A -> B paths
(dashed, A and B marked) with their arm-base shell and height limit -- all of them valid by construction (see
``hybrid_control.rl.policy_with_controller.trajectories``).

Usage (needs torch + matplotlib; works in the Isaac Lab environment):

    PYTHONPATH=src python scripts/plot_reference_trajectories.py [--n 40] [--difficulty 1.0] [--out out/reference_trajectories.png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

from hybrid_control.rl.policy_with_controller.trajectories import QUINTIC, SINE, ReferenceTrajectories, TrajectoryCfg

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=40)
parser.add_argument("--difficulty", type=float, default=1.0)
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--spatial_order", type=int, default=1, help="Bezier order of the quintic paths (1 = straight).")
parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "out" / "reference_trajectories.png"))
args = parser.parse_args()

torch.manual_seed(args.seed)
cfg = TrajectoryCfg(difficulty=args.difficulty, spatial_order=args.spatial_order)
trajs = ReferenceTrajectories(args.n, "cpu", cfg)
start = torch.tensor([0.61, 0.0, 0.635]).expand(args.n, 3)  # panda_hand at spawn in Isaac (folded arm)
trajs.reset(torch.arange(args.n), start)
times = torch.linspace(cfg.blend_time, args.seconds, 400)  # after the blend-in
path = torch.stack([trajs.position(torch.full((args.n,), float(t))) for t in times], dim=1).numpy()  # (n, T, 3)
q_times = torch.linspace(0.0, sum(cfg.segment_time), 400)  # quintic: from the spawn pose
q_path = torch.stack([trajs.position(torch.full((args.n,), float(t))) for t in q_times], dim=1).numpy()
kind = trajs.kind.numpy()
moving = kind == SINE
quintic = kind == QUINTIC
shoulder, arm_base = np.array(cfg.shoulder), np.array(cfg.arm_base)
waypoints = trajs.waypoints.numpy()
control = trajs.control_points.numpy()  # (n, 2, order + 1, 3)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
colors = plt.cm.tab20(np.linspace(0, 1, 20))
for ax, (u, v, xl, yl) in zip(axes, [(0, 1, "x [m]", "y [m]"), (0, 2, "x [m]", "z [m]")]):
    for i in range(args.n):
        c = colors[i % 20]
        if quintic[i]:
            ax.plot(q_path[i, :, u], q_path[i, :, v], color=c, lw=1.4, ls="--", alpha=0.9)
            if args.spatial_order > 1:  # control polygons
                for seg in control[i]:
                    ax.plot(seg[:, u], seg[:, v], ":", color=c, lw=0.8, alpha=0.6, marker=".", ms=3)
            ax.plot(waypoints[i, 1, u], waypoints[i, 1, v], "D", color=c, ms=4)
            ax.plot(waypoints[i, 2, u], waypoints[i, 2, v], "*", color=c, ms=8)
        elif moving[i]:
            ax.plot(path[i, :, u], path[i, :, v], color=c, lw=1.4, alpha=0.9)
            ax.plot(path[i, 0, u], path[i, 0, v], "o", color=c, ms=4)
        else:
            ax.plot(path[i, 0, u], path[i, 0, v], "s", color=c, ms=6)
    for r, style in ((cfg.reach, "--"), (cfg.min_radius, ":")):
        ax.add_patch(Circle((shoulder[u], shoulder[v]), r, fill=False, ls=style, ec="k", lw=1.2))
    ax.plot(shoulder[u], shoulder[v], "k^", ms=9, label="shoulder")
    ax.add_patch(Circle((arm_base[u], arm_base[v]), cfg.outer_radius, fill=False, ls="-", ec="tab:blue", lw=1.0))
    if v == 1:  # top view: the exclusion cylinder is a circle
        ax.add_patch(Circle((arm_base[0], arm_base[1]), cfg.inner_radius_xy, fill=False, ls="-", ec="tab:blue", lw=1.0))
    else:  # side view: its two walls
        for x in (arm_base[0] - cfg.inner_radius_xy, arm_base[0] + cfg.inner_radius_xy):
            ax.axvline(x, color="tab:blue", lw=1.0, ls=":")
    ax.plot(arm_base[u], arm_base[v], "bv", ms=8, label="arm base (quintic shell)")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal"); ax.grid(alpha=0.3)
axes[0].add_patch(Rectangle((0.3 - 0.2, -0.2), 0.4, 0.4, fc="0.6", ec="k", alpha=0.8, label="platform (0.4 x 0.4 m example)"))
axes[0].set_title(f"Top view: {args.n} reference trajectories (squares = fixed points, dashed = quintic, D = A, * = B)")
axes[1].add_patch(Rectangle((0.3 - 0.2, 0.0), 0.4, 0.28, fc="0.6", ec="k", alpha=0.8))
axes[1].axhline(cfg.min_z, color="r", ls="-.", lw=1, label=f"min height {cfg.min_z} m")
axes[1].axhline(cfg.quintic_min_z, color="tab:blue", ls="-.", lw=1, label=f"quintic min height {cfg.quintic_min_z:g} m")
axes[1].axhline(0.0, color="k", lw=1)
axes[1].set_title("Side view; dashed = max reach, dotted = min radius from the shoulder")
axes[0].legend(loc="upper right"); axes[1].legend(loc="upper right")
axes[1].set_ylim(-0.05, 1.5)
fig.suptitle(f"Reference trajectories in world coordinates (base spawns at the origin, difficulty {args.difficulty}, quintic path order {args.spatial_order}; dotted = control polygons)")
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=130)
d = np.linalg.norm(path[~quintic] - shoulder, axis=-1)
print(f"[INFO] saved {args.out}: {int(moving.sum())} sine / {int(quintic.sum())} quintic / "
      f"{int((~moving & ~quintic).sum())} fixed; distance from shoulder {d.min():.2f}..{d.max():.2f} m "
      f"(limits {cfg.min_radius}..{cfg.reach}), min z {path[~quintic][..., 2].min():.2f} m")
if quintic.any():
    dq = np.linalg.norm(q_path[quintic] - arm_base, axis=-1)
    dxy = np.linalg.norm(q_path[quintic][..., :2] - arm_base[:2], axis=-1)
    print(f"[INFO] quintic: x-y distance from the arm axis >= {dxy.min():.2f} m (limit {cfg.inner_radius_xy}), "
          f"distance from arm base <= {dq.max():.2f} m (limit {cfg.outer_radius}), min z {q_path[quintic][..., 2].min():.2f} m (limit {cfg.quintic_min_z:g})")
