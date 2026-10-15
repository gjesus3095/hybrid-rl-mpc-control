"""Plot sampled reference trajectories (world frame, env origin = where the base spawns) with the arm's reach limits.

Top view (x-y) and side view (x-z): shoulder, reach sphere/min radius, minimum height, the base footprint at spawn,
every trajectory (fixed points as dots, the start of each moving trajectory marked) -- all of them reachable by
construction (see ``hybrid_control.rl.trajectories``).

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

from hybrid_control.rl.policy_with_controller.trajectories import ReferenceTrajectories, TrajectoryCfg

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=40)
parser.add_argument("--difficulty", type=float, default=1.0)
parser.add_argument("--seconds", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "out" / "reference_trajectories.png"))
args = parser.parse_args()

torch.manual_seed(args.seed)
cfg = TrajectoryCfg(difficulty=args.difficulty)
trajs = ReferenceTrajectories(args.n, "cpu", cfg)
start = torch.tensor([0.45, 0.0, 0.55]).expand(args.n, 3)  # roughly the folded-arm end-effector at spawn
trajs.reset(torch.arange(args.n), start)
times = torch.linspace(cfg.blend_time, args.seconds, 400)  # after the blend-in
path = torch.stack([trajs.position(torch.full((args.n,), float(t))) for t in times], dim=1).numpy()  # (n, T, 3)
moving = (trajs.amplitude.norm(dim=-1) > 1e-6).numpy()
shoulder = np.array(cfg.shoulder)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
colors = plt.cm.tab20(np.linspace(0, 1, 20))
for ax, (u, v, xl, yl) in zip(axes, [(0, 1, "x [m]", "y [m]"), (0, 2, "x [m]", "z [m]")]):
    for i in range(args.n):
        c = colors[i % 20]
        if moving[i]:
            ax.plot(path[i, :, u], path[i, :, v], color=c, lw=1.4, alpha=0.9)
            ax.plot(path[i, 0, u], path[i, 0, v], "o", color=c, ms=4)
        else:
            ax.plot(path[i, 0, u], path[i, 0, v], "s", color=c, ms=6)
    for r, style in ((cfg.reach, "--"), (cfg.min_radius, ":")):
        ax.add_patch(Circle((shoulder[u], shoulder[v]), r, fill=False, ls=style, ec="k", lw=1.2))
    ax.plot(shoulder[u], shoulder[v], "k^", ms=9, label="shoulder")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal"); ax.grid(alpha=0.3)
axes[0].add_patch(Rectangle((0.3 - 0.2, -0.2), 0.4, 0.4, fc="0.6", ec="k", alpha=0.8, label="platform (0.4 x 0.4 m example)"))
axes[0].set_title(f"Top view: {args.n} reference trajectories (squares = fixed points)")
axes[1].add_patch(Rectangle((0.3 - 0.2, 0.0), 0.4, 0.28, fc="0.6", ec="k", alpha=0.8))
axes[1].axhline(cfg.min_z, color="r", ls="-.", lw=1, label=f"min height {cfg.min_z} m")
axes[1].axhline(0.0, color="k", lw=1)
axes[1].set_title("Side view; dashed = max reach, dotted = min radius from the shoulder")
axes[0].legend(loc="upper right"); axes[1].legend(loc="upper right")
axes[1].set_ylim(-0.05, 1.5)
fig.suptitle(f"Reference trajectories in world coordinates (base spawns at the origin, difficulty {args.difficulty})")
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=130)
d = np.linalg.norm(path - shoulder, axis=-1)
print(f"[INFO] saved {args.out}: {int(moving.sum())} moving / {int((~moving).sum())} fixed; "
      f"distance from shoulder {d.min():.2f}..{d.max():.2f} m (limits {cfg.min_radius}..{cfg.reach}), min z {path[..., 2].min():.2f} m")
