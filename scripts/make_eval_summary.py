"""Write every metric of one or more ``evaluate_policy.py`` runs to a plain-text file. No Isaac needed, only numpy.

    python scripts/make_eval_summary.py out/eval/<run> out/eval/<run>_order3 --out out/eval_summary.txt

Each run folder holds a ``summary.json``; per run the file lists the run metadata, the compact table of
``metrics.format_table`` and then all statistics of every set (errors in cm / deg, margins in cm).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from hybrid_control.rl.evaluation.metrics import format_table

parser = argparse.ArgumentParser(description="Plain-text summary of evaluation runs.")
parser.add_argument("eval_dirs", nargs="+", type=str, help="Run folders with a summary.json.")
parser.add_argument("--out", type=str, required=True, help="Text file to write.")
args = parser.parse_args()

CM, DEG = 100.0, float(np.degrees(1.0))


def block(title: str, stats: dict, scale: float, unit: str) -> list[str]:
    flat = {k: v for k, v in stats.items() if not isinstance(v, dict)}
    lines = [f"    {title} [{unit}]: " + ", ".join(f"{k} {scale * v:.2f}" for k, v in flat.items())]
    for key, sub in stats.items():
        if isinstance(sub, dict):
            lines.append(f"      {key}: " + ", ".join(f"{k} {scale * v:.2f}" for k, v in sub.items()))
    return lines


def describe(name: str, s: dict) -> list[str]:
    lines = [f"  {name}: {s['scenarios']} scenarios, {s['failed']} failed ({100 * s['failure_rate']:.1f} %)"
             + (f", mean failure time {s['failure_time_s']:.2f} s" if s["failed"] else "")
             + (f"; tipped {s['tipped']}, COM left the footprint {s['left_footprint']}" if "tipped" in s else "")
             + (f"; IK failures {s['ik_failures']}" if "ik_failures" in s else "")]
    lines += block("position error", s["pos_err_m"], CM, "cm")
    lines += block("orientation error", s["ori_err_rad"], DEG, "deg")
    lines += block("final position error (last 0.5 s, completed scenarios)", s["final_pos_err_m"], CM, "cm")
    lines += block("COM margin to the footprint edge", s["com_margin_m"], CM, "cm")
    lines.append("    COM offset rho (fraction of the half footprint, > 1 = outside): "
                 + ", ".join(f"{k} {v:.3f}" for k, v in s["com_rho"].items()))
    b = s["base"]
    lines.append(f"    base: mean speed {b['speed_mean_mps']:.3f} m/s, max speed {b['speed_max_mps']:.3f} m/s, "
                 f"mean |yaw rate| {b['yaw_rate_abs_mean']:.3f} rad/s, max tilt {np.degrees(b['max_tilt_rad']):.2f} deg, "
                 f"max distance from spawn {b['max_distance_from_spawn_m']:.2f} m")
    return lines


out = []
for eval_dir in args.eval_dirs:
    data = json.loads((Path(eval_dir) / "summary.json").read_text())
    meta, sets = data["meta"], data["sets"]
    out += ["=" * 100, f"{meta['name']}  ({eval_dir})", "=" * 100]
    out += [f"{k}: {v}" for k, v in meta.items() if k not in ("name", "set_fingerprints")]
    out += ["set fingerprints: " + ", ".join(f"{k} {v}" for k, v in meta.get("set_fingerprints", {}).items()), ""]
    out += [format_table(sets), "", "All metrics (pooled over the valid steps of all scenarios unless noted):"]
    for name, s in sets.items():
        out += describe(name, s) + [""]

path = Path(args.out)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(out) + "\n")
print(f"wrote {path}")
