"""Write one readable Markdown report from an ``evaluate_policy.py`` run (with or without ``--payloads``). No Isaac
needed, only numpy.

    python scripts/make_eval_report.py out/eval/impedance_payloads            # -> out/eval/impedance_payloads/report.md
    python scripts/make_eval_report.py out/eval/zero_policy                   # a run where every scenario has its own payload
"""

import argparse
from pathlib import Path

from hybrid_control.rl.evaluation.report import build_report

parser = argparse.ArgumentParser(description="Markdown report of an evaluation run.")
parser.add_argument("eval_dir", type=str, help="Run folder: its summary.json and <set>.npz, or payload_<kg>kg subfolders.")
parser.add_argument("--out", type=str, default=None, help="Default: <eval_dir>/report.md.")
args = parser.parse_args()

path = Path(args.out or Path(args.eval_dir) / "report.md")
path.write_text(build_report(args.eval_dir))
print(f"wrote {path}")
