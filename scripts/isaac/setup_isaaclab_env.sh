#!/usr/bin/env bash
# Install hybrid-rl-ik-control (editable) into an external Isaac Lab Python environment.
#
# Usage:
#   export ISAACLAB_PYTHON=/path/to/IsaacLab/.venv/bin/python
#   bash scripts/isaac/setup_isaaclab_env.sh
#
# The script snapshots installed packages before and after the install and
# reports what was added or upgraded. Snapshot files are written to logs/.
set -euo pipefail

PYTHON="${ISAACLAB_PYTHON:?Set ISAACLAB_PYTHON to the Isaac Lab interpreter path}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"
BEFORE="$LOGS_DIR/isaaclab_snapshot_before.txt"
AFTER="$LOGS_DIR/isaaclab_snapshot_after.txt"

mkdir -p "$LOGS_DIR"

echo "==> Isaac Lab interpreter: $PYTHON"
echo "==> Python version: $("$PYTHON" --version)"
echo

echo "==> Snapshotting environment (before)..."
"$PYTHON" -c "
import importlib.metadata as m
for d in sorted(m.distributions(), key=lambda d: d.metadata['Name'].lower()):
    print(d.metadata['Name'] + '==' + d.version)
" > "$BEFORE"
echo "    $BEFORE ($(wc -l < "$BEFORE") packages)"
echo

echo "==> Installing hybrid-rl-ik-control in editable mode..."
# Install without extras: Isaac Lab already provides pink, daqp, qpsolvers, and
# all other controller deps. --constraint locks pre-existing packages so uv
# cannot upgrade them while resolving our core deps.
uv pip install --python "$PYTHON" --constraint "$BEFORE" -e "$REPO_ROOT"
echo

echo "==> Installing video recording deps..."
uv pip install --python "$PYTHON" "moviepy>=1.0.3,<2.0.0.dev0"
echo

echo "==> Snapshotting environment (after)..."
"$PYTHON" -c "
import importlib.metadata as m
for d in sorted(m.distributions(), key=lambda d: d.metadata['Name'].lower()):
    print(d.metadata['Name'] + '==' + d.version)
" > "$AFTER"
echo "    $AFTER ($(wc -l < "$AFTER") packages)"
echo

echo "==> Diff (added / upgraded):"
python3 - "$BEFORE" "$AFTER" <<'EOF'
import sys

def load(path):
    pkgs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if '==' in line:
                name, ver = line.split('==', 1)
                pkgs[name.lower()] = (name, ver)
    return pkgs

before = load(sys.argv[1])
after  = load(sys.argv[2])

added    = {k: after[k] for k in after if k not in before}
upgraded = {k: (before[k][1], after[k][1]) for k in after
            if k in before and before[k][1] != after[k][1]}

if added:
    print("  Added:")
    for k, (name, ver) in sorted(added.items()):
        print(f"    {name}=={ver}")
else:
    print("  No packages added.")

if upgraded:
    print("  Upgraded:")
    for k, (old, new) in sorted(upgraded.items()):
        print(f"    {before[k][0]}: {old} -> {new}")
else:
    print("  No packages upgraded.")
EOF

echo
echo "==> Done. Run scripts/isaac/validate_isaaclab_env.sh to verify."
echo "    To undo: bash scripts/isaac/cleanup_isaaclab_env.sh"
