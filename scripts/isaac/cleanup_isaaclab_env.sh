#!/usr/bin/env bash
# Remove hybrid-rl-ik-control and any packages added by setup_isaaclab_env.sh
# from the external Isaac Lab Python environment.
#
# Usage:
#   export ISAACLAB_PYTHON=/path/to/IsaacLab/.venv/bin/python
#   bash scripts/isaac/cleanup_isaaclab_env.sh
#
# Packages that existed before setup (in logs/isaaclab_snapshot_before.txt) are
# never removed or downgraded. Upgraded packages are reported but left as-is;
# restore them manually if needed:
#   uv pip install --python "$ISAACLAB_PYTHON" <pkg>==<old_version>
set -euo pipefail

PYTHON="${ISAACLAB_PYTHON:?Set ISAACLAB_PYTHON to the Isaac Lab interpreter path}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"
BEFORE="$LOGS_DIR/isaaclab_snapshot_before.txt"
AFTER="$LOGS_DIR/isaaclab_snapshot_after.txt"

if [[ ! -f "$BEFORE" ]]; then
    echo "ERROR: $BEFORE not found. Run setup_isaaclab_env.sh first." >&2
    exit 1
fi

echo "==> Isaac Lab interpreter: $PYTHON"
echo

echo "==> Uninstalling hybrid-rl-ik-control..."
uv pip uninstall --python "$PYTHON" hybrid-rl-ik-control || true
echo

if [[ -f "$AFTER" ]]; then
    echo "==> Removing packages added by setup (not in before-snapshot)..."
    python3 - "$BEFORE" "$AFTER" "$PYTHON" <<'EOF'
import sys, subprocess

def load(path):
    pkgs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if '==' in line:
                name, ver = line.split('==', 1)
                pkgs[name.lower()] = (name, ver)
    return pkgs

before  = load(sys.argv[1])
after   = load(sys.argv[2])
python  = sys.argv[3]

added    = {k: after[k] for k in after if k not in before}
upgraded = {k: (before[k][1], after[k][1]) for k in after
            if k in before and before[k][1] != after[k][1]}

# Never remove our own package — already uninstalled above.
added.pop('hybrid-rl-ik-control', None)

if added:
    names = [after[k][0] for k in sorted(added)]
    print(f"  Removing: {', '.join(names)}")
    subprocess.run(
        ['uv', 'pip', 'uninstall', '--python', python] + names,
        check=True,
    )
else:
    print("  No added packages to remove.")

if upgraded:
    print()
    print("  WARNING: the following packages were upgraded and have NOT been restored.")
    print("  Restore manually with: uv pip install --python \"$ISAACLAB_PYTHON\" <pkg>==<old>")
    for k, (old, new) in sorted(upgraded.items()):
        print(f"    {before[k][0]}: {old}  (current: {new})")
EOF
else
    echo "  (no after-snapshot found; skipping added-package removal)"
fi

echo
echo "==> Done."
