#!/usr/bin/env bash
# Verify hybrid-rl-ik-control is importable and core deps are compatible
# inside the external Isaac Lab Python environment.
#
# Usage:
#   export ISAACLAB_PYTHON=/path/to/IsaacLab/.venv/bin/python
#   bash scripts/isaac/validate_isaaclab_env.sh
set -euo pipefail

PYTHON="${ISAACLAB_PYTHON:?Set ISAACLAB_PYTHON to the Isaac Lab interpreter path}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "==> Isaac Lab interpreter: $PYTHON"
echo "==> Python version: $("$PYTHON" --version)"
echo

echo "==> Checking imports and key dependency versions..."
"$PYTHON" - <<'EOF'
import sys

failures = []

def check(label, fn):
    try:
        fn()
        print(f"  OK  {label}")
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        failures.append(label)

def imp(mod):
    __import__(mod)

check("hybrid_control",                      lambda: imp("hybrid_control"))
check("hybrid_control.controllers.ik_qp",   lambda: imp("hybrid_control.controllers.ik_qp.ik_qp"))
check("hybrid_control.controllers.ik_qp (mujoco)", lambda: imp("hybrid_control.controllers.ik_qp.ik_qp_mujoco"))
check("hybrid_control.controllers.impedance", lambda: imp("hybrid_control.controllers.impedance.impedance_control"))
check("hybrid_control.sim_interfaces.mujoco", lambda: imp("hybrid_control.sim_interfaces.mujoco.adapter"))
check("pink (Configuration, build_ik)",      lambda: __import__("pink", fromlist=["Configuration", "build_ik"]))
check("pink.tasks.FrameTask",                lambda: __import__("pink.tasks", fromlist=["FrameTask"]))
check("qpsolvers",                           lambda: imp("qpsolvers"))
check("osqp",                                lambda: imp("osqp"))
check("daqp",                                lambda: imp("daqp"))
check("mujoco",                              lambda: imp("mujoco"))
check("numpy",                               lambda: imp("numpy"))
check("scipy",                               lambda: imp("scipy"))

import importlib.metadata as m
for pkg in ("numpy", "scipy", "osqp", "qpsolvers", "mujoco", "daqp"):
    try:
        ver = m.version(pkg)
        print(f"       {pkg}=={ver}")
    except m.PackageNotFoundError:
        pass

if failures:
    print(f"\nFAILED: {failures}")
    sys.exit(1)
else:
    print("\nAll checks passed.")
EOF

echo
echo "==> Running MuJoCo tests (env -u PYTHONPATH)..."
if "$PYTHON" -c "import pytest" 2>/dev/null; then
    env -u PYTHONPATH "$PYTHON" -m pytest "$REPO_ROOT/tests/" -q -x \
        --ignore="$REPO_ROOT/tests/test_isaac_lab.py" 2>/dev/null || \
    env -u PYTHONPATH "$PYTHON" -m pytest "$REPO_ROOT/tests/" -q -x
else
    echo "  pytest not found in Isaac Lab env — skipping."
    echo "  To run tests: uv pip install --python \"\$ISAACLAB_PYTHON\" pytest && re-run this script."
fi

echo
echo "==> Validation complete."
