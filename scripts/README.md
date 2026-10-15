# Scripts

Scripts are grouped by simulator and, within Isaac Lab, by control mode.

## MuJoCo (uv-managed environment)

All MuJoCo scripts run with the project's uv-managed environment (Python 3.10 / 3.11):

```bash
uv run python scripts/mujoco/<subfolder>/<script>.py
```

| Folder | Contents |
|---|---|
| [`mujoco/qp_controller/`](mujoco/qp_controller/README.md) | IK/QP: pose tracking, scripted and keyboard base disturbance |
| [`mujoco/impedance_controller/`](mujoco/impedance_controller/README.md) | Impedance QP: trajectory tracking, keyboard control, singularity probes |

## Isaac Lab (Isaac Lab interpreter)

Isaac Lab scripts require `ISAACLAB_PYTHON` set and the package installed into the Isaac Lab environment. See [Isaac Lab setup](../README.md#isaac-lab-30--isaac-sim-61-python-312).

Run from the project root:

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/<subfolder>/<script>.py [options]
```

| Folder | Contents |
|---|---|
| [`isaac/no_controller/`](isaac/no_controller/README.md) | Moving-mass policy: train, evaluate, demos, and environment checks |
| [`isaac/with_controller/`](isaac/with_controller/README.md) | Ridgeback-Franka with IK/QP or impedance arm controller: train, evaluate, demos, and disturbance scripts |

Setup, validation, and cleanup shell scripts are directly under `scripts/isaac/`:

```bash
export ISAACLAB_PYTHON=~/workspace/IsaacLab/.venv/bin/python
bash scripts/isaac/setup_isaaclab_env.sh
bash scripts/isaac/validate_isaaclab_env.sh
bash scripts/isaac/cleanup_isaaclab_env.sh   # to uninstall
```
