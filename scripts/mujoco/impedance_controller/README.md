# MuJoCo — Impedance Controller Scripts

Run from the project root with the uv-managed environment (Python 3.10 / 3.11):

```bash
uv run python scripts/mujoco/impedance_controller/<script>.py
```

See the [impedance controller README](../../../src/hybrid_control/controllers/impedance/README.md) for the QP formulation and parameter reference.

---

## `impedance_tracking.py`

Tracks a quintic or sinusoidal Cartesian trajectory with the inverse-dynamics QP, asserts tracking error, and optionally saves a plot.

| Constant | Default | Description |
|---|---|---|
| `TRAJECTORY` | `"quintic"` | `"quintic"` (rest-to-rest waypoints) or `"sine"` |
| `KP_POS`, `KP_ORI` | — | Task-space stiffness |
| `KD_POS`, `KD_ORI` | `2√Kp` | Damping (critical by default) |
| `TAU_MAX`, `DDQ_MAX` | Franka limits | Torque and acceleration bounds |
| `DT_CTRL` | `0.002 s` | Controller period (500 Hz, every physics step) |
| `USE_NULLSPACE` | `True` | Null-space posture torque after QP |
| `SAVE_PLOT` | `None` | PNG path for reference vs. actual |

Each segment must end within 5 mm / 0.03 rad and stay within 20 mm while moving.

---

## `impedance_track_targets.py`

Runs the impedance controller through seeded Cartesian pose targets (same target generation as the IK/QP `track_targets.py`).

---

## `impedance_keyboard.py`

Interactive impedance control with keyboard-driven Cartesian EE commands.

---

## `impedance_singularity.py`

Drives the arm near kinematic singularities to evaluate manipulability regularisation in the QP.
