# Scripts

All MuJoCo scripts run with the project's uv-managed environment (Python 3.10/3.11):

```bash
uv run python scripts/mujoco/<script>.py
```

Isaac Lab scripts require the Isaac Lab interpreter. Set `ISAACLAB_PYTHON` once per shell session (see [Isaac Lab setup](../README.md#isaac-lab-30--isaac-sim-61-python-312)):

```bash
$ISAACLAB_PYTHON scripts/isaac/<script>.py
```

---

## MuJoCo — IK/QP scripts (`scripts/mujoco/qp_controller/`)

Configuration lives in the **Configuration & Constants** block at the top of each script; there are no command-line options.

### `track_targets.py`

Runs the Panda arm through seeded Cartesian pose targets and asserts tracking error.

```bash
uv run python scripts/mujoco/qp_controller/track_targets.py
```

Key constants:

| Constant | Default | Description |
|---|---|---|
| `SHOW_VIEWER` | `True` | Open the interactive MuJoCo viewer |
| `N_TARGETS` | `10` | Number of pose references (first and last use `ORIENTATION_CENTER_DEG`) |
| `RANDOM_SEED` | `7` | Reproducible position and orientation targets |
| `TARGET_RADIUS` | `0.10 m` | ±XYZ offset around initial EE position |
| `TRACKING_SECONDS` | `5.0 s` | Simulated time per reference |
| `ROTATION_ORDER` | `'zyx'` | Extrinsic Euler order for orientation targets |
| `ORIENTATION_CENTER_DEG` | `[-90, 0, 180]` | Centre orientation in degrees |
| `ORIENTATION_RANGE_DEG` | `[40, 30, 40]` | Uniform random offset per axis |
| `DT_CTRL` | `0.02 s` | Controller period (50 Hz) |
| `KP_POS`, `KP_ORI` | `80`, `60` | CLIK proportional gains |
| `JOINT_VEL_WEIGHT` | `1e5` | QP regularisation (damping) |
| `POSITION_TOLERANCE` | `5 mm` | Pass/fail threshold |
| `ORIENTATION_TOLERANCE` | `0.03 rad` | Pass/fail threshold |

Assertions check position error, orientation error, and joint limits. A failed target prints the context, seed, elapsed time, and measured vs. allowed values. Unreachable references (outside workspace) produce FAIL with large position error.

---

### `base_disturbance.py`

Tests that the IK/QP holds a fixed world-frame EE pose while the base moves through random X/Y/Yaw waypoints. `MujocoModifier.build_mobile_base_spec` adds three uncontrolled DOFs to `link0` via `MjSpec`; the IK compensates purely through arm motion.

```bash
uv run python scripts/mujoco/qp_controller/base_disturbance.py
```

| Constant | Default | Description |
|---|---|---|
| `BASE_MOTION_RADIUS` | `0.2 m` | Max XY command per axis (scene limits ±2 m) |
| `BASE_YAW_RADIUS` | `π/6 rad` | Max yaw command |
| `N_BASE_POSITIONS` | `5` | Number of random base waypoints |
| `RANDOM_SEED` | `7` | Reproducible waypoints |
| `SHOW_VIEWER` | `True` | Open the viewer |

Base displacements that push the world-frame target outside the arm workspace report FAIL with a large position error.

---

### `base_disturbance_keyboard.py`

Same base-disturbance experiment driven interactively with keyboard input.

```bash
uv run python scripts/mujoco/qp_controller/base_disturbance_keyboard.py
```

| Key | Action |
|---|---|
| `← / →` | base X |
| `↑ / ↓` | base Y |
| `Space` | Yaw CCW (+Z) |
| `AltGr` | Yaw CW (−Z) |

---

## MuJoCo — Impedance scripts (`scripts/mujoco/impedance_controller/`)

See the [impedance controller README](../src/hybrid_control/controllers/impedance/README.md) for the QP formulation and parameter reference.

### `impedance_tracking.py`

Tracks a quintic or sinusoidal Cartesian trajectory with the inverse-dynamics QP, asserts tracking error, and optionally saves a plot.

```bash
uv run python scripts/mujoco/impedance_controller/impedance_tracking.py
```

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

### `impedance_track_targets.py`

Runs the impedance controller through seeded Cartesian pose targets (same target generation as the IK/QP `track_targets.py`).

```bash
uv run python scripts/mujoco/impedance_controller/impedance_track_targets.py
```

---

### `impedance_keyboard.py`

Interactive impedance control with keyboard-driven Cartesian EE commands.

```bash
uv run python scripts/mujoco/impedance_controller/impedance_keyboard.py
```

---

### `impedance_singularity.py`

Drives the arm near kinematic singularities to evaluate manipulability regularisation in the QP.

```bash
uv run python scripts/mujoco/impedance_controller/impedance_singularity.py
```

---

## Isaac Lab scripts (`scripts/isaac/`)

Require `ISAACLAB_PYTHON` set and the package installed into the Isaac Lab environment (see [Isaac Lab setup](../README.md#isaac-lab-30--isaac-sim-61-python-312)).

See [`docs/isaac_base_disturbance.md`](../docs/isaac_base_disturbance.md) for full launch commands and Isaac-specific options.

### `base_disturbance_keyboard.py`

Keyboard-driven Ridgeback base with Pink IK/QP holding the Franka TCP.

```bash
$ISAACLAB_PYTHON scripts/isaac/base_disturbance_keyboard.py
```

### `base_disturbance_impedance.py`

Same base-disturbance experiment with the torque-level impedance QP (default: `--control_hz 50`, `--physics_hz 200`).

```bash
$ISAACLAB_PYTHON scripts/isaac/base_disturbance_impedance.py
```
