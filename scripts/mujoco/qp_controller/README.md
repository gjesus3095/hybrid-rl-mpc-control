# MuJoCo — IK/QP Scripts

Run from the project root with the uv-managed environment (Python 3.10 / 3.11):

```bash
uv run python scripts/mujoco/qp_controller/<script>.py
```

Configuration lives in the **Configuration & Constants** block at the top of each script; there are no command-line options.

---

## `track_targets.py`

Runs the Panda arm through seeded Cartesian pose targets and asserts tracking error.

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

## `base_disturbance.py`

Tests that the IK/QP holds a fixed world-frame EE pose while the base moves through random X/Y/Yaw waypoints. `MujocoModifier.build_mobile_base_spec` adds three uncontrolled DOFs to `link0` via `MjSpec`; the IK compensates purely through arm motion.

| Constant | Default | Description |
|---|---|---|
| `BASE_MOTION_RADIUS` | `0.2 m` | Max XY command per axis (scene limits ±2 m) |
| `BASE_YAW_RADIUS` | `π/6 rad` | Max yaw command |
| `N_BASE_POSITIONS` | `5` | Number of random base waypoints |
| `RANDOM_SEED` | `7` | Reproducible waypoints |
| `SHOW_VIEWER` | `True` | Open the viewer |

Base displacements that push the world-frame target outside the arm workspace report FAIL with a large position error.

---

## `base_disturbance_keyboard.py`

Same base-disturbance experiment driven interactively with keyboard input.

| Key | Action |
|---|---|
| `← / →` | base X |
| `↑ / ↓` | base Y |
| `Space` | Yaw CCW (+Z) |
| `AltGr` | Yaw CW (−Z) |
