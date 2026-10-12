# Hybrid RL-IK Control

A modular hybrid control framework combining **Reinforcement Learning (RL)** for base control with **optimization-based Inverse Kinematics (IK/QP)** for robotic manipulation.

The initial setup consists of a **3-DoF planar base (X, Y, yaw)** controlled exclusively by RL and a **7-DoF Franka Panda arm** controlled by IK/QP. The controllers operate independently but remain coupled through the robot's kinematics and dynamics: the arm compensates for base motion to maintain end-effector tracking, while the RL policy accounts for changes in the system's center of mass (CoM) caused by arm movements.

The long-term goal is to extend this architecture to **humanoid whole-body control**, combining RL-based balancing and locomotion with precise model-based manipulation.

## Imported Panda Model and Controller

The Panda model and controller were imported from `../hybrid_push_control_panda`:

- [Panda model](src/hybrid_control/sim_interfaces/mujoco/models/panda/panda.xml) and [simulation scene](src/hybrid_control/sim_interfaces/mujoco/models/panda/scene.xml), including robot meshes and model variants. The scene includes a ground plane at world Z = 0 with a repeating grid texture. Tables, loose objects, and furniture textures have been removed.
- [IK/QP controller](src/hybrid_control/controllers/ik_qp/ik_qp_mujoco.py), including its admittance and low-pass filters.
- [Math helpers](src/hybrid_control/utils/math_helpers.py) required by the controller.

The controller computes commands through a simulator-independent `RobotModel` interface. `MujocoRobotModel` wraps an existing simulation using explicit joint-to-actuator mappings, and the generic `Controller` base supports future controllers. See [model and controller interfaces](docs/model_controller_interfaces.md) for conventions and usage.

## Setup

Use Python 3.10 or 3.11. Test dependencies (`pytest`) are in a dependency group and are included by default with `uv sync`. The Pink IK/QP controller (`controllers/ik_qp/ik_qp.py`) needs the `isaac` extra: `uv sync --extra isaac`. It uses a temporary copy of Isaac Lab's Franka URDF until the Isaac Lab 3 (Python 3.12) requirement is resolved. Solvers `osqp` and `daqp` are supported; see [IK/QP with Pink](docs/ik_qp_pink.md) for the comparison and open items.

```bash
uv sync
uv run pytest -q
```

Alternatively, install with `python -m pip install -e '.[test]'` and run `pytest -q`.

If a sourced ROS environment injects incompatible pytest plugins through `PYTHONPATH`, run `env -u PYTHONPATH uv run pytest -q`.

## IK/QP Tracking Experiment

`scripts/mujoco/track_targets.py` runs the Panda arm through seeded Cartesian pose targets and reports tracking errors.

```bash
uv run python scripts/mujoco/track_targets.py
```

Edit the **Configuration & Constants** block at the top of the script; there are no command-line options:

- `SHOW_VIEWER`: set `True` for the interactive view; red sphere marks the position target, green sphere tracks the robot's world CoM. `COM_MARKER_DIAMETER` sets its diameter (default 5 cm).
- `Q_INIT`: seven initial arm joint angles in radians; actuator commands start at the same pose.
- `TRACKING_SECONDS`: simulated seconds allowed per reference before checking accuracy, rounded up to a whole control interval.
- `TARGET_RADIUS` and `RANDOM_SEED`: reproducible XYZ offsets around the starting hand position, defaulting to ±10 cm per axis.
- `N_TARGETS`: total references, including the reference orientation at the start and end (default 5).
- `ROTATION_ORDER`, `ORIENTATION_CENTER_DEG`, and `ORIENTATION_RANGE_DEG`: random Euler-angle targets around a reference orientation, with independent uniform offsets of ±40°, ±30°, and ±40°. The default `zyx` order means extrinsic rotations about fixed world Z, Y, and X axes. `RANDOM_SEED` reproduces both positions and orientations; the first and last orientations equal `ORIENTATION_CENTER_DEG`.
- `DT_CTRL`, gains, weights, velocity limits, and acceptance thresholds: controller and assertion settings.

Each reference runs for five simulated seconds by default, with control at 50 Hz and MuJoCo at 500 Hz. Assertions require position error below 5 mm, orientation error below 0.03 rad, and joint commands/states within limits, allowing solver and actuator tolerances. Failures report the target, seed, elapsed simulation time, and measured values versus limits. Joint bias-force compensation is enabled. Large offsets or different initial poses may produce unreachable references; the same tracking assertions apply.

## Base Disturbance Experiment

`scripts/mujoco/base_disturbance.py` tests whether the IK/QP controller can maintain a fixed world-frame end-effector pose while the robot base moves in X/Y/Yaw.

`MujocoModifier.build_mobile_base_spec` adds three DOFs to `link0` via `MjSpec`: two prismatic joints (X, Y) and one hinge joint (Yaw around Z). All three are outside the IK — they act as a disturbance. The IK/QP compensates purely through arm motion.

```bash
uv run python scripts/mujoco/base_disturbance.py
```

Configuration at the top of the script:

- `BASE_MOTION_RADIUS`: max XY command per axis in metres (default 0.2 m; scene limits ±2 m).
- `BASE_YAW_RADIUS`: max yaw command in radians (default π/6).
- `N_BASE_POSITIONS`: number of random waypoints.
- `RANDOM_SEED`: reproducible waypoints.
- `SHOW_VIEWER`: set `True` for the interactive view.

Note: base displacements that push the world-frame target outside the arm workspace report FAIL with a large position error.

## Interactive Base Disturbance

`scripts/mujoco/base_disturbance_keyboard.py` runs the same experiment interactively with keyboard control.

```bash
uv run python scripts/mujoco/base_disturbance_keyboard.py
```

| Key | Action |
|-----|--------|
| `← / →` | base X (left / right) |
| `↑ / ↓` | base Y (forward / back) |
| `Space` | base Yaw CCW (+Z) |
| `AltGr` | base Yaw CW (−Z) |

The EE target is fixed at the initial grasp world-frame pose. The arm must compensate for all base motion purely through joint motion.

## Project Structure

Implemented files are unmarked; planned files are marked `[planned]`. Empty directories exist locally but are not tracked by Git.

```text
hybrid-rl-ik-control/
│
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/                                [planned]
│   ├── robot.yaml
│   ├── ik_qp.yaml
│   └── rl.yaml
│
├── src/
│   └── hybrid_control/
│       ├── controllers/
│       │   ├── base.py
│       │   └── ik_qp/
│       │       ├── controller.py
│       │       └── tasks.py               [planned]
│       │
│       ├── robots/
│       │   ├── model.py
│       │   └── state.py
│       │
│       ├── sim_interfaces/
│       │   ├── mujoco/
│       │   │   ├── adapter.py
│       │   │   └── models/panda/          (panda.xml, scene.xml, hand.xml, assets/)
│       │   └── isaac/                     [planned]
│       │       ├── adapter.py
│       │       └── assets/
│       │
│       ├── controllers/rl/                [planned]
│       │   ├── policy.py
│       │   └── observations.py
│       │
│       ├── environments/isaac_lab/        [planned]
│       │   ├── env_cfg.py
│       │   ├── robot_cfg.py
│       │   ├── observations.py
│       │   ├── rewards.py
│       │   ├── actions.py
│       │   └── terminations.py
│       │
│       ├── integration/                   [planned]
│       │   ├── hybrid_controller.py
│       │   └── coordinator.py
│       │
│       └── utils/
│           ├── math_helpers.py
│           ├── trajectories.py            [planned]
│           └── transforms.py             [planned]
│
├── scripts/
│   ├── mujoco/
│   │   ├── track_targets.py
│   │   ├── base_disturbance.py
│   │   ├── base_disturbance_keyboard.py
│   │   └── test_trajectory.py            [planned]
│   ├── isaac/                            [planned]
│   │   ├── test_ik.py
│   │   └── compare_jacobians.py
│   └── rl/                               [planned]
│       ├── train.py
│       └── evaluate.py
│
├── tests/
│   ├── test_model_controller.py
│   ├── test_kinematics.py                [planned]
│   └── test_hybrid.py                   [planned]
│
└── docs/
    ├── model_controller_interfaces.md
    ├── architecture.md                    [planned]
    └── experiments.md                    [planned]
```
