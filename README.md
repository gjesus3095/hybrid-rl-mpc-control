# Hybrid RL-IK Control

A modular hybrid control framework combining **Reinforcement Learning (RL)** for base control with **optimization-based Inverse Kinematics (IK/QP)** for robotic manipulation.

The initial setup consists of a **3-DoF planar base (X, Y, yaw)** controlled exclusively by RL and a **7-DoF Franka Panda arm** controlled by IK/QP. The controllers operate independently but remain coupled through the robot's kinematics and dynamics: the arm compensates for base motion to maintain end-effector tracking, while the RL policy accounts for changes in the system's center of mass (CoM) caused by arm movements.

The long-term goal is to extend this architecture to **humanoid whole-body control**, combining RL-based balancing and locomotion with precise model-based manipulation.

## Setup

### uv-managed environment (Python 3.10 / 3.11)

```bash
uv sync
uv run pytest -q
```

The `isaac` extra installs Pink IK/QP dependencies (`uv sync --extra isaac`). If a sourced ROS environment injects incompatible pytest plugins via `PYTHONPATH`, prefix with `env -u PYTHONPATH`.

### Isaac Lab 3.0 / Isaac Sim 6.1 (Python 3.12)

This workflow installs the project into an existing Isaac Lab virtual environment in editable mode. Isaac Lab already provides all required dependencies, so the install adds only the editable package entry.

**Requirements:** `uv` on `PATH`, Isaac Lab installed and its Python interpreter known.

```bash
export ISAACLAB_PYTHON=~/workspace/IsaacLab/.venv/bin/python   # adjust to your path
bash scripts/isaac/setup_isaaclab_env.sh
bash scripts/isaac/validate_isaaclab_env.sh
```

`setup_isaaclab_env.sh` snapshots the environment before and after, and passes the before-snapshot as a constraints file so no pre-existing package is upgraded. To uninstall:

```bash
bash scripts/isaac/cleanup_isaaclab_env.sh
```

Packages upgraded during setup are reported but not automatically restored; the before-snapshot (`logs/isaaclab_snapshot_before.txt`) records their original versions.

## Running scripts

All MuJoCo scripts use the uv-managed environment:

```bash
uv run python scripts/mujoco/qp_controller/track_targets.py
```

Isaac Lab scripts use the Isaac Lab interpreter:

```bash
$ISAACLAB_PYTHON scripts/isaac/base_disturbance_keyboard.py
```

See [scripts/README.md](scripts/README.md) for all scripts, their configuration constants, and key-binding tables.

## Controllers

### IK/QP (differential, position-level)

`src/hybrid_control/controllers/ik_qp/` — solves a velocity-level QP at each control step and integrates to a joint-position command. See the [IK/QP README](src/hybrid_control/controllers/ik_qp/README.md) for the formulation, solver comparison (OSQP vs DAQP), and open items.

### Impedance (inverse-dynamics, torque-level)

`src/hybrid_control/controllers/impedance/` — solves an inverse-dynamics QP over `z = [ddq, tau]` each step with optional null-space posture torque. See the [impedance controller README](src/hybrid_control/controllers/impedance/README.md) for the full QP formulation and parameter reference.

## Interfaces

The controller computes commands through a simulator-independent `RobotModel` interface. `MujocoRobotModel` wraps an existing simulation using explicit joint-to-actuator mappings. See [docs/model_controller_interfaces.md](docs/model_controller_interfaces.md) for conventions and usage.

## Project Structure

Implemented files are unmarked; planned files are marked `[planned]`.

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
│       │   ├── ik_qp/
│       │   │   ├── controller.py
│       │   │   └── tasks.py               [planned]
│       │   └── impedance/
│       │       └── impedance_control.py
│       │
│       ├── robots/
│       │   ├── model.py
│       │   └── state.py
│       │
│       ├── sim_interfaces/
│       │   ├── mujoco/
│       │   │   ├── adapter.py
│       │   │   ├── modifier.py
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
│           ├── trajectories.py
│           └── transforms.py             [planned]
│
├── scripts/
│   ├── README.md                          all script descriptions and configuration
│   ├── mujoco/
│   │   ├── qp_controller/
│   │   │   ├── track_targets.py
│   │   │   ├── base_disturbance.py
│   │   │   └── base_disturbance_keyboard.py
│   │   └── impedance_controller/
│   │       ├── impedance_tracking.py
│   │       ├── impedance_track_targets.py
│   │       ├── impedance_keyboard.py
│   │       └── impedance_singularity.py
│   ├── isaac/
│   │   ├── base_disturbance_keyboard.py
│   │   ├── base_disturbance_impedance.py
│   │   ├── setup_isaaclab_env.sh
│   │   ├── validate_isaaclab_env.sh
│   │   └── cleanup_isaaclab_env.sh
│   └── rl/                               [planned]
│       ├── train.py
│       └── evaluate.py
│
├── tests/
│   ├── test_model_controller.py
│   ├── test_impedance_control.py
│   ├── test_kinematics.py                [planned]
│   └── test_hybrid.py                   [planned]
│
└── docs/
    ├── model_controller_interfaces.md
    ├── architecture.md                    [planned]
    └── experiments.md                    [planned]
```
