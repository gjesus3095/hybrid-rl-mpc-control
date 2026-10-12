# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (Python 3.10 or 3.11 required; test group included by default)
uv sync

# Run all tests
uv run pytest -q

# Run IK/QP tracking experiment (edit SHOW_VIEWER=True for viewer)
uv run python scripts/mujoco/track_targets.py

# If ROS PYTHONPATH interferes
env -u PYTHONPATH uv run pytest -q
```

## Architecture

Hybrid controller for a 3-DoF planar mobile base (RL) + 7-DoF Franka Panda arm (IK/QP). Controllers are independent but kinematically coupled: arm compensates for base motion; RL policy accounts for arm's CoM shift.

**Implemented:**
- `src/hybrid_control/robots/model.py` — `RobotModel` ABC: `joint_names`, `position_limits`, `get_state()`, `get_com_state()`, `get_pose(frame)`, `get_jacobian(frame)`, `apply_command(cmd)`
- `src/hybrid_control/robots/state.py` — `JointState`, `CenterOfMassState`, `Pose` (xyzw quat), `JointPositionCommand`; all validate on construction
- `src/hybrid_control/sim_interfaces/mujoco/adapter.py` — `MujocoRobotModel`: wraps `MjModel`/`MjData` with `joint_actuator_pairs`; exposes `from_xml_path` + `initialize`
- `src/hybrid_control/controllers/base.py` — `Controller[TargetT, CommandT]` ABC: `compute(target, dt)`, `reset()`
- `src/hybrid_control/controllers/ik_qp/config.py` — `PoseTarget`, `IKQPConfig` shared by both IK/QP controllers
- `src/hybrid_control/controllers/ik_qp/ik_qp_mujoco.py` — `InverseKinematicsController(robot, IKQPConfig)`; QP (OSQP) on the robot's Jacobian, integrates to position command
- `src/hybrid_control/controllers/ik_qp/ik_qp.py` — `PinkInverseKinematicsController(robot, IKQPConfig)`; same QP solved with Pink on the Isaac Lab Franka URDF (Isaac Lab joint/frame names, `uv sync --extra isaac`)
- `src/hybrid_control/sim_interfaces/isaac/assets/franka/lula_franka_gen.urdf` — temporary copy of Isaac Lab's Franka URDF (see `ik_qp.py` docstring)
- `src/hybrid_control/utils/math_helpers.py` — math utilities for kinematics
- `src/hybrid_control/sim_interfaces/mujoco/models/panda/` — Panda XML model + meshes

**Planned (not yet implemented):** RL policy/observations, Isaac adapter, Isaac Lab env configs, `integration/hybrid_controller.py`, trajectory utilities.

## Key Conventions

- Units: radians for hinge joints, metres for slides; velocities in those units/second; simulation time in seconds
- Quaternions: **xyzw** order at all Python interfaces (adapter converts from MuJoCo's wxyz)
- Jacobian: `(6, n)` — world linear velocity rows first, then angular; columns follow `joint_names` order
- `RobotModel` reads never call `mj_forward`/`mj_step`; caller must refresh kinematics after editing state
- `apply_command` writes `data.ctrl` only — does not clamp, change `qpos`, or step physics
- CoM subtree: auto-detected as common ancestor of selected joints; override with `root_body=` if needed

## Test Configuration

Edit constants at top of `tests/test_ik_qp.py` to configure the IK/QP experiment:
- `SHOW_VIEWER`, `Q_INIT`, `N_TARGETS`, `TRACKING_SECONDS`, `RANDOM_SEED`
- `TARGET_RADIUS`, `ROTATION_ORDER`, `ORIENTATION_CENTER_DEG`, `ORIENTATION_RANGE_DEG`
- `DT_CTRL`, gains, weights, velocity limits, acceptance thresholds

See `docs/model_controller_interfaces.md` for the full interface contract and usage patterns.
