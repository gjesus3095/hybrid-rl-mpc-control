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
- `src/hybrid_control/controllers/ik_qp/controller.py` — `InverseKinematicsController(robot, IKQPConfig)` with `PoseTarget`; QP-based velocity solver integrates to position command
- `src/hybrid_control/utils/math_helpers.py` — math utilities for kinematics
- `src/hybrid_control/sim_interfaces/mujoco/models/panda/` — Panda XML model + meshes
- `src/hybrid_control/rl/` — RSL-RL training env (`BaseStabilizationEnv`), observations, rewards, trajectories (reachable from the spawn pose), QP worker pool, PPO config; Isaac `RobotModel` adapter in `sim_interfaces/isaac/ridgeback/arm_model.py`; entry points `scripts/isaac/train_base_policy.py`, `play_base_policy.py` (see `docs/code_overview.md` section 4)

**Planned (not yet implemented):** `integration/hybrid_controller.py` (trained policy + QP outside the training wrapper), automatic curriculum, observation noise.

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
