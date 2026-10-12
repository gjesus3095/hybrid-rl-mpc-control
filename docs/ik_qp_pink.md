# IK/QP with Pink on Isaac Lab assets

## What changed

| Item | Change |
|---|---|
| `controllers/ik_qp/controller.py` | Renamed to `ik_qp_mujoco.py`; imports in scripts, tests and docs updated. Behaviour unchanged. |
| `controllers/ik_qp/config.py` | New. `PoseTarget` and `IKQPConfig` moved here and shared by both controllers. They are still importable from `ik_qp_mujoco`. |
| `controllers/ik_qp/ik_qp.py` | New. `PinkInverseKinematicsController(robot, IKQPConfig, solver="osqp" \| "daqp")`, with the same `compute(PoseTarget, dt) -> JointPositionCommand` interface. |
| `sim_interfaces/isaac/assets/franka/lula_franka_gen.urdf` | Unmodified copy of Isaac Lab's Franka URDF (IsaacLab @ `6ec3e9b`). License: `docs/licenses/IsaacLab-LICENSE`. |
| `pyproject.toml` | `isaac` extra: `pin-pink`, `qpsolvers<4.9` (4.9+ needs osqp ≥ 1.0, which `ik_qp_mujoco` does not support), `daqp==0.8.5` (same pin as Isaac Lab). |
| `tests/test_ik_qp_pink.py` | Runs both solvers on a Pinocchio-only robot (no simulator) whose base is shifted and rotated. Checks the result against the exact QP optimum (BVLS), checks bounds, and checks that bad joint names and solvers are rejected. |

Install: `uv sync --extra isaac --group test`. Test: `env -u PYTHONPATH uv run pytest -q`.

### How `ik_qp.py` maps the `ik_qp_mujoco` QP onto Pink

The decision variable is `Δq = dq·dt`.

- **Tasks:** two `FrameTask`s, one for position and one for orientation, so the two gains stay separate.
  - Position gain = `position_weight·kp_pos·dt`.
  - Orientation gain = `0.5·orientation_weight·kp_ori·dt`. The 0.5 is needed because Pink's SE(3)-log error is about θ, while the quaternion ε-error is about θ/2.
- **Damping:** Pink damping = `config.damping / 1e8`, where 1e8 is the task weight in `ik_qp_mujoco`.
- **Bounds:** the same box bounds as `ik_qp_mujoco`, `max(dq_lower, (q_min−q)/dt) ≤ dq ≤ min(dq_upper, (q_max−q)/dt)`, implemented as a custom Pink `Limit`.
- **Kinematics:** taken from the URDF. The robot provides `q`, the position limits and the world pose of `panda_link0`. World targets are re-expressed relative to `panda_link0`, so a moving base is handled.
- **Names:** the `RobotModel` must use URDF names (`panda_joint1..7`, `panda_hand`, …).
- **Fingers:** they are held still with zero velocity bounds. Pinocchio 2.6 can't build a reduced model from this URDF because `panda_end_effector` is the name of both a link and a joint.

## Solver comparison (osqp vs daqp)

- **Setup:** 100 random configurations near the `FRANKA_PANDA_CFG` home pose, 4 tuning cases (damping 5e3, 1e5 and 1e8; dq_max 20 and 0.3 rad/s), `dt = 0.01`.
- **Reference:** each result is compared with the exact optimum of `ik_qp_mujoco`'s QP, computed with bounded least squares.
- **Error metric:** relative error = max |dq − dq*| / max |dq*|.

| Pose error | Controller | Median rel. error | Worst rel. error | Max bound violation | Time / compute |
|---|---|---|---|---|---|
| 1 mm / 1 mrad | `ik_qp_mujoco` (OSQP 0.6 defaults) | 6e-13 | 3e-6 | 1e-14 | 0.72 ms |
| | Pink + osqp (eps 1e-7, polish) | 7e-5 | 2.9e-3 | 1e-14 | 0.88 ms |
| | Pink + daqp | 7e-5 | 2.9e-3 | 1e-14 | **0.28 ms** |
| 3 cm / 30 mrad | `ik_qp_mujoco` (OSQP 0.6 defaults) | 7e-14 | **5.8e-1** | 4e-13 | 0.71 ms |
| | Pink + osqp (eps 1e-7, polish) | 3.3e-3 | 7.8e-2 | 2e-14 | 0.91 ms |
| | Pink + daqp | 3.3e-3 | 7.8e-2 | 2e-13 | **0.29 ms** |

In the 3 cm runs, 64 of the 100 cases have a bound active at the optimum. In the 1 mm runs, 25 do.

- **osqp vs daqp in Pink:** the two give the same solution (relative difference ≤ 5e-8), and daqp is about 3× faster. With default tolerances, osqp was up to 76% off once a bound became active. That's why `ik_qp.py` sets `eps_abs = eps_rel = 1e-7` and `polish=True`.
- **What is left of the Pink error:** it comes from the formulation, not from the solver. Pink uses the SE(3) log, which couples rotation into the position error, while `ik_qp_mujoco` uses a decoupled position error plus a quaternion ε-error. The difference grows with pose error (worst case 7.8% at 3 cm).
- **`ik_qp_mujoco` with larger errors:** with OSQP 0.6 defaults it occasionally lands far from its own optimum (58% worst case at 3 cm). It is accurate in the 1 mm case.

## Still needed for `ik_qp`

1. **Isaac Lab 3 requirement.** Isaac Lab needs Python 3.12, while this repo pins Python < 3.12 and numpy 1.23.
   - Once that's resolved, drop the vendored URDF and load `isaaclab.controllers.config/data/lula_franka_gen.urdf` (see the `ik_qp.py` docstring).
   - Then revisit the `pin` (2.6 → 3.x), `qpsolvers<4.9` and osqp 0.6 pins.
2. **Isaac `RobotModel` adapter.** `ik_qp` has only been run on the Pinocchio test robot. It needs an Isaac Lab articulation adapter that exposes the `panda_*` names, position limits, `get_pose("panda_link0")` and `apply_command`.
3. **Closed-loop validation in Isaac Lab.** Port the equivalent of `scripts/mujoco/track_targets.py` and `base_disturbance.py`. The current tests only check single QP steps.
4. **Batching for RL.** The controller is a single-environment NumPy/CPU implementation, about 0.3 ms per step with daqp. Isaac Lab RL steps `num_envs` environments on GPU, so decide between one controller per environment and a batched version.
5. **End-effector frame.** The MuJoCo scripts track `grasp` (`hand` + 0.07 m). The URDF has `panda_hand` and `panda_end_effector` (`panda_hand` + 0.107 m) but no `grasp`. Agree on a TCP frame, or add one to the URDF.
6. **Gains.** The current scripts use `kp_pos = 80` with `dt = 0.02`, which gives a Pink position gain of 1.6. Pink documents gains in [0, 1], and gains above 1 overshoot within a step. Retune, or accept that it matches `ik_qp_mujoco`.
7. **Formulation choice.** Decide whether to keep the SE(3)-log error, including the 0.5 orientation factor, or match `ik_qp_mujoco` exactly. The two differ by up to about 8% for large pose errors.
8. **Default solver.** Choose between daqp, which Isaac Lab uses and which is exact and 3× faster here, and osqp, the Isaac Sim Pink default. The code currently defaults to `osqp`.
9. **Base motion.** The target is re-expressed at the current base pose, but neither controller feeds forward the base velocity. The RL-driven base will therefore cause tracking lag.
10. **`ik_qp_mujoco` accuracy.** Its OSQP defaults can be far from the optimum when pose errors are large. Apply the same tolerance and polish settings there if it stays in use.
