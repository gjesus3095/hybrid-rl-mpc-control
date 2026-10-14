# Robot model and controller interfaces

The robot model reads state and kinematics and applies commands. A controller computes a command from a target. The experiment owns the simulation and decides when to step physics.

## RobotModel

`hybrid_control.robots.model.RobotModel` is the abstract interface. MuJoCo and Isaac adapters implement it.

| Method / Property | Returns | Notes |
|---|---|---|
| `control_mode` | `"position"`, `"velocity"`, or `"torque"` | Read-only; accepted command type |
| `joint_names` | `tuple[str, ...]` | Selected joints in control order |
| `position_limits` | `(lower, upper)` each `(n,)` | Copies; unlimited joints use ±inf |
| `get_state()` | `JointState(time, positions, velocities)` | Snapshot; does not step physics |
| `get_com_state()` | `CenterOfMassState(time, position)` | Mass-weighted world CoM of whole robot |
| `get_pose(frame_name)` | `Pose(position, quaternion)` | Body origin in world frame |
| `get_jacobian(frame_name)` | `np.ndarray (6, n)` | World linear rows first, then angular |
| `apply_command(command)` | `None` | Writes actuator references; does not step |

**Units:** radians for hinge joints, metres for slides; velocities in those units/second.  
**Quaternions:** xyzw order everywhere (MuJoCo's internal wxyz is converted at the boundary).  
**Jacobian:** `J @ dq` gives world twist `[vx, vy, vz, wx, wy, wz]` in m/s and rad/s.

State classes live in `hybrid_control.robots.state`: `JointState`, `CenterOfMassState`, `Pose`, `JointPositionCommand`, `JointVelocityCommand`, `JointTorqueCommand`. They copy inputs, validate on construction, and return read-only arrays.

## MujocoRobotModel

Wraps `MjModel` and `MjData` with explicit joint-to-actuator mappings and a required
`control_mode`. Selected actuators must match the mode: stateless position or velocity
servos, or unit-gain torque motors, all driving scalar joints directly with unit gear.
Equivalent `general` actuators are accepted. The adapter validates rather than converts
actuators. Base actuators and gripper extras are outside this selected group.

`initialize` zeros selected joint velocities and sets selected controls to the initial
positions in position mode, or zero in velocity/torque mode. `apply_command` accepts only
the matching `JointPositionCommand`, `JointVelocityCommand`, or `JointTorqueCommand`,
with exact joint order and finite `(n,)` values; invalid commands leave controls unchanged.

Mass matrices, nonlinear forces, and Jacobian derivatives are available in all MuJoCo
modes. `torque_limits` is available only in torque mode and intersects actuator control
and force limits. Other `RobotModel` implementations raise `NotImplementedError` for
unsupported dynamics operations.

`MujocoTorqueRobotModel` and the shared `TorqueRobotModel` have been removed. Replace
MuJoCo torque construction with `MujocoRobotModel(..., control_mode="torque")`. Isaac
retains `IsaacRobotModel` (position) and `IsaacTorqueRobotModel` (torque).

```python
from hybrid_control.sim_interfaces.mujoco.adapter import MujocoRobotModel

joint_actuator_pairs = tuple((f"joint{i}", f"actuator{i}") for i in range(1, 8))

# Load from XML (refreshes kinematics once)
robot = MujocoRobotModel.from_xml_path("scene.xml", joint_actuator_pairs, control_mode="position")

# Set initial pose; zeros selected joint velocities, then calls mj_forward
robot.initialize(
    Q_INIT,
    extra_joint_positions={"finger_joint1": 0.04, "finger_joint2": 0.04},
    extra_actuator_controls={"actuator8": 255},
)
```

`apply_command` writes `data.ctrl` only — it does not clamp, change `qpos`, or step physics. The caller must call `mj_forward` after editing state manually. The native `model` and `data` are available on the instance.

CoM is scoped to the robot's body subtree, excluding other objects in the scene. Override with `root_body="name"` if the auto-detected root is wrong.

## Controller

`hybrid_control.controllers.base.Controller[TargetT, CommandT]` declares a read-only
`control_mode` describing its output commands. Both IK controllers declare `"position"`
(their velocity solutions are integrated); impedance declares `"torque"`.

Concrete constructors call `super().__init__(robot)` before solver setup. This checks
`robot.validate_control_mode(self.control_mode)` and raises `ValueError` on a mismatch.
Scripts therefore fail during construction, before the control loop. Batched Pink IK
also exposes `control_mode`; pass it as `expected_control_mode` to the Isaac model
constructor to check compatibility before configuring workers. Robot model constructors
call `super().__init__()` to validate their declared mode automatically. New controllers
must declare their mode and call the base constructor.

Lifecycle methods:

- `compute(target, dt)` — returns a command; must not apply it or mutate the simulation.
- `reset()` — clears internal state between episodes.

### IK/QP controller

`utils/control_utils.py` defines one `TaskSpaceTarget` for IK and impedance, plus
`CartesianSample`, the controller configs, and impedance diagnostics. Targets contain a
world-frame pose and optional world twist/acceleration (linear then angular), defaulting
to zero. IK consumes the pose; impedance also consumes the derivatives.

`InverseKinematicsController` takes a `TaskSpaceTarget` and returns a `JointPositionCommand`.

```python
from hybrid_control.controllers.ik_qp.ik_qp_mujoco import (
    InverseKinematicsController,
)
from hybrid_control.utils.control_utils import IKQPConfig, TaskSpaceTarget
from hybrid_control.robots.state import Pose

config = IKQPConfig(
    velocity_lower=DQ_MIN, velocity_upper=DQ_MAX,
    kp_pos=80.0, kp_ori=60.0, damping=1e5,
)
controller = InverseKinematicsController(robot, config)

target = TaskSpaceTarget("hand", Pose(target_position, target_quaternion_xyzw))
command = controller.compute(target, dt)   # does not apply the command
robot.apply_command(command)               # writes actuator references
# caller steps physics and calls mj_forward
```

The QP solves for joint velocities bounded by `velocity_lower`/`velocity_upper`, then integrates: `q_cmd = q_current + dq * dt`.

### Pink IK/QP controller (Isaac Lab assets)

`PinkInverseKinematicsController` in `controllers/ik_qp/ik_qp.py` takes the same `IKQPConfig` and `TaskSpaceTarget` and solves the same QP with Pink, using kinematics from Isaac Lab's Franka URDF instead of `robot.get_jacobian`. The `RobotModel` must use the URDF names (`panda_joint1..7`, frames such as `panda_link0`, `panda_hand`). Targets are world-frame and re-expressed relative to `panda_link0`, so a moving base is handled. Requires `uv sync --extra isaac`.

## Running tests and the tracking experiment

```bash
# Contract tests (disable unrelated externally installed pytest plugins)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q

# Pose tracking experiment — edit SHOW_VIEWER and constants at the top
uv run python scripts/mujoco/track_targets.py
```
