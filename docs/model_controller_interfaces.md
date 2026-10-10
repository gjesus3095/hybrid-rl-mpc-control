# Robot models and controllers

The robot model exposes state, kinematics, and position command application for
one ordered group of joints. A controller computes a command. The experiment
owns the simulation and determines when to apply that command and advance physics.

This version supports NumPy and MuJoCo. Isaac integration, batched tensors,
mass matrices, force sensing, and velocity/effort commands are deferred.

## Shared contracts

`hybrid_control.robots.model.RobotModel` is an abstract base class with:

| Interface | Meaning |
| --- | --- |
| `joint_names` | Ordered tuple of selected scalar joint names |
| `position_limits` | Copies of lower/upper vectors `(n,)`; unlimited joints use infinities |
| `get_state()` | `JointState(time, positions, velocities)` in selected joint order |
| `get_com_state()` | `CenterOfMassState(time, position)` for the robot's mass-weighted world CoM |
| `get_pose(frame_name)` | `Pose(position, quaternion)` at a named body origin |
| `get_jacobian(frame_name)` | `(6, n)` matrix: world linear velocity followed by world angular velocity |
| `apply_command(command)` | Set selected position actuator references without advancing physics |

State and command vectors use radians for hinge joints and metres for slides;
velocities use the corresponding units per second. Time is simulated seconds.
Poses use world-frame metres and normalized **xyzw** quaternions. Jacobian columns
follow `joint_names`; the matrix gives the selected joints' contribution to body
motion, excluding any unselected base or gripper DoFs.

`Pose`, `JointState`, `CenterOfMassState`, and `JointPositionCommand` live in
`hybrid_control.robots.state`. They validate dimensions and finite values and own
read-only array copies. Pose construction normalizes nonzero quaternions and
rejects zero/nonfinite quaternions. Commands must contain exactly the receiving
model's joint names in its declared order.

## MuJoCo adapter

`MujocoRobotModel(model, data, joint_actuator_pairs)` wraps existing `MjModel` and
`MjData`. For example, the Panda arm mapping is:

```python
joint_actuator_pairs = tuple(
    (f"joint{i}", f"actuator{i}") for i in range(1, 8)
)
robot = MujocoRobotModel(model, data, joint_actuator_pairs)
```

CoM includes the fixed base and every descendant link, including unselected
gripper joints. By default its root is the common top-level body tree containing
the selected joints; other scene objects are excluded. Pass `root_body="name"`
to the constructor or `from_xml_path` to choose a subtree explicitly. That body
must contain all selected joints and have positive subtree mass; the world body
is not a robot root. `get_com_state()` reads MuJoCo's current `subtree_com` without
stepping; update kinematics after changing state, just as for pose reads.

For a standalone experiment, loading and initialization can be kept compact:

```python
robot = MujocoRobotModel.from_xml_path(XML_FILE, joint_actuator_pairs)
robot.initialize(
    Q_INIT,
    extra_joint_positions={"finger_joint1": 0.04, "finger_joint2": 0.04},
    extra_actuator_controls={"actuator8": 255},
)
```

`initialize` validates all names, positions, and controls before writing. It sets
the selected joint positions and matching actuator references, zeros velocities
for initialized joints, and refreshes kinematics. Optional named extras handle
unselected joints and actuators such as the Panda gripper. Initialization retains
simulation time, other joint state, other controls, and applied forces; it does
not advance physics or perform a full simulation reset.

The adapter resolves joint IDs, `qpos` addresses, velocity/Jacobian addresses, and
actuator IDs independently. Arbitrary joint order and noncontiguous selections
are supported, including scenes with unselected free joints. Selected joints
must be hinges or slides with direct, unit-gear position servos. Equivalent
stateless `general` servos, such as those in the Panda XML, are supported. Tendon,
motor, velocity, and activation-dynamics actuators are rejected for this interface.

The adapter exposes named **body origins**, not sites or body centres of mass.
It converts MuJoCo's wxyz quaternion to xyzw at the boundary. Reads do not call
`mj_forward` or `mj_step`; loading and initialization helpers refresh kinematics,
while the caller must do so after manually editing state. Do not step concurrently
with controller computation.

`apply_command` writes only the selected `data.ctrl` entries. It does not change
`qpos`, clamp references, reset state, or advance time. Joint limits are those in
the model; MuJoCo still enforces its configured actuator control and force limits
during physics. The gripper remains explicitly managed by the experiment.
The adapter offers XML loading and initialization helpers. Full simulation resets,
stepping, rendering, shutdown, and test-specific bias-force compensation remain
in the experiment. Native `model` and `data`
are available on the concrete adapter for those operations.

## Controller lifecycle and IK

`hybrid_control.controllers.base.Controller[TargetT, CommandT]` defines
`compute(target, dt)` and `reset()`. Computation may update a controller's own
memory but must not apply commands or change the robot. `reset()` clears internal
memory between episodes; its default implementation does nothing.

`InverseKinematicsController(robot, config)` implements this contract with
`PoseTarget(frame_name, pose)` and `JointPositionCommand`. `IKQPConfig` contains
per-joint velocity bounds, position/orientation gains and weights, and damping.
Bounds follow the model's joint order. The existing QP objective, solver settings,
and feasibility clamp are retained; the controller integrates its computed
velocity as `q_command = q_current + dq * dt`.

```python
from hybrid_control.sim_interfaces.mujoco.adapter import MujocoRobotModel
from hybrid_control.controllers.ik_qp.controller import (
    IKQPConfig, InverseKinematicsController, PoseTarget,
)
from hybrid_control.robots.state import Pose

# model/data are already initialized and kinematics have been refreshed.
robot = MujocoRobotModel(model, data, joint_actuator_pairs)
controller = InverseKinematicsController(robot, IKQPConfig(
    velocity_lower=DQ_MIN, velocity_upper=DQ_MAX,
    kp_pos=KP_POS, kp_ori=KP_ORI, damping=JOINT_VEL_WEIGHT,
))
target = PoseTarget("hand", Pose(target_position, target_quaternion_xyzw))
command = controller.compute(target, dt)
robot.apply_command(command)
# The experiment now performs its physics substeps and refreshes kinematics.
```

The previous `ik_qp_velocity_step_pose` and `send_position_command` entry points
are replaced by `compute`. All repository callers use the new interface. To inspect
the requested velocity, use `(command.positions - state.positions) / dt`, where
`state` is captured before computation.

To add a controller, subclass `Controller` with its target and command types,
accept the dependencies it needs in its constructor, implement `compute`, and
override `reset` only if it has internal state. Keep simulator access in a model
adapter. A future controller can return another command type without changing
the base class; applying a new command mode will require an explicit model API
extension and compatible simulator actuators.

## Validation

Run all contract and Panda tracking checks with:

```bash
env -u PYTHONPATH uv run --extra test pytest -q
```

The tests exercise named and reordered mappings, free-joint address differences,
linear/angular Jacobians against finite differences, quaternion conventions,
invalid inputs, command isolation, computation without simulation changes, and
IK against a small simulator-independent model. The existing seeded Panda
position-only and combined pose tracking scenarios retain their acceptance
thresholds. Configure those experiments using the constants at the top of
`tests/test_ik_qp.py`; standalone execution retains the viewer and grid ground.
The viewer draws the target in red and a green, 5 cm diameter sphere at the
robot CoM. The CoM marker updates each control cycle and is display-only, with
no added mass or collisions. Its size is configured by `COM_MARKER_DIAMETER`.
