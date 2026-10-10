# Panda Import Notes

## Provenance

Copied from the local working tree of `../hybrid_push_control_panda`, at commit
`7053ba933800f5cb6634e32d35506ea2651a695a`. The initial import matched the source
byte for byte; Python caches were excluded. Subsequent changes are recorded below.

| Source | Destination |
| --- | --- |
| `model/` | `src/hybrid_control/backends/mujoco/models/panda/` |
| `controllers/ik_qp.py` | `src/hybrid_control/controllers/ik_qp/controller.py` |
| `commons/math_helpers.py` | `src/hybrid_control/utils/math_helpers.py` |

The model package preserves relative robot mesh paths. Its `panda.xml` describes
the arm; `scene.xml` includes the robot, viewer settings, and a ground plane at
world Z = 0 with a procedural grid texture. The original table XML files,
furniture textures, floor, and box were removed. The package retains
the standalone hand and MJX model variant. Robot dynamics and controller
algorithms are unchanged.

## Integration Status

The controller imports `MathHelpers` from `hybrid_control.utils.math_helpers`
and implements the generic `Controller` compute/reset contract. Its constructor
accepts a `RobotModel` and `IKQPConfig`; `compute(PoseTarget, dt)` returns a
`JointPositionCommand`. The imported QP formulation and numerical solver settings
are retained. Command application and simulation stepping are external.

`MujocoRobotModel` provides named joint state, limits, world poses/Jacobians, and
position command application. It converts MuJoCo quaternions to xyzw. The tracking
test uses this adapter for seeded position-only and position/orientation scenarios,
compensates joint bias forces, and checks tracking and joint limits using simulated
actuators. See [interface documentation](model_controller_interfaces.md) for the
shared conventions and ownership boundaries.

The original simulation workers, robot interfaces, pushing state machine,
entry point, logging, plotting, and videos are outside this import.

All six original requirements, including pandas and matplotlib, are declared
in `pyproject.toml`. Python is limited to 3.10–3.11 to support the retained
NumPy/SciPy versions. The `test` extra installs pytest; Isaac Sim integration
will receive a separate extra later. See the root README for run commands.

## License Notices

The controller and helper retain the source repository's
[MIT license](licenses/hybrid_push_control_panda-LICENSE).
The model package includes its original
[Apache-2.0 license](../src/hybrid_control/sim_interfaces/mujoco/models/panda/LICENSE),
README, and changelog. The source README credits MuJoCo Menagerie for the Panda
model and `furniture_sim` for the table and environment assets.
