"""MuJoCo implementation of the single-robot model contract."""

from collections.abc import Mapping, Sequence
from os import PathLike

import mujoco
import numpy as np

from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import CenterOfMassState, JointPositionCommand, JointState, Pose


class MujocoRobotModel(RobotModel):
    """Wrap MjModel/MjData, with helpers for loading and initializing a scene.

    ``joint_actuator_pairs`` is an ordered sequence of (joint, actuator) names.
    Selected joints must be hinges/slides driven by direct unit-gear position
    servos (including equivalent ``general`` actuators). Other joints, such as
    a free base or a tendon-driven gripper, may exist outside this selection.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_actuator_pairs: Sequence[tuple[str, str]],
        *,
        root_body: str | None = None,
    ):
        """Initialize the adapter with the MuJoCo model, state, and selected joint-actuator pairs.

        The selected pairs define the controlled joints and the corresponding actuators that
        are expected to operate as direct position servos. The constructor validates each
        pairing and precomputes the relevant MuJoCo index arrays used by later state and
        command operations.
        """
        pairs = tuple(joint_actuator_pairs)
        if not pairs or any(
            not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(not isinstance(name, str) or not name for name in pair)
            for pair in pairs
        ):
            raise ValueError("joint_actuator_pairs must contain (joint, actuator) name pairs")
        self.model = model
        self.data = data
        self._joint_names = tuple(pair[0] for pair in pairs)
        actuator_names = tuple(pair[1] for pair in pairs)
        if len(set(self._joint_names)) != len(pairs) or len(set(actuator_names)) != len(pairs):
            raise ValueError("Joint and actuator names must each be unique")

        joint_ids = np.array([
            self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self._joint_names
        ])
        self._actuator_ids = np.array([
            self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names
        ])
        for joint_id, actuator_id in zip(joint_ids, self._actuator_ids):
            self._validate_position_servo(joint_id, actuator_id)

        self._qpos_indices = model.jnt_qposadr[joint_ids].copy()
        self._dof_indices = model.jnt_dofadr[joint_ids].copy()
        limited = model.jnt_limited[joint_ids].astype(bool)
        ranges = model.jnt_range[joint_ids]
        self._lower = np.where(limited, ranges[:, 0], -np.inf)
        self._upper = np.where(limited, ranges[:, 1], np.inf)
        # Scope CoM to the robot's tree, excluding unrelated objects and robots.
        body_ids = model.jnt_bodyid[joint_ids]
        if root_body is None:
            roots = {self._top_level_body(int(body_id)) for body_id in body_ids}
            if len(roots) != 1:
                raise ValueError("Selected joints must belong to one robot body tree")
            self._root_body_id = roots.pop()
        else:
            self._root_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, root_body)
            for body_id in body_ids:
                while body_id != 0 and body_id != self._root_body_id:
                    body_id = model.body_parentid[body_id]
                if body_id != self._root_body_id:
                    raise ValueError("root_body must contain all selected joints")
        if self._root_body_id == 0 or model.body_subtreemass[self._root_body_id] <= 0:
            raise ValueError("Robot root must be a non-world body with positive subtree mass")

    def _top_level_body(self, body_id: int) -> int:
        while self.model.body_parentid[body_id] != 0:
            body_id = int(self.model.body_parentid[body_id])
        return body_id

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str | PathLike,
        joint_actuator_pairs: Sequence[tuple[str, str]],
        *,
        root_body: str | None = None,
    ) -> "MujocoRobotModel":
        """Load a MuJoCo model from XML and create a model adapter around its scene state.

        The XML path is used to instantiate the underlying MuJoCo model and data objects,
        after which the current kinematics are propagated once so the initial scene state is
        consistent before the adapter is used.
        """
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return cls(model, data, joint_actuator_pairs, root_body=root_body)

    def initialize(
        self,
        positions: Sequence[float],
        *,
        extra_joint_positions: Mapping[str, float] | None = None,
        extra_actuator_controls: Mapping[str, float] | None = None,
    ) -> None:
        """Set selected joints and matching servo references, then refresh kinematics.

        Positions follow ``joint_names`` in radians/metres. Optional named extras
        initialize unselected scalar joints and actuator controls, e.g. a gripper.
        All inputs are checked before writing. Initialized joint velocities are
        zeroed; time, other joints, applied forces, and other controls are retained.
        This explicitly sets state without advancing or resetting the simulation.
        """
        command = JointPositionCommand(self.joint_names, positions)
        if np.any(command.positions < self._lower) or np.any(command.positions > self._upper):
            raise ValueError("Initial positions must lie within joint limits")
        qpos_indices = list(self._qpos_indices)
        dof_indices = list(self._dof_indices)
        joint_values = list(command.positions)
        controls = dict(zip(self._actuator_ids, command.positions))

        for name, value in (extra_joint_positions or {}).items():
            joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            if name in self.joint_names:
                raise ValueError(f"Extra joint {name!r} overlaps the selected joints")
            if self.model.jnt_type[joint_id] not in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(f"Extra joint {name!r} must be a scalar hinge or slide")
            lower, upper = self.model.jnt_range[joint_id]
            if not np.isfinite(value) or (
                self.model.jnt_limited[joint_id] and not lower <= value <= upper
            ):
                raise ValueError(f"Initial position for {name!r} must be finite and within joint limits")
            qpos_indices.append(self.model.jnt_qposadr[joint_id])
            dof_indices.append(self.model.jnt_dofadr[joint_id])
            joint_values.append(value)

        for name, value in (extra_actuator_controls or {}).items():
            actuator_id = self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id in controls:
                raise ValueError(f"Extra actuator {name!r} overlaps the selected actuators")
            controls[actuator_id] = value
        for actuator_id, value in controls.items():
            lower, upper = self.model.actuator_ctrlrange[actuator_id]
            if not np.isfinite(value) or (
                self.model.actuator_ctrllimited[actuator_id] and not lower <= value <= upper
            ):
                name = self.model.actuator(actuator_id).name
                raise ValueError(f"Initial control for {name!r} must be finite and within control limits")

        self.data.qpos[qpos_indices] = joint_values
        self.data.qvel[dof_indices] = 0.0
        self.data.ctrl[list(controls)] = list(controls.values())
        mujoco.mj_forward(self.model, self.data)

    def _name_id(self, kind: mujoco.mjtObj, name: str) -> int:
        """Resolve a MuJoCo object name to its internal index and validate that it exists."""
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"Unknown {kind.name}: {name!r}")
        return index

    def _validate_position_servo(self, joint_id: int, actuator_id: int) -> None:
        """Ensure a selected joint-actuator pairing behaves like a direct position servo."""
        model = self.model
        joint_name = model.joint(joint_id).name
        actuator_name = model.actuator(actuator_id).name
        if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Joint {joint_name!r} must be a scalar hinge or slide")
        if (
            model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT
            or model.actuator_trnid[actuator_id, 0] != joint_id
            or not np.array_equal(model.actuator_gear[actuator_id], [1, 0, 0, 0, 0, 0])
        ):
            raise ValueError(
                f"Actuator {actuator_name!r} must directly drive {joint_name!r} with unit gear"
            )

        gain = model.actuator_gainprm[actuator_id, 0]
        bias = model.actuator_biasprm[actuator_id, :3]
        if not (
            model.actuator_dyntype[actuator_id] == mujoco.mjtDyn.mjDYN_NONE
            and model.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED
            and model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE
            and np.isfinite(gain) and gain > 0
            and np.all(np.isfinite(bias))
            and bias[0] == 0 and np.isclose(bias[1], -gain) and bias[2] <= 0
        ):
            raise ValueError(f"Actuator {actuator_name!r} must be a stateless position servo")

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the selected joint names in their configured control order."""
        return self._joint_names

    @property
    def position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper position bounds for the selected joints as copies."""
        return self._lower.copy(), self._upper.copy()

    def get_state(self) -> JointState:
        """Return the current simulation time, joint positions, and joint velocities."""
        return JointState(
            float(self.data.time),
            self.data.qpos[self._qpos_indices],
            self.data.qvel[self._dof_indices],
        )

    def get_com_state(self) -> CenterOfMassState:
        """Read the robot subtree's mass-weighted world CoM after kinematics update.

        Includes the fixed base and all descendant links, including unselected
        gripper joints. MuJoCo maintains subtree_com during mj_forward/mj_step;
        the read itself does not advance or alter simulation state.
        """
        return CenterOfMassState(
            float(self.data.time), self.data.subtree_com[self._root_body_id],
        )

    def get_pose(self, frame_name: str) -> Pose:
        """Return the world position and quaternion for a named body frame."""
        body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, frame_name)
        return Pose(self.data.xpos[body_id], self.data.xquat[body_id, [1, 2, 3, 0]])

    def get_jacobian(self, frame_name: str) -> np.ndarray:
        """Return the spatial Jacobian for a named body relative to the selected joints."""
        body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, frame_name)
        linear = np.zeros((3, self.model.nv))
        angular = np.zeros_like(linear)
        mujoco.mj_jacBody(self.model, self.data, linear, angular, body_id)
        return np.vstack((linear, angular))[:, self._dof_indices].copy()

    def apply_command(self, command: JointPositionCommand) -> None:
        """Write a joint-position command to the underlying MuJoCo actuator controls."""
        if not isinstance(command, JointPositionCommand):
            raise TypeError("Expected a JointPositionCommand")
        if command.joint_names != self.joint_names:
            raise ValueError(f"Command joint names/order must equal {self.joint_names}")
        if (
            command.positions.shape != (len(self.joint_names),)
            or not np.all(np.isfinite(command.positions))
        ):
            raise ValueError("Command positions must be finite and match the selected joints")
        self.data.ctrl[self._actuator_ids] = command.positions
