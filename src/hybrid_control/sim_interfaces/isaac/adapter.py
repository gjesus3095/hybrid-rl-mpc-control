"""Isaac Lab articulation exposed as one ``RobotModel`` per environment.

All environments share an ``IsaacArticulationBuffer``: ``read()`` copies the simulator state
GPU->CPU once per control step and ``write()`` sends all joint targets CPU->GPU once, so the
per-environment controllers only touch numpy buffers in between. No Isaac Lab import is needed;
torch/warp are imported only inside ``read()``/``write()``. ``IsaacDynamicsBuffer`` and
``IsaacTorqueRobotModel`` add the dynamics and effort targets for torque control.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from hybrid_control.robots.model import ControlMode, RobotModel
from hybrid_control.robots.state import CenterOfMassState, JointPositionCommand, JointState, JointTorqueCommand, Pose


def compose(parent: Pose, offset: Pose) -> Pose:
    """World pose of ``offset`` expressed in ``parent``."""
    rotation = Rotation.from_quat(np.array(parent.quaternion))
    return Pose(parent.position + rotation.apply(np.array(offset.position)),
                (rotation * Rotation.from_quat(np.array(offset.quaternion))).as_quat())


class IsaacArticulationBuffer:
    """CPU snapshot of an Isaac Lab ``Articulation`` and its pending joint position targets."""

    def __init__(self, articulation, joint_names: Sequence[str]):
        self.articulation = articulation
        self.joint_names = tuple(joint_names)
        missing = [n for n in self.joint_names if n not in articulation.joint_names]
        if missing:
            raise ValueError(f"Joints {missing} are not in the articulation")
        self.joint_ids = [articulation.joint_names.index(n) for n in self.joint_names]
        self.body_ids = {n: i for i, n in enumerate(articulation.body_names)}
        data = articulation.data
        self.joint_pos_limits = data.joint_pos_limits.torch.cpu().numpy()[:, self.joint_ids]  # static
        self.body_mass = data.body_mass.torch.cpu().numpy()
        self.time = 0.0
        self.read(0.0)
        self.targets = self.joint_pos[:, self.joint_ids].copy()
        # Velocity feedforward for the implicit PD: without it the damping term pulls the joint
        # velocity towards 0 and the joints only reach a fraction of the commanded velocity.
        self.velocity_targets = np.zeros_like(self.targets)

    @property
    def num_envs(self) -> int:
        return self.joint_pos.shape[0]

    def read(self, time: float) -> None:
        """Copy the current simulator state to the CPU (call after ``scene.update``)."""
        data = self.articulation.data
        self.time = float(time)
        self.joint_pos = data.joint_pos.torch.cpu().numpy()
        self.joint_vel = data.joint_vel.torch.cpu().numpy()
        self.body_pose = data.body_link_pose_w.torch.cpu().numpy()  # (envs, bodies, 7), xyzw
        self.body_com = data.body_com_pos_w.torch.cpu().numpy()

    def write(self) -> None:
        """Send all pending joint position and velocity targets to the simulator (applied at the next sim step)."""
        import torch  # only needed inside Isaac Lab

        device, command = self.articulation.device, self.articulation.actuators.target_command
        command.set_position_index(value=torch.as_tensor(self.targets, dtype=torch.float32, device=device),
                                   joint_ids=self.joint_ids)
        command.set_velocity_index(value=torch.as_tensor(self.velocity_targets, dtype=torch.float32, device=device),
                                   joint_ids=self.joint_ids)

    def com(self, env: int, body_names: Sequence[str]) -> np.ndarray:
        """Mass-weighted world CoM of the named bodies."""
        ids = [self.body_ids[n] for n in body_names]
        mass = self.body_mass[env, ids]
        return mass @ self.body_com[env, ids] / mass.sum()

    def coms(self, body_names: Sequence[str]) -> np.ndarray:
        """``com`` of all environments, (num_envs, 3)."""
        ids = [self.body_ids[n] for n in body_names]
        mass = self.body_mass[:, ids]
        return np.einsum("nb,nbk->nk", mass, self.body_com[:, ids]) / mass.sum(axis=1, keepdims=True)


class IsaacRobotModel(RobotModel):
    """``RobotModel`` view of environment ``env`` in an ``IsaacArticulationBuffer``.

    ``extra_frames`` = ``{name: (parent_body, offset)}`` adds fixed frames that are not USD bodies
    (e.g. the TCP). ``pose_overrides`` replaces the reported pose of a frame, e.g. the base pose
    predicted after the commanded base action, which the IK/QP should react to.
    """

    @property
    def control_mode(self) -> ControlMode:
        return "position"

    def __init__(self, buffer: IsaacArticulationBuffer, env: int,
                 extra_frames: Mapping[str, tuple[str, Pose]] | None = None, *,
                 expected_control_mode: ControlMode | None = None):
        super().__init__(expected_control_mode=expected_control_mode)
        self.buffer = buffer
        self.env = env
        self.extra_frames = dict(extra_frames or {})
        self.pose_overrides: dict[str, Pose] = {}

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.buffer.joint_names

    @property
    def position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        limits = self.buffer.joint_pos_limits[self.env]
        return limits[:, 0].astype(float), limits[:, 1].astype(float)

    def get_state(self) -> JointState:
        ids = self.buffer.joint_ids
        return JointState(self.buffer.time, self.buffer.joint_pos[self.env, ids], self.buffer.joint_vel[self.env, ids])

    def get_com_state(self) -> CenterOfMassState:
        return CenterOfMassState(self.buffer.time, self.buffer.com(self.env, self.buffer.articulation.body_names))

    def get_pose(self, frame_name: str) -> Pose:
        if frame_name in self.pose_overrides:
            return self.pose_overrides[frame_name]
        if frame_name in self.extra_frames:
            parent, offset = self.extra_frames[frame_name]
            return compose(self.get_pose(parent), offset)
        pose = self.buffer.body_pose[self.env, self._body_id(frame_name)]
        return Pose(pose[:3], pose[3:])

    def get_jacobian(self, frame_name: str) -> np.ndarray:
        """Body-origin Jacobian, read from the GPU on each call (not used by the Pink IK)."""
        articulation = self.buffer.articulation
        body = self._body_id(frame_name) - (1 if articulation.is_fixed_base else 0)  # fixed root has no row
        columns = [j + articulation.num_base_dofs for j in self.buffer.joint_ids]
        return articulation.data.body_link_jacobian_w.torch[self.env, body][:, columns].cpu().numpy().astype(float)

    def apply_command(self, command: JointPositionCommand) -> None:
        if tuple(command.joint_names) != self.joint_names:
            raise ValueError("Command joints must match the model's joint order")
        self.buffer.targets[self.env] = command.positions

    def _body_id(self, name: str) -> int:
        if name not in self.buffer.body_ids:
            raise ValueError(f"Body '{name}' is not in the articulation")
        return self.buffer.body_ids[name]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


class IsaacDynamicsBuffer(IsaacArticulationBuffer):
    """``IsaacArticulationBuffer`` plus the dynamics a torque controller needs, read once per step.

    Per environment, restricted to ``joint_names`` (coupling to other joints, e.g. the base, is
    neglected): mass matrix M, nonlinear forces h = gravity + Coriolis/centrifugal compensation
    forces (PhysX), and the world Jacobians at the origins of ``jacobian_bodies``. ``write()``
    sends ``efforts`` as joint effort targets; the joints' implicit PD gains should be zero.
    """

    def __init__(self, articulation, joint_names: Sequence[str], jacobian_bodies: Sequence[str]):
        data = articulation.data
        if data.has_joint_ordering or getattr(data, "_has_reversed_joints", False):
            raise ValueError("Coriolis forces are read in PhysX joint order; reordered joints are not supported")
        self.jacobian_bodies = tuple(jacobian_bodies)
        super().__init__(articulation, joint_names)
        self.effort_limits = data.joint_effort_limits.torch.cpu().numpy()[:, self.joint_ids]
        self.efforts = np.zeros_like(self.targets)

    def read(self, time: float) -> None:
        super().read(time)
        articulation, data = self.articulation, self.articulation.data
        dofs = [j + articulation.num_base_dofs for j in self.joint_ids]
        self.mass_matrix = data.mass_matrix.torch[:, dofs][:, :, dofs].cpu().numpy()
        self.nonlinear_forces = (data.gravity_compensation_forces.torch + self._coriolis())[:, dofs].cpu().numpy()
        jacobians = data.body_link_jacobian_w.torch  # fixed base: the root body has no row
        offset = 1 if articulation.is_fixed_base else 0
        self.jacobians = {b: jacobians[:, self.body_ids[b] - offset][:, :, dofs].cpu().numpy()
                          for b in self.jacobian_bodies}

    def _coriolis(self):
        """Coriolis/centrifugal compensation forces (envs, dofs); Isaac Lab's data has no accessor for them."""
        import warp as wp  # only needed inside Isaac Lab

        return wp.to_torch(self.articulation.root_view.get_coriolis_and_centrifugal_compensation_forces())

    def write(self) -> None:
        """Send all pending joint effort targets to the simulator (applied at the next sim step)."""
        import torch

        self.articulation.actuators.target_command.set_effort_index(
            value=torch.as_tensor(self.efforts, dtype=torch.float32, device=self.articulation.device),
            joint_ids=self.joint_ids)


class IsaacTorqueRobotModel(IsaacRobotModel):
    """Torque-controlled ``RobotModel`` view of environment ``env`` in an ``IsaacDynamicsBuffer``.

    Jacobians of ``extra_frames`` are their parent body's Jacobian shifted to the frame origin.
    ``get_jacobian_derivative`` is a backward difference over the last two ``buffer.read`` times
    (zero on the first step); PhysX has no dJ/dt.
    """

    @property
    def control_mode(self) -> ControlMode:
        return "torque"

    def __init__(self, buffer: IsaacDynamicsBuffer, env: int,
                 extra_frames: Mapping[str, tuple[str, Pose]] | None = None, *,
                 expected_control_mode: ControlMode | None = None):
        super().__init__(buffer, env, extra_frames, expected_control_mode=expected_control_mode)
        self._jacobian_history: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
        self._jacobian_cache: dict[str, tuple[float, np.ndarray]] = {}  # the controller asks several times per step

    @property
    def torque_limits(self) -> tuple[np.ndarray, np.ndarray]:
        limit = self.buffer.effort_limits[self.env].astype(float)
        return -limit, limit

    def get_mass_matrix(self) -> np.ndarray:
        return self.buffer.mass_matrix[self.env].astype(float)

    def get_nonlinear_forces(self) -> np.ndarray:
        return self.buffer.nonlinear_forces[self.env].astype(float)

    def get_jacobian(self, frame_name: str) -> np.ndarray:
        """World Jacobian (6, n) at the frame origin; read-only and cached until the next ``buffer.read``."""
        time, jacobian = self._jacobian_cache.get(frame_name, (None, None))
        if time != self.buffer.time:
            jacobian = self._compute_jacobian(frame_name)
            jacobian.setflags(write=False)
            self._jacobian_cache[frame_name] = (self.buffer.time, jacobian)
        return jacobian

    def _compute_jacobian(self, frame_name: str) -> np.ndarray:
        body = self.extra_frames[frame_name][0] if frame_name in self.extra_frames else frame_name
        if body not in self.buffer.jacobians:
            raise ValueError(f"Body '{body}' is not in the buffer's jacobian_bodies")
        jacobian = self.buffer.jacobians[body][self.env].astype(float)
        if body != frame_name:  # v_frame = v_body + w x r = v_body - [r]x w
            r = self.get_pose(frame_name).position - self.get_pose(body).position
            jacobian[:3] -= _skew(r) @ jacobian[3:]
        return jacobian

    def get_jacobian_derivative(self, frame_name: str) -> np.ndarray:
        time, jacobian = self.buffer.time, self.get_jacobian(frame_name)
        last_time, last_jacobian, derivative = self._jacobian_history.get(
            frame_name, (time, jacobian, np.zeros_like(jacobian)))
        if time > last_time:
            derivative = (jacobian - last_jacobian) / (time - last_time)
        if time != last_time or frame_name not in self._jacobian_history:
            self._jacobian_history[frame_name] = (time, jacobian, derivative)
        return derivative

    def apply_command(self, command: JointTorqueCommand) -> None:
        if not isinstance(command, JointTorqueCommand):
            raise TypeError("Expected a JointTorqueCommand")
        if tuple(command.joint_names) != self.joint_names:
            raise ValueError("Command joints must match the model's joint order")
        self.buffer.efforts[self.env] = command.torques
