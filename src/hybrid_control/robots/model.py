"""Read robot state and kinematics, and apply commands independently of a simulator."""

from abc import ABC, abstractmethod

import numpy as np

from hybrid_control.robots.state import (
    CenterOfMassState, JointPositionCommand, JointState, JointTorqueCommand, Pose,
)


class RobotModel(ABC):
    """One ordered group of scalar joints in an externally managed simulation.

    Reads never step physics or change state. The caller must update simulation
    kinematics before reading and must not step concurrently with controller
    computation. Returned arrays must not alias simulator-owned memory.
    """

    @property
    @abstractmethod
    def joint_names(self) -> tuple[str, ...]:
        """Controlled joints, in state/limit/Jacobian column order."""

    @property
    @abstractmethod
    def position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper joint limits (n,), radians/metres; unbounded uses ±inf."""

    @abstractmethod
    def get_state(self) -> JointState:
        """Return a snapshot of the selected joints and simulation time."""

    @abstractmethod
    def get_com_state(self) -> CenterOfMassState:
        """Return the mass-weighted CoM of the robot, including unselected links."""

    @abstractmethod
    def get_pose(self, frame_name: str) -> Pose:
        """Return a named body origin's world pose, with an xyzw quaternion."""

    @abstractmethod
    def get_jacobian(self, frame_name: str) -> np.ndarray:
        """Return (6, n): world linear then angular velocity at the body origin.

        Multiplying by the selected joint velocities gives their contribution
        to the body's world twist (m/s, rad/s). Unselected DoFs are excluded.
        """

    @abstractmethod
    def apply_command(self, command: JointPositionCommand) -> None:
        """Set selected position actuator references without stepping or clipping.

        Commands must use exactly this model's joint names and order. Physics
        and actuator force/control limits remain the simulation's responsibility.
        """


class TorqueRobotModel(RobotModel):
    """Robot model whose selected joints are driven by direct torque (motor) actuators.

    Adds the joint-space dynamics needed by torque-level controllers. All quantities
    are restricted to the selected joints, in ``joint_names`` order, and satisfy

        M(q) ddq + h(q, dq) = tau

    where coupling to unselected DoFs (e.g. gripper fingers) is neglected.
    """

    @property
    @abstractmethod
    def torque_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper actuator torque limits (n,), N·m for hinges and N for slides."""

    @abstractmethod
    def get_mass_matrix(self) -> np.ndarray:
        """Return the symmetric joint-space inertia matrix M(q), shape (n, n)."""

    @abstractmethod
    def get_nonlinear_forces(self) -> np.ndarray:
        """Return h(q, dq) (n,): Coriolis, centrifugal and gravity minus passive forces."""

    @abstractmethod
    def get_jacobian_derivative(self, frame_name: str) -> np.ndarray:
        """Return dJ/dt (6, n) for ``get_jacobian(frame_name)`` at the current state.

        ``get_jacobian_derivative(frame) @ dq`` is the twist acceleration bias, so the
        body's world acceleration is ``J ddq + dJ dq`` (m/s², rad/s²).
        """

    @abstractmethod
    def apply_command(self, command: JointTorqueCommand) -> None:
        """Set selected motor torques without stepping or clipping.

        Commands must use exactly this model's joint names and order. Actuator
        control limits remain the simulation's responsibility.
        """
