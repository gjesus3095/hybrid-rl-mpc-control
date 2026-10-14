"""Read robot state and kinematics, and apply commands independently of a simulator."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from hybrid_control.robots.state import (
    CenterOfMassState, JointPositionCommand, JointState, JointTorqueCommand, JointVelocityCommand, Pose,
)


ControlMode = Literal["position", "velocity", "torque"]


class RobotModel(ABC):
    """One ordered group of scalar joints in an externally managed simulation.

    Reads never step physics or change state. The caller must update simulation
    kinematics before reading and must not step concurrently with controller
    computation. Returned arrays must not alias simulator-owned memory.
    """

    def __init__(self, *, expected_control_mode: ControlMode | None = None):
        """Validate the model mode and, when supplied, the intended controller mode.

        Subclasses must initialize mode-dependent state before calling this constructor.
        """
        self.validate_control_mode(self.control_mode)
        if expected_control_mode is not None:
            self.validate_control_mode(expected_control_mode)

    @property
    @abstractmethod
    def control_mode(self) -> ControlMode:
        """Accepted command mode for the selected joint group."""

    def validate_control_mode(self, expected: ControlMode) -> None:
        """Reject a controller mode incompatible with this model."""
        if expected not in ("position", "velocity", "torque"):
            raise ValueError(f"Unknown control mode: {expected!r}")
        if self.control_mode != expected:
            raise ValueError(
                f"Controller requires control_mode={expected!r}, "
                f"but {type(self).__name__} has control_mode={self.control_mode!r}"
            )

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
    def apply_command(
        self, command: JointPositionCommand | JointVelocityCommand | JointTorqueCommand,
    ) -> None:
        """Set selected actuator references matching control_mode without stepping or clipping.

        Commands must use exactly this model's joint names and order. Physics
        and actuator force/control limits remain the simulation's responsibility.
        """

    @property
    def torque_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper direct-torque limits (n,), N·m for hinges and N for slides."""
        raise NotImplementedError(f"{type(self).__name__} does not support direct torque limits")

    def get_mass_matrix(self) -> np.ndarray:
        """Return selected joint-space inertia M(q), shape (n, n)."""
        raise NotImplementedError(f"{type(self).__name__} does not provide a mass matrix")

    def get_nonlinear_forces(self) -> np.ndarray:
        """Return h(q, dq) (n,): Coriolis, centrifugal and gravity minus passive forces.

        M ddq + h = tau neglects coupling to unselected DoFs.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide nonlinear forces")

    def get_jacobian_derivative(self, frame_name: str) -> np.ndarray:
        """Return dJ/dt (6, n) for ``get_jacobian(frame_name)`` at the current state.

        ``get_jacobian_derivative(frame) @ dq`` is the twist acceleration bias, so the
        body's world acceleration is ``J ddq + dJ dq`` (m/s², rad/s²).
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide Jacobian derivatives")
