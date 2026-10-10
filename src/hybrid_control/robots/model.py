"""Read robot state and kinematics, and apply commands independently of a simulator."""

from abc import ABC, abstractmethod

import numpy as np

from hybrid_control.robots.state import CenterOfMassState, JointPositionCommand, JointState, Pose


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
