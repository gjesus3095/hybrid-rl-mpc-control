"""Common lifecycle for controllers with different target and command types."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from hybrid_control.robots.model import ControlMode, RobotModel

TargetT = TypeVar("TargetT")
CommandT = TypeVar("CommandT")


class Controller(ABC, Generic[TargetT, CommandT]):
    """Compute commands; the caller owns command application and physics stepping."""

    def __init__(self, robot: RobotModel):
        # Validate that the robot's control mode matches the controller's expected control mode
        robot.validate_control_mode(self.control_mode)
        self.robot = robot

    @property
    @abstractmethod
    def control_mode(self) -> ControlMode:
        """Type of actuator reference produced by this controller."""

    @abstractmethod
    def compute(self, target: TargetT, dt: float) -> CommandT:
        """Return a command for a positive interval dt (s), without changing the robot."""

    def reset(self) -> None:
        """Clear controller memory between episodes; stateless controllers do nothing."""
