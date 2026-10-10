"""Common lifecycle for controllers with different target and command types."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


TargetT = TypeVar("TargetT")
CommandT = TypeVar("CommandT")


class Controller(ABC, Generic[TargetT, CommandT]):
    """Compute commands; the caller owns command application and physics stepping."""

    @abstractmethod
    def compute(self, target: TargetT, dt: float) -> CommandT:
        """Return a command for a positive interval dt (s), without changing the robot."""

    def reset(self) -> None:
        """Clear controller memory between episodes; stateless controllers do nothing."""
