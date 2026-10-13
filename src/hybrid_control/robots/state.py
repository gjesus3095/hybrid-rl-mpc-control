"""Simulator-independent data for one robot, using NumPy and SI units."""

from dataclasses import dataclass

import numpy as np


def _vector(value, name: str, size: int | None = None) -> np.ndarray:
    """Own a finite, read-only vector rather than aliasing caller/simulator memory."""
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 1 or (size is not None and result.shape != (size,)):
        expected = "a vector" if size is None else f"a vector of length {size}"
        raise ValueError(f"{name} must be {expected}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Pose:
    """World-frame position (3,) in metres and unit quaternion (4,) in xyzw order."""

    position: np.ndarray
    quaternion: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "position", _vector(self.position, "position", 3))
        quaternion = _vector(self.quaternion, "quaternion", 4)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("quaternion must have a finite, nonzero norm")
        object.__setattr__(self, "quaternion", _vector(quaternion / norm, "quaternion", 4))


@dataclass(frozen=True)
class CenterOfMassState:
    """Robot CoM snapshot: simulation time (s) and world position (3,) in metres."""

    time: float
    position: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")
        object.__setattr__(self, "position", _vector(self.position, "CoM position", 3))


@dataclass(frozen=True)
class JointState:
    """Snapshot in model joint order: time (s), positions (n,), velocities (n,).

    Hinge joints use radians and rad/s; slide joints use metres and m/s.
    Optionally stores a matching torque vector for the selected joints.
    """

    time: float
    positions: np.ndarray
    velocities: np.ndarray
    torque: np.ndarray | None = None

    def __post_init__(self):
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")
        positions = _vector(self.positions, "positions")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(
            self, "velocities", _vector(self.velocities, "velocities", len(positions)),
        )
        if self.torque is not None:
            object.__setattr__(self, "torque", _vector(self.torque, "torque", len(positions)))

    @property
    def torques(self) -> np.ndarray | None:
        """Backward-compatible alias for the optional joint torque vector."""
        return self.torque


@dataclass(frozen=True)
class JointPositionCommand:
    """Position references (n,), in the explicitly supplied joint-name order.

    Hinge references are radians; slide references are metres. These are actuator
    targets, not instructions to teleport the simulated joint configuration.
    """

    joint_names: tuple[str, ...]
    positions: np.ndarray

    def __post_init__(self):
        names = tuple(self.joint_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("joint_names must contain nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "positions", _vector(self.positions, "positions", len(names)))


@dataclass(frozen=True)
class JointVelocityCommand:
    """Velocity references (n,), in the explicitly supplied joint-name order.

    Hinge commands use rad/s; slide commands use m/s. These are actuator references
    for instantaneous joint rates and are not intended to teleport the simulation.
    """

    joint_names: tuple[str, ...]
    velocities: np.ndarray

    def __post_init__(self):
        names = tuple(self.joint_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("joint_names must contain nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "velocities", _vector(self.velocities, "velocities", len(names)))


@dataclass(frozen=True)
class JointTorqueCommand:
    """Joint effort commands (n,), in the explicitly supplied joint-name order.

    Hinge commands are torques in N·m; slide commands are forces in N. These are
    applied directly by motor actuators and do not include any servo feedback.
    """

    joint_names: tuple[str, ...]
    torques: np.ndarray

    def __post_init__(self):
        names = tuple(self.joint_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("joint_names must contain nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "torques", _vector(self.torques, "torques", len(names)))
