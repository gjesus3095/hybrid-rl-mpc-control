"""Time-parameterized Cartesian reference trajectories for task-space controllers.

Each trajectory returns a ``CartesianSample``: a world-frame pose plus the twist
velocity and acceleration (6,), ordered [linear, angular] like ``RobotModel``
Jacobian rows. Units are metres, radians and seconds.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from hybrid_control.robots.state import Pose


@dataclass(frozen=True)
class CartesianSample:
    """Reference pose with world twist velocity (m/s, rad/s) and acceleration (m/s², rad/s²)."""

    pose: Pose
    velocity: np.ndarray
    acceleration: np.ndarray


def quintic_time_scaling(t: float, duration: float) -> tuple[float, float, float]:
    """Return s(t), ds/dt and d²s/dt² for a rest-to-rest quintic from 0 to 1.

    s = 10τ³ - 15τ⁴ + 6τ⁵ with τ = t/duration, so velocity and acceleration are
    zero at both ends. Times outside [0, duration] hold the nearest endpoint.
    """
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    tau = float(np.clip(t / duration, 0.0, 1.0))
    s = tau**3 * (10 - 15 * tau + 6 * tau**2)
    ds = 30 * tau**2 * (1 - tau) ** 2 / duration
    dds = 60 * tau * (1 - tau) * (1 - 2 * tau) / duration**2
    return s, ds, dds


class QuinticTrajectory:
    """Rest-to-rest pose interpolation between two poses over ``duration`` seconds.

    Position moves along the straight line between the poses. Orientation rotates
    about the fixed world axis of the relative rotation R_goal R_start^T, so the
    angular velocity and acceleration are that rotation vector scaled by ds and dds.
    """

    def __init__(self, start: Pose, goal: Pose, duration: float):
        quintic_time_scaling(0.0, duration)  # Validate duration.
        self.start = start
        self.goal = goal
        self.duration = float(duration)
        self._start_rotation = Rotation.from_quat(np.array(start.quaternion))
        relative = Rotation.from_quat(np.array(goal.quaternion)) * self._start_rotation.inv()
        self._rotation_vector = relative.as_rotvec()  # Shortest rotation, world frame.
        self._displacement = goal.position - start.position

    def sample(self, t: float) -> CartesianSample:
        """Return the reference at time t (s) since the start of the segment."""
        s, ds, dds = quintic_time_scaling(t, self.duration)
        rotation = Rotation.from_rotvec(s * self._rotation_vector) * self._start_rotation
        pose = Pose(self.start.position + s * self._displacement, rotation.as_quat())
        velocity = np.hstack([ds * self._displacement, ds * self._rotation_vector])
        acceleration = np.hstack([dds * self._displacement, dds * self._rotation_vector])
        return CartesianSample(pose, velocity, acceleration)


class SinusoidalTrajectory:
    """Per-axis sinusoidal position about a center, holding the center orientation.

    p(t) = center + amplitude * sin(2π f t + phase), elementwise on world X, Y, Z.
    With zero phase the motion starts at the center with nonzero velocity; use a
    phase of -π/2 and an offset center to start at rest instead.
    """

    def __init__(self, center: Pose, amplitude, frequency_hz, phase=0.0):
        self.center = center
        self.amplitude = np.broadcast_to(np.asarray(amplitude, dtype=float), (3,)).copy()
        self.omega = 2 * np.pi * np.broadcast_to(np.asarray(frequency_hz, dtype=float), (3,)).copy()
        self.phase = np.broadcast_to(np.asarray(phase, dtype=float), (3,)).copy()
        for name, value in (("amplitude", self.amplitude), ("frequency_hz", self.omega),
                            ("phase", self.phase)):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if np.any(self.omega < 0):
            raise ValueError("frequency_hz must be nonnegative")

    def sample(self, t: float) -> CartesianSample:
        """Return the reference at time t (s)."""
        angle = self.omega * t + self.phase
        position = self.center.position + self.amplitude * np.sin(angle)
        linear_velocity = self.amplitude * self.omega * np.cos(angle)
        linear_acceleration = -self.amplitude * self.omega**2 * np.sin(angle)
        zeros = np.zeros(3)
        return CartesianSample(
            Pose(position, self.center.quaternion),
            np.hstack([linear_velocity, zeros]),
            np.hstack([linear_acceleration, zeros]),
        )
