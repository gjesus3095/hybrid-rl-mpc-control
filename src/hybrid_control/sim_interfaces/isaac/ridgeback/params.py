"""Per-environment physical parameters of the free-base Ridgeback-Franka and their randomization.

Pure Python (no Isaac imports), so it can be used and tested without starting Kit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

VIDEO_LIGHT = 1200.0  # distant light intensity of the video scripts (RidgebackDREnv default: 3000)


@dataclass(frozen=True)
class BaseParams:
    """Parameters that differ between environments.

    footprint: platform length (x) and width (y) [m]; the stock Ridgeback is 0.93 x 0.79. Fixed at spawn.
    base_mass: mass of ``base_link`` [kg]; the stock value is 165.
    base_com_z: height of the platform COM above the base frame [m].
    payload: mass added at the gripper tool centre point [kg] (Panda rating: 3).
    """

    footprint: tuple[float, float] = (0.40, 0.40)
    base_mass: float = 30.0
    base_com_z: float = 0.10
    payload: float = 3.0

    def __post_init__(self):
        if min(self.footprint) <= 0.0:
            raise ValueError(f"footprint must be positive, got {self.footprint}")
        if self.base_mass <= 0.0:
            raise ValueError(f"base_mass must be positive, got {self.base_mass}")
        if self.payload < 0.0:
            raise ValueError(f"payload must be non-negative, got {self.payload}")

    def with_dynamics(self, other: BaseParams) -> BaseParams:
        """Copy of ``self`` with the dynamics of ``other`` (mass, COM height, payload) but the geometry of ``self``.

        The footprint is baked into the USD collision geometry at spawn, so it cannot change at a reset.
        """
        return replace(other, footprint=self.footprint)


@dataclass(frozen=True)
class DomainRandomization:
    """Sampling ranges ``(low, high)`` for ``BaseParams``, skewed toward the hard end by ``hard_bias``.

    Hard = small, light platform and heavy payload. With ``hard_bias`` b > 1 the platform size and mass are drawn as
    ``low + (high - low) * u**b`` and the payload as ``high - (high - low) * u**b`` (u uniform in [0, 1]), so easy
    (large, heavy) platforms are rare; b = 1 is uniform. The platform COM height is fixed by default.
    """

    length: tuple[float, float] = (0.19, 0.52)
    width: tuple[float, float] = (0.19, 0.52)
    base_mass: tuple[float, float] = (5.5, 38.0)
    base_com_z: tuple[float, float] = (0.10, 0.10)
    payload: tuple[float, float] = (0.0, 3.0)  # 0 = no payload (MassModel leaves the hand unchanged)
    hard_bias: float = 2.2

    def _low(self, rng: random.Random, bounds: tuple[float, float]) -> float:
        return bounds[0] + (bounds[1] - bounds[0]) * rng.random() ** self.hard_bias

    def _high(self, rng: random.Random, bounds: tuple[float, float]) -> float:
        return bounds[1] - (bounds[1] - bounds[0]) * rng.random() ** self.hard_bias

    def sample(self, rng: random.Random) -> BaseParams:
        return BaseParams(
            footprint=(self._low(rng, self.length), self._low(rng, self.width)),
            base_mass=self._low(rng, self.base_mass),
            base_com_z=rng.uniform(*self.base_com_z),
            payload=self._high(rng, self.payload),
        )

    def sample_many(self, n: int, seed: int | None = None) -> list[BaseParams]:
        rng = random.Random(seed)
        return [self.sample(rng) for _ in range(n)]
