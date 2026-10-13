"""Per-environment physical parameters of the free-base Ridgeback-Franka and their randomization.

Pure Python (no Isaac imports), so it can be used and tested without starting Kit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace


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
    """Uniform sampling ranges ``(low, high)`` for ``BaseParams``.

    The defaults span the cases tested with the tipping demo: from platforms that tip as soon as the arm
    reaches out (small, light) to ones that stay upright over the whole workspace (large, heavy).
    """

    length: tuple[float, float] = (0.25, 0.60)
    width: tuple[float, float] = (0.25, 0.60)
    base_mass: tuple[float, float] = (8.0, 45.0)
    base_com_z: tuple[float, float] = (0.06, 0.14)
    payload: tuple[float, float] = (0.0, 3.0)

    def sample(self, rng: random.Random) -> BaseParams:
        return BaseParams(
            footprint=(rng.uniform(*self.length), rng.uniform(*self.width)),
            base_mass=rng.uniform(*self.base_mass),
            base_com_z=rng.uniform(*self.base_com_z),
            payload=rng.uniform(*self.payload),
        )

    def sample_many(self, n: int, seed: int | None = None) -> list[BaseParams]:
        rng = random.Random(seed)
        return [self.sample(rng) for _ in range(n)]
