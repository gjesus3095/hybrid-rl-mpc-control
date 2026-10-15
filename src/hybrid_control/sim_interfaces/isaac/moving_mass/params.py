"""Moving-mass geometry and episode dynamics, independent of Isaac Lab."""
from dataclasses import dataclass, field, replace
import math
import random

from hybrid_control.sim_interfaces.isaac.ridgeback.params import (
    BaseParams as RidgebackParams,
    DomainRandomization as RidgebackRandomization,
)



@dataclass(frozen=True)
class MechanismCfg:
    mast_length: float = 0.5
    platform_height: float = 0.28
    lower: tuple[float, float, float] = (-math.pi/2, -math.pi/2, 0.0)
    upper: tuple[float, float, float] = (math.pi/2, math.pi/2, 0.5)
    initial: tuple[float, float, float] = (0.0, 0.0, 0.2)
    stiffness: tuple[float, float, float] = (1000.0, 1000.0, 10000.0)
    damping: tuple[float, float, float] = (100.0, 100.0, 500.0)
    effort_limit: tuple[float, float, float] = (150.0, 150.0, 500.0)
    velocity_limit: tuple[float, float, float] = (3.0, 3.0, 0.6)

    def __post_init__(self):
        for name in ('lower', 'upper', 'initial', 'stiffness', 'damping', 'effort_limit', 'velocity_limit'):
            if len(getattr(self, name)) != 3 or not all(math.isfinite(x) for x in getattr(self, name)):
                raise ValueError(f'{name} must contain three finite values')
        if any(not lo <= q <= hi or lo >= hi for lo, q, hi in zip(self.lower, self.initial, self.upper)):
            raise ValueError('initial joints must lie within ordered limits')
        if self.lower[2] < 0:
            raise ValueError('lift travel must be non-negative')
        if any(x <= 0 for name in ('stiffness', 'damping', 'effort_limit', 'velocity_limit') for x in getattr(self, name)):
            raise ValueError('actuator parameters must be positive')
        if not all(math.isfinite(x) and x > 0 for x in (self.platform_height, self.mast_length)):
            raise ValueError('mechanism dimensions and masses must be positive and finite')


@dataclass(frozen=True)
class MovingMassParams:
    footprint: tuple[float, float] = RidgebackParams.footprint
    base_mass: float = RidgebackParams.base_mass
    base_com_z: float = RidgebackParams.base_com_z
    terminal_mass: float = RidgebackParams.payload
    carriage_mass: float = 0.5
    gimbal_mass: float = 0.25
    mast_mass: float = 0.25

    def __post_init__(self):
        if len(self.footprint) != 2 or not all(math.isfinite(x) and x > 0 for x in (*self.footprint, self.base_mass, self.terminal_mass, self.carriage_mass, self.gimbal_mass, self.mast_mass)):
            raise ValueError('footprint and masses must be positive and finite')
        if not math.isfinite(self.base_com_z):
            raise ValueError('base_com_z must be finite')

    def with_dynamics(self, other):
        return replace(other, footprint=self.footprint)


@dataclass(frozen=True)
class PlatformRandomization:
    """Defaults follow ridgeback.params; overrides remain local to this configuration."""
    length: tuple[float, float] = RidgebackRandomization.length
    width: tuple[float, float] = RidgebackRandomization.width
    base_mass: tuple[float, float] = RidgebackRandomization.base_mass
    base_com_z: tuple[float, float] = RidgebackRandomization.base_com_z
    hard_bias: float = RidgebackRandomization.hard_bias

    def __post_init__(self):
        for name in ('length', 'width', 'base_mass', 'base_com_z'):
            lo, hi = getattr(self, name)
            if not all(math.isfinite(x) for x in (lo, hi)) or lo > hi:
                raise ValueError(f'{name} must be an ordered finite range')
            if name != 'base_com_z' and lo <= 0:
                raise ValueError(f'{name} must be positive')
        if not math.isfinite(self.hard_bias) or self.hard_bias <= 0:
            raise ValueError('hard_bias must be positive and finite')

    def sample(self, rng):
        def low(bounds):
            return bounds[0] + (bounds[1]-bounds[0]) * rng.random()**self.hard_bias
        return (low(self.length), low(self.width)), low(self.base_mass), rng.uniform(*self.base_com_z)


@dataclass(frozen=True)
class MovingMassRandomization:
    platform: PlatformRandomization = field(default_factory=PlatformRandomization)
    terminal_mass: tuple[float, float] = (2.0, 10.0)
    carriage_mass: tuple[float, float] = (0.25, 1.0)
    gimbal_mass: tuple[float, float] = (0.125, 0.5)
    mast_mass: tuple[float, float] = (0.125, 0.5)

    def __post_init__(self):
        for name in ('terminal_mass', 'carriage_mass', 'gimbal_mass', 'mast_mass'):
            lo, hi = getattr(self, name)
            if not (0 < lo <= hi < math.inf):
                raise ValueError(f'{name} must be a positive ordered range')

    def sample(self, rng: random.Random) -> MovingMassParams:
        footprint, mass, height = self.platform.sample(rng)
        # Match Ridgeback's payload distribution: bias toward the heavier end.
        lo, hi = self.terminal_mass
        terminal = hi - (hi - lo) * rng.random()**self.platform.hard_bias
        return MovingMassParams(footprint, mass, height, terminal,
                                rng.uniform(*self.carriage_mass), rng.uniform(*self.gimbal_mass),
                                rng.uniform(*self.mast_mass))

    def sample_many(self, n: int, seed: int = 0):
        rng = random.Random(seed)
        return [self.sample(rng) for _ in range(n)]
