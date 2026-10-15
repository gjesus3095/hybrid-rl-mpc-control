"""Bounded cubic B-splines in joint space; pure Torch, no simulator imports.

Joint order is (roll [rad], pitch [rad], lift [m]). Each environment
owns an RNG stream, so resetting one slot cannot change another slot's samples.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TrajectoryCfg:
    knot_interval: float = 1.0
    max_velocity: tuple[float, float, float] = (2.0, 2.0, 0.4)
    min_peak_velocity: tuple[float, float, float] = (0.9, 0.9, 0.18)  # per-episode floor, not at reversals
    max_acceleration: tuple[float, float, float] = (6.0, 6.0, 1.2)
    difficulty: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.knot_interval) or self.knot_interval <= 0:
            raise ValueError('knot_interval must be positive and finite')
        if not 0 <= self.difficulty <= 1:
            raise ValueError('difficulty must be in [0, 1]')
        for values in (self.max_velocity, self.max_acceleration):
            if len(values) != 3 or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError('derivative limits must contain three positive finite values')

        if len(self.min_peak_velocity) != 3 or any(not math.isfinite(x) or x < 0 for x in self.min_peak_velocity):
            raise ValueError('min_peak_velocity must contain three non-negative finite values')
        if any(lo >= hi for lo, hi in zip(self.min_peak_velocity, self.max_velocity)):
            raise ValueError('min_peak_velocity must be strictly below max_velocity')


class JointSplines:
    def __init__(self, num_envs, device, duration, mechanism, cfg=None, seed=0):
        self.cfg = cfg or TrajectoryCfg()
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('duration must be positive and finite')
        self.device, self.duration = torch.device(device), duration
        segments = max(4, math.ceil(duration / self.cfg.knot_interval))
        grid = torch.linspace(0, duration, segments + 1, device=self.device)
        self.knots = torch.cat((grid[:1].repeat(3), grid, grid[-1:].repeat(3)))
        self.n_control = segments + 3
        self.control = torch.zeros(num_envs, self.n_control, 3, device=self.device)
        self.lower = torch.tensor(mechanism.lower, device=self.device)
        self.upper = torch.tensor(mechanism.upper, device=self.device)
        self.initial = torch.tensor(mechanism.initial, device=self.device)
        self.generators = [torch.Generator(device=self.device).manual_seed(seed + 1000003 * i) for i in range(num_envs)]
        self.reset(torch.arange(num_envs, device=self.device))

    @staticmethod
    def _derivative(control, knots, degree):
        denom = knots[degree + 1:-1] - knots[1:-degree - 1]
        return degree * (control[:, 1:] - control[:, :-1]) / denom[None, :, None], knots[1:-1]

    def reset(self, ids):
        # A velocity floor at every instant is incompatible with smooth reversals.
        # Instead require each joint to reach a minimum speed during its episode.
        times = torch.linspace(0, self.duration, max(65, 8*self.n_control), device=self.device)
        floor = torch.tensor(self.cfg.min_peak_velocity, device=self.device) * self.cfg.difficulty
        for i in ids.tolist():
            selected = self.initial.expand(self.n_control, -1).clone()
            pending = torch.ones(3, dtype=torch.bool, device=self.device)
            for _ in range(128):
                u = torch.rand(self.n_control, 3, generator=self.generators[i], device=self.device)
                p = self.lower + u * (self.upper - self.lower)
                p[:3] = self.initial
                p[-3:] = p[-1].clone()
                p = self.initial + self.cfg.difficulty * (p - self.initial)
                v, vk = self._derivative(p[None], self.knots, 3)
                a, _ = self._derivative(v, vk, 2)
                v_bound, a_bound = v.abs().amax((0, 1)), a.abs().amax((0, 1))
                scale = torch.minimum(torch.tensor(self.cfg.max_velocity, device=self.device) / v_bound.clamp_min(1e-12),
                                      torch.tensor(self.cfg.max_acceleration, device=self.device) / a_bound.clamp_min(1e-12)).clamp_max(1)
                candidate = self.initial + scale * (p - self.initial)
                peaks = self._evaluate(v.expand(len(times), -1, -1), vk, 2, times).abs().amax(0) * scale
                accepted = pending & (peaks >= floor)
                selected[:, accepted] = candidate[:, accepted]
                pending &= ~accepted
                if not pending.any():
                    break
            if pending.any():
                raise ValueError('Cannot meet min_peak_velocity within position/acceleration bounds; '
                                 'lower the speed floor or increase acceleration/episode duration')
            self.control[i] = selected

    @staticmethod
    def _evaluate(control, knots, degree, t):
        """Batched de Boor evaluation, including the closed right endpoint."""
        n = control.shape[1]
        span = (torch.searchsorted(knots, t.contiguous(), right=True) - 1).clamp(degree, n - 1)
        indices = span[:, None] - degree + torch.arange(degree + 1, device=t.device)
        d = control.gather(1, indices[..., None].expand(-1, -1, 3)).clone()
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                i = span - degree + j
                alpha = ((t - knots[i]) / (knots[i + degree - r + 1] - knots[i])).unsqueeze(-1)
                d[:, j] = (1 - alpha) * d[:, j - 1] + alpha * d[:, j]
        return d[:, degree]

    def evaluate(self, time):
        """Return position, velocity, acceleration, each (N, 3); time is (N,) seconds.

        Outside [0, duration], hold the nearest endpoint with zero derivatives.
        """
        t = time.to(self.device).clamp(0, self.duration)
        v, vk = self._derivative(self.control, self.knots, 3)
        a, ak = self._derivative(v, vk, 2)
        q = self._evaluate(self.control, self.knots, 3, t)
        qd = self._evaluate(v, vk, 2, t)
        qdd = self._evaluate(a, ak, 1, t)
        outside = ((time <= 0) | (time >= self.duration)).to(self.device)
        qd[outside], qdd[outside] = 0, 0
        return q, qd, qdd
