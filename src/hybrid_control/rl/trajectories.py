"""Randomized world-frame end-effector reference trajectories (pure torch, no Isaac imports).

Each episode, every env gets either a fixed point or a smooth Lissajous-style trajectory

    ref(t) = centre + amplitude * sin(2 pi freq t + phase)        (per axis)

blended in from the end-effector's initial position over ``blend_time`` (smoothstep), so the arm is not asked to
jump at the start. Positions are env-local world coordinates (origin = the env origin, where the base spawns). The
trajectories are fixed in the world: they never move with the base.

Every trajectory is reachable by the arm from the spawn pose, so the QP can track it without any base motion: all
points stay within ``reach`` of the shoulder, clear of the robot's own body (``min_radius``) and above the platform
(``min_z``). Candidates are drawn by rejection; a trajectory that cannot be made to fit has its amplitude shrunk, and
finally becomes a fixed point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TrajectoryCfg:
    """``difficulty`` in [0, 1] scales amplitude and the top frequency."""

    shoulder: tuple[float, float, float] = (0.30, 0.0, 0.613)  # arm shoulder at spawn: mount x, link0 top + 0.333 [m]
    reach: float = 0.78  # max distance of any reference point from the shoulder [m] (Panda: 0.855 to the flange)
    min_radius: float = 0.30  # min distance from the shoulder, keeps the target clear of the robot's body [m]
    min_z: float = 0.35  # min height, above the platform top (0.28 m) [m]
    azimuth_deg: float = 150.0  # centre direction: +-azimuth around the forward (+x) axis; 180 = all around
    elevation_deg: tuple[float, float] = (-35.0, 50.0)
    reach_bias: float = 1.3  # > 1 draws the centre distance toward ``reach`` (extended arm, the hard cases)
    amplitude: tuple[float, float, float] = (0.28, 0.28, 0.18)  # max half-extent per axis at difficulty 1 [m]
    frequency: tuple[float, float] = (0.10, 0.45)  # [Hz]; the top end is reached at difficulty 1
    p_fixed: float = 0.25  # probability of a fixed point (zero amplitude)
    blend_time: float = 1.5  # [s] smoothstep from the initial end-effector position onto the reference
    horizon: float = 20.0  # [s] reachability is checked over this time span
    max_tries: int = 40
    difficulty: float = 1.0


class ReferenceTrajectories:
    def __init__(self, num_envs: int, device: str | torch.device, cfg: TrajectoryCfg | None = None):
        self.cfg, self.device, self.num_envs = cfg or TrajectoryCfg(), device, num_envs
        z = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        self.centre, self.amplitude, self.frequency, self.phase, self.start = (z(num_envs, 3) for _ in range(5))

    # ------------------------------------------------------------------------------------------
    def _uniform(self, low, high, *shape):
        return low + (high - low) * torch.rand(*shape, device=self.device)

    def _sample_centre(self, n: int) -> torch.Tensor:
        c = self.cfg
        theta = math.radians(c.azimuth_deg) * (2 * torch.rand(n, device=self.device) - 1)
        phi = torch.deg2rad(self._uniform(c.elevation_deg[0], c.elevation_deg[1], n))
        r = c.min_radius + (c.reach - c.min_radius) * torch.rand(n, device=self.device) ** (1.0 / c.reach_bias)
        direction = torch.stack([torch.cos(phi) * torch.cos(theta), torch.cos(phi) * torch.sin(theta), torch.sin(phi)], -1)
        return torch.tensor(c.shoulder, device=self.device) + r[:, None] * direction

    def reachable(self, centre, amplitude, frequency, phase) -> torch.Tensor:
        """(n,) bool: whether every point of the raw trajectory over ``horizon`` is within the reachable region."""
        c = self.cfg
        t = torch.linspace(0.0, c.horizon, 160, device=self.device)[None, :, None]
        p = centre[:, None] + amplitude[:, None] * torch.sin(2 * math.pi * frequency[:, None] * t + phase[:, None])
        d = (p - torch.tensor(c.shoulder, device=self.device)).norm(dim=-1)
        ok = (d <= c.reach) & (d >= c.min_radius) & (p[..., 2] >= c.min_z)
        return ok.all(dim=1)

    def reset(self, env_ids: torch.Tensor, start: torch.Tensor):
        """Draw new reachable trajectories for ``env_ids``; ``start`` (len(env_ids), 3) is the initial EE position."""
        c, n = self.cfg, len(env_ids)
        d = float(min(max(c.difficulty, 0.0), 1.0))
        f_top = c.frequency[0] + (c.frequency[1] - c.frequency[0]) * d
        amp_max = torch.tensor(c.amplitude, device=self.device) * d

        centre = torch.zeros(n, 3, device=self.device)
        amp = torch.zeros(n, 3, device=self.device)
        freq = self._uniform(c.frequency[0], f_top, n, 3)
        phase = self._uniform(0.0, 2 * math.pi, n, 3)
        pending = torch.ones(n, dtype=torch.bool, device=self.device)
        moving = torch.rand(n, device=self.device) >= c.p_fixed
        shrink = torch.ones(n, device=self.device)
        for _ in range(c.max_tries):
            idx = pending.nonzero(as_tuple=False).flatten()
            if not len(idx):
                break
            cand_c = self._sample_centre(len(idx))
            cand_a = torch.rand(len(idx), 3, device=self.device) * amp_max * (moving[idx] * shrink[idx])[:, None]
            ok = self.reachable(cand_c, cand_a, freq[idx], phase[idx])
            centre[idx[ok]], amp[idx[ok]] = cand_c[ok], cand_a[ok]
            pending[idx[ok]] = False
            shrink[idx[~ok]] *= 0.93  # keep the centre search going but ask for smaller motions
        if pending.any():  # fall back to a fixed point at a safe spot straight ahead
            idx = pending.nonzero(as_tuple=False).flatten()
            centre[idx] = torch.tensor(c.shoulder, device=self.device) + torch.tensor([0.45, 0.0, 0.0], device=self.device)
            amp[idx] = 0.0

        self.centre[env_ids], self.amplitude[env_ids] = centre, amp
        self.frequency[env_ids], self.phase[env_ids] = freq, phase
        self.start[env_ids] = start

    # ------------------------------------------------------------------------------------------
    def position(self, t: torch.Tensor) -> torch.Tensor:
        """Reference position (N, 3) at episode time ``t`` (N,) [s]."""
        return self.lookahead(t, torch.zeros(1, device=self.device))[:, 0]

    def lookahead(self, t: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Reference positions (N, K, 3) at times ``t + offsets`` (K,)."""
        tt = t[:, None] + offsets[None, :]  # (N, K)
        raw = self.centre[:, None] + self.amplitude[:, None] * torch.sin(
            2 * math.pi * self.frequency[:, None] * tt[..., None] + self.phase[:, None]
        )
        s = (tt / max(self.cfg.blend_time, 1e-6)).clamp(0.0, 1.0)
        s = (s * s * (3.0 - 2.0 * s))[..., None]
        return (1.0 - s) * self.start[:, None] + s * raw
