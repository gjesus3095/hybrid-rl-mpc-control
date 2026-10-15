"""Randomized world-frame end-effector reference trajectories (pure torch, no Isaac imports).

Each episode, every env draws one of three kinds (probabilities ``p_fixed``, ``p_sine``, ``p_quintic``):

- ``FIXED``: a fixed point.
- ``SINE``: a smooth Lissajous-style trajectory ``ref(t) = centre + amplitude * sin(2 pi freq t + phase)`` (per axis),
  with the orientation oscillating the same way about a down-pointing centre orientation (drawn like quintic A/B):
  ``q(t) = exp(theta(t)) q_centre`` with ``theta_i(t) = ori_amplitude_i * sin(2 pi ori_freq_i t + ori_phase_i)`` about
  the world x, y, z axes (own amplitudes, frequencies and phases).
- ``QUINTIC``: rest-to-rest quintic pose segments spawn pose -> A -> B (``segment_time``), then B is held. Position
  moves along the straight segment between two poses and orientation rotates about the fixed world axis of their
  relative rotation, as in the MuJoCo ``QuinticTrajectory`` (``hybrid_control.utils.trajectories``).

Fixed points and sines are blended in from the end-effector's initial pose over ``blend_time`` (smoothstep; the
orientation along the shortest rotation), so the arm is not asked to jump at the start. Fixed points hold the
end-effector orientation at reset. Every kind lasts
``duration`` (= the sum of ``segment_time``); later times, e.g. lookahead points past the end, hold the final pose. Positions are env-local
world coordinates (origin = the env origin, where the base spawns). The trajectories are fixed in the world: they
never move with the base.

Fixed points and sines are reachable by the arm from the spawn pose, so the QP can track them without any base motion:
all points stay within ``reach`` of the shoulder, clear of the robot's own body (``min_radius``) and above the
platform (``min_z``). Candidates are drawn by rejection until they fit; each failed draw shrinks the amplitude of the
next one for that env.

Quintic poses A and B are drawn uniformly from the ball of radius ``outer_radius`` around the arm base
(``panda_link0`` at spawn), outside the vertical cylinder of radius ``inner_radius_xy`` around the arm base's z axis
(a circle in x-y) and above ``base_height + height_clearance``, with the TCP pointing down up to ``tilt_max_deg`` and a
random yaw about the TCP axis. Because each segment is a straight line, both segments are checked exactly: the outer
ball and the height limit are convex (checking the endpoints is enough) and the cylinder is checked with the point of
the segment's x-y projection closest to the axis. Quaternion signs are aligned along spawn -> A -> B, so the
reference quaternion is continuous across the segment boundary. A failing segment is redrawn (A until spawn -> A is
valid, then B until A -> B is valid), so every quintic env gets a valid trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from hybrid_control.utils.torch_rotations import quat_conj, quat_from_rotvec, quat_mul, quat_to_rotvec

FIXED, SINE, QUINTIC = 0, 1, 2
KIND_NAMES = ("fixed point", "sinusoidal", "quintic")  # display names, indexed by kind
KINDS = {"fixed": FIXED, "sine": SINE, "quintic": QUINTIC}  # command-line names
Q_DOWN = (1.0, 0.0, 0.0, 0.0)  # xyzw: rotation by pi about world x, the TCP z axis points along -world z


@dataclass
class TrajectoryCfg:
    """``difficulty`` in [0, 1] scales the sine amplitude and top frequency (quintic trajectories ignore it)."""

    # --- kind mix (normalized) ---
    p_fixed: float = 0.2  # probability of a fixed point (zero amplitude)
    p_sine: float = 0.4
    p_quintic: float = 0.4
    # --- fixed points and sines ---
    shoulder: tuple[float, float, float] = (0.30, 0.0, 0.613)  # arm shoulder at spawn: mount x, link0 top + 0.333 [m]
    reach: float = 0.78  # max distance of any reference point from the shoulder [m] (Panda: 0.855 to the flange)
    min_radius: float = 0.30  # min distance from the shoulder, keeps the target clear of the robot's body [m]
    min_z: float = 0.35  # min height, above the platform top (0.28 m) [m]
    azimuth_deg: float = 150.0  # centre direction: +-azimuth around the forward (+x) axis; 180 = all around
    elevation_deg: tuple[float, float] = (-35.0, 50.0)
    reach_bias: float = 1.3  # > 1 draws the centre distance toward ``reach`` (extended arm, the hard cases)
    amplitude: tuple[float, float, float] = (0.28, 0.28, 0.18)  # max half-extent per axis at difficulty 1 [m]
    frequency: tuple[float, float] = (0.10, 0.45)  # [Hz]; the top end is reached at difficulty 1 (also orientation)
    ori_amplitude_deg: float = 20.0  # max sine orientation amplitude per world axis at difficulty 1 [deg]
    blend_time: float = 1.5  # [s] smoothstep from the initial end-effector position onto the reference
    difficulty: float = 1.0
    # --- quintic spawn -> A -> B ---
    arm_base: tuple[float, float, float] = (0.30, 0.0, 0.28)  # panda_link0 at spawn: (MOUNT_X, 0, STOCK_HEIGHT) [m]
    inner_radius_xy: float = 0.20  # every point stays at least this far from the arm base's z axis (x-y distance) [m]
    outer_radius: float = 0.80  # ... and at most this far from the arm base (3D distance) [m]
    base_height: float = 0.28  # mobile base top plate height (STOCK_HEIGHT) [m]
    height_clearance: float = 0.10  # every point stays this far above the base height [m]
    segment_time: tuple[float, float] = (2.5, 2.5)  # [s] durations of spawn -> A and A -> B
    max_draws: int = 1000  # rejection rounds before giving up with an error (only for an impossible config/spawn)
    tilt_max_deg: float = 30.0  # max angle between the TCP z axis and straight down
    yaw_centre_deg: float = 0.0  # yaw about the TCP z axis, relative to Q_DOWN ...
    yaw_range_deg: float = 90.0  # ... drawn uniformly in yaw_centre +- yaw_range

    @property
    def quintic_min_z(self) -> float:
        return self.base_height + self.height_clearance

    @property
    def duration(self) -> float:
        """Length of every trajectory [s] (= the episode length); afterwards the final pose is held."""
        return float(sum(self.segment_time))


def quintic_time_scaling(t: torch.Tensor, duration: torch.Tensor | float):
    """s(t), ds/dt and d²s/dt² of a rest-to-rest quintic from 0 to 1 (batched torch port of
    ``hybrid_control.utils.trajectories.quintic_time_scaling``). Times outside [0, duration] hold the nearest end."""
    tau = (t / duration).clamp(0.0, 1.0)
    s = tau**3 * (10 - 15 * tau + 6 * tau**2)
    ds = 30 * tau**2 * (1 - tau) ** 2 / duration
    dds = 60 * tau * (1 - tau) * (1 - 2 * tau) / duration**2
    return s, ds, dds


class ReferenceTrajectories:
    # per-env state tensors (e.g. to copy a scenario from one env to another)
    STATE = ("kind", "centre", "amplitude", "frequency", "phase", "start", "start_quat", "centre_quat", "ori_amplitude",
             "ori_frequency", "ori_phase", "waypoints", "waypoint_quat", "segment_rotvec")

    def __init__(self, num_envs: int, device: str | torch.device, cfg: TrajectoryCfg | None = None):
        self.cfg, self.device, self.num_envs = cfg or TrajectoryCfg(), device, num_envs
        z = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        self.centre, self.amplitude, self.frequency, self.phase, self.start = (z(num_envs, 3) for _ in range(5))
        self.kind = torch.full((num_envs,), FIXED, dtype=torch.long, device=device)
        self.start_quat = z(num_envs, 4)
        self.start_quat[:, 3] = 1.0
        self.waypoints = z(num_envs, 3, 3)  # spawn, A, B positions (quintic envs)
        self.waypoint_quat = z(num_envs, 3, 4)  # spawn, A, B orientations (xyzw)
        self.waypoint_quat[..., 3] = 1.0
        self.segment_rotvec = z(num_envs, 2, 3)  # world-frame rotation vector of each segment
        self.centre_quat = z(num_envs, 4)  # sine orientation centre (= start_quat for the other kinds)
        self.centre_quat[:, 3] = 1.0
        self.ori_amplitude, self.ori_frequency, self.ori_phase = z(num_envs, 3), z(num_envs, 3), z(num_envs, 3)

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
        """(n,) bool: whether every point of the raw sine over ``duration`` is within the reachable region."""
        c = self.cfg
        t = torch.linspace(0.0, c.duration, 160, device=self.device)[None, :, None]
        p = centre[:, None] + amplitude[:, None] * torch.sin(2 * math.pi * frequency[:, None] * t + phase[:, None])
        d = (p - torch.tensor(c.shoulder, device=self.device)).norm(dim=-1)
        ok = (d <= c.reach) & (d >= c.min_radius) & (p[..., 2] >= c.min_z)
        return ok.all(dim=1)

    def _sample_sine(self, moving: torch.Tensor):
        """Reachable sines for ``moving`` envs and fixed points (zero amplitude) for the others; failed draws are
        redrawn with a new centre and a smaller amplitude."""
        c, n = self.cfg, len(moving)
        d = float(min(max(c.difficulty, 0.0), 1.0))
        f_top = c.frequency[0] + (c.frequency[1] - c.frequency[0]) * d
        amp_max = torch.tensor(c.amplitude, device=self.device) * d

        centre = torch.zeros(n, 3, device=self.device)
        amp = torch.zeros(n, 3, device=self.device)
        freq = self._uniform(c.frequency[0], f_top, n, 3)
        phase = self._uniform(0.0, 2 * math.pi, n, 3)
        pending = torch.ones(n, dtype=torch.bool, device=self.device)
        shrink = torch.ones(n, device=self.device)
        for _ in range(c.max_draws):
            idx = pending.nonzero(as_tuple=False).flatten()
            if not len(idx):
                return centre, amp, freq, phase
            cand_c = self._sample_centre(len(idx))
            cand_a = torch.rand(len(idx), 3, device=self.device) * amp_max * (moving[idx] * shrink[idx])[:, None]
            ok = self.reachable(cand_c, cand_a, freq[idx], phase[idx])
            centre[idx[ok]], amp[idx[ok]] = cand_c[ok], cand_a[ok]
            pending[idx[ok]] = False
            shrink[idx[~ok]] *= 0.93  # keep the centre search going but ask for smaller motions
        raise RuntimeError(f"no reachable sine after {c.max_draws} draws; check the TrajectoryCfg reach limits")

    # ------------------------------------------------------------------------------------------
    def segment_valid(self, p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        """(n,) bool: whether every point of the straight segments p0 -> p1 (n, 3) is inside the outer ball around
        ``arm_base``, outside the vertical cylinder of radius ``inner_radius_xy`` around its z axis and at least
        ``quintic_min_z`` high. Exact, no sampling."""
        c = self.cfg
        centre = torch.tensor(c.arm_base, device=p0.device)
        d = (p1 - p0)[:, :2]  # the cylinder test only needs the x-y projection
        length2 = (d * d).sum(-1)
        u = ((centre[:2] - p0[:, :2]) * d).sum(-1) / length2.clamp_min(1e-12)
        u = torch.where(length2 > 1e-12, u.clamp(0.0, 1.0), torch.zeros_like(u))
        closest_xy = p0[:, :2] + u[:, None] * d
        inner = (closest_xy - centre[:2]).norm(dim=-1) >= c.inner_radius_xy
        outer = ((p0 - centre).norm(dim=-1) <= c.outer_radius) & ((p1 - centre).norm(dim=-1) <= c.outer_radius)
        high = (p0[:, 2] >= c.quintic_min_z) & (p1[:, 2] >= c.quintic_min_z)
        return inner & outer & high

    def _sample_point(self, n: int) -> torch.Tensor:
        """(n, 3) points uniform in the ball around ``arm_base``, outside the ``inner_radius_xy`` cylinder and above
        ``quintic_min_z`` (rejection sampling)."""
        c = self.cfg
        out = torch.zeros(n, 3, device=self.device)
        need = torch.ones(n, dtype=torch.bool, device=self.device)
        centre = torch.tensor(c.arm_base, device=self.device)
        for _ in range(1000):
            idx = need.nonzero(as_tuple=False).flatten()
            if not len(idx):
                return out
            direction = torch.randn(len(idx), 3, device=self.device)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            r = c.outer_radius * torch.rand(len(idx), device=self.device) ** (1.0 / 3.0)
            cand = centre + r[:, None] * direction
            ok = (cand[:, 2] >= c.quintic_min_z) & ((cand[:, :2] - centre[:2]).norm(dim=-1) >= c.inner_radius_xy)
            out[idx[ok]] = cand[ok]
            need[idx[ok]] = False
        raise RuntimeError("no point of the ball lies outside the cylinder and above quintic_min_z; check the TrajectoryCfg")

    def _sample_quat(self, n: int) -> torch.Tensor:
        """(n, 4) TCP orientations: Q_DOWN, tilted by up to ``tilt_max_deg`` about a random axis in the TCP xy plane,
        then yawed about the TCP z axis."""
        c = self.cfg
        z = torch.zeros(n, device=self.device)
        yaw = torch.deg2rad(c.yaw_centre_deg + c.yaw_range_deg * (2 * torch.rand(n, device=self.device) - 1))
        tilt = math.radians(c.tilt_max_deg) * torch.rand(n, device=self.device)
        axis = 2 * math.pi * torch.rand(n, device=self.device)
        q_tilt = quat_from_rotvec(torch.stack([tilt * torch.cos(axis), tilt * torch.sin(axis), z], -1))
        q_yaw = quat_from_rotvec(torch.stack([z, z, yaw], -1))
        q_down = torch.tensor(Q_DOWN, device=self.device).expand(n, 4)
        return quat_mul(quat_mul(q_down, q_tilt), q_yaw)  # body-frame rotations: the yaw keeps the tilted z axis

    def _sample_endpoints(self, p0: torch.Tensor) -> torch.Tensor:
        """(m, 3) endpoints p1 such that every segment p0 -> p1 is valid; failing segments are redrawn."""
        p1 = torch.zeros_like(p0)
        pending = torch.ones(len(p0), dtype=torch.bool, device=self.device)
        for _ in range(self.cfg.max_draws):
            idx = pending.nonzero(as_tuple=False).flatten()
            if not len(idx):
                return p1
            cand = self._sample_point(len(idx))
            ok = self.segment_valid(p0[idx], cand)
            p1[idx[ok]] = cand[ok]
            pending[idx[ok]] = False
        raise RuntimeError(f"no valid quintic segment from {p0[pending][:3].tolist()} after {self.cfg.max_draws} "
                           "draws: the start lies outside the shell or below quintic_min_z (check the spawn pose / TrajectoryCfg)")

    def _sample_quintic(self, start: torch.Tensor, start_quat: torch.Tensor):
        """Waypoints (m, 3, 3) spawn, A, B with valid segments and their orientations (m, 3, 4)."""
        m = len(start)
        a = self._sample_endpoints(start)
        b = self._sample_endpoints(a)
        quats = torch.cat([start_quat[:, None], self._sample_quat(2 * m).reshape(m, 2, 4)], dim=1)
        # q and -q are the same rotation. Flip A to the hemisphere of the spawn quaternion and B to that of A: then
        # exp(rotvec(q1 q0^-1)) q0 = q1 exactly (the shortest-rotation rotvec needs a non-negative scalar part
        # q1 . q0), so each segment ends on the quaternion the next one starts from.
        for k in (1, 2):
            sign = torch.where((quats[:, k] * quats[:, k - 1]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
            quats[:, k] = sign * quats[:, k]
        return torch.stack([start, a, b], dim=1), quats

    # ------------------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor, start: torch.Tensor, start_quat: torch.Tensor | None = None,
              kind: torch.Tensor | None = None):
        """Draw new trajectories for ``env_ids``; ``start`` (len(env_ids), 3) and ``start_quat`` (len(env_ids), 4,
        xyzw; identity if omitted) are the initial end-effector pose. ``kind`` (len(env_ids),) forces the trajectory
        kinds (``FIXED``/``SINE``/``QUINTIC``) instead of drawing them with ``p_fixed``/``p_sine``/``p_quintic``."""
        c, n = self.cfg, len(env_ids)
        start = start.to(self.device, torch.float32)
        if start_quat is None:
            start_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(n, 4)
        start_quat = start_quat.to(self.device, torch.float32)
        if kind is None:
            probs = torch.tensor([c.p_fixed, c.p_sine, c.p_quintic], device=self.device, dtype=torch.float32)
            kind = torch.multinomial(probs / probs.sum(), n, replacement=True)
        else:
            kind = torch.as_tensor(kind, dtype=torch.long, device=self.device).expand(n).clone()

        centre, amp, freq, phase = self._sample_sine(kind == SINE)
        points = start[:, None].repeat(1, 3, 1)
        quats = start_quat[:, None].repeat(1, 3, 1)
        q_idx = (kind == QUINTIC).nonzero(as_tuple=False).flatten()
        if len(q_idx):
            q_points, q_quats = self._sample_quintic(start[q_idx], start_quat[q_idx])
            points[q_idx], quats[q_idx] = q_points, q_quats
            centre[q_idx] = 0.5 * (q_points[:, 1] + q_points[:, 2])  # centre of the A -> B segment
            amp[q_idx] = 0.0
        rotvec = quat_to_rotvec(quat_mul(quats[:, 1:], quat_conj(quats[:, :-1])))  # q_goal q_start^-1, shortest

        # sine orientation: a down-cone centre (same hemisphere as the start quaternion) and per-axis oscillations
        d = float(min(max(c.difficulty, 0.0), 1.0))
        f_top = c.frequency[0] + (c.frequency[1] - c.frequency[0]) * d
        centre_quat = start_quat.clone()
        ori_amp = torch.zeros(n, 3, device=self.device)
        s_idx = (kind == SINE).nonzero(as_tuple=False).flatten()
        if len(s_idx):
            q_c = self._sample_quat(len(s_idx))
            q_c = torch.where((q_c * start_quat[s_idx]).sum(-1, keepdim=True) < 0, -q_c, q_c)
            centre_quat[s_idx] = q_c
            ori_amp[s_idx] = math.radians(c.ori_amplitude_deg) * d * torch.rand(len(s_idx), 3, device=self.device)
        ori_freq = self._uniform(c.frequency[0], f_top, n, 3)
        ori_phase = self._uniform(0.0, 2 * math.pi, n, 3)

        self.kind[env_ids], self.centre[env_ids], self.amplitude[env_ids] = kind, centre, amp
        self.frequency[env_ids], self.phase[env_ids] = freq, phase
        self.start[env_ids], self.start_quat[env_ids] = start, start_quat
        self.waypoints[env_ids], self.waypoint_quat[env_ids], self.segment_rotvec[env_ids] = points, quats, rotvec
        self.centre_quat[env_ids], self.ori_amplitude[env_ids] = centre_quat, ori_amp
        self.ori_frequency[env_ids], self.ori_phase[env_ids] = ori_freq, ori_phase

    # ------------------------------------------------------------------------------------------
    def kind_name(self, i: int) -> str:
        """Display name of env ``i``'s trajectory kind."""
        return KIND_NAMES[int(self.kind[i])]

    def phase_name(self, i: int, t: float) -> str:
        """Display name of the phase env ``i`` is in at episode time ``t`` [s]: quintic ``spawn -> A``, ``A -> B``,
        ``holding B``; fixed points and sines ``spawn -> point`` / ``spawn -> sine`` while blending in, then
        ``holding point`` / ``following sine``."""
        kind = int(self.kind[i])
        if kind == QUINTIC:
            t0 = self.cfg.segment_time[0]
            return "spawn -> A" if t < t0 else ("A -> B" if t < self.cfg.duration else "holding B")
        target = "point" if kind == FIXED else "sine"
        if t < self.cfg.blend_time:
            return f"spawn -> {target}"
        return "holding point" if kind == FIXED else "following sine"

    def position(self, t: torch.Tensor) -> torch.Tensor:
        """Reference position (N, 3) at episode time ``t`` (N,) [s]."""
        return self.lookahead(t, torch.zeros(1, device=self.device))[:, 0]

    def pose(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference position (N, 3) and orientation (N, 4, xyzw) at episode time ``t`` (N,) [s]."""
        pos, quat = self.lookahead_pose(t, torch.zeros(1, device=self.device))
        return pos[:, 0], quat[:, 0]

    def lookahead(self, t: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Reference positions (N, K, 3) at times ``t + offsets`` (K,)."""
        return self.lookahead_pose(t, offsets)[0]

    def lookahead_pose(self, t: torch.Tensor, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference positions (N, K, 3) and orientations (N, K, 4, xyzw) at times ``t + offsets`` (K,). Times past
        ``duration`` hold the final pose, as when tracking a trajectory that has ended."""
        tt = (t[:, None] + offsets[None, :]).clamp(max=self.cfg.duration)  # (N, K)
        # fixed points and sines: blend in from the start pose (fixed points have zero amplitudes and
        # centre_quat = start_quat, so they hold the start orientation)
        raw = self.centre[:, None] + self.amplitude[:, None] * torch.sin(
            2 * math.pi * self.frequency[:, None] * tt[..., None] + self.phase[:, None]
        )
        theta = self.ori_amplitude[:, None] * torch.sin(
            2 * math.pi * self.ori_frequency[:, None] * tt[..., None] + self.ori_phase[:, None]
        )
        raw_quat = quat_mul(quat_from_rotvec(theta), self.centre_quat[:, None].expand(-1, tt.shape[1], -1))
        s = (tt / max(self.cfg.blend_time, 1e-6)).clamp(0.0, 1.0)
        s = (s * s * (3.0 - 2.0 * s))[..., None]
        pos = (1.0 - s) * self.start[:, None] + s * raw
        start_quat = self.start_quat[:, None].expand_as(raw_quat)
        blend = quat_to_rotvec(quat_mul(raw_quat, quat_conj(start_quat)))  # shortest rotation start -> raw
        quat = quat_mul(quat_from_rotvec(s * blend), start_quat)

        quintic = self.kind == QUINTIC
        if quintic.any():
            t0, t1 = self.cfg.segment_time
            first = (tt < t0)[..., None]  # (N, K, 1)
            s, _, _ = quintic_time_scaling(torch.where(first[..., 0], tt, tt - t0), torch.where(first[..., 0], t0, t1))
            s = s[..., None]
            p0 = torch.where(first, self.waypoints[:, None, 0], self.waypoints[:, None, 1])
            p1 = torch.where(first, self.waypoints[:, None, 1], self.waypoints[:, None, 2])
            q0 = torch.where(first, self.waypoint_quat[:, None, 0], self.waypoint_quat[:, None, 1])
            r = torch.where(first, self.segment_rotvec[:, None, 0], self.segment_rotvec[:, None, 1])
            q_pos = p0 + s * (p1 - p0)
            q_quat = quat_mul(quat_from_rotvec(s * r), q0)
            sel = quintic[:, None, None]
            pos, quat = torch.where(sel, q_pos, pos), torch.where(sel, q_quat, quat)
        return pos, quat
