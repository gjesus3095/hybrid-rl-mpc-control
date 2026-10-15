"""Deterministic evaluation benchmark for the base policy (pure torch / numpy, no Isaac imports).

``generate_benchmark(cfg)`` always returns the same scenarios for the same ``BenchmarkCfg`` (every random draw is seeded),
so different policies can be compared on exactly the same references and dynamics. A scenario is a dense world-frame
reference pose table (env-local coordinates, sampled at ``REF_DT``) plus the base dynamics it is played with. Five sets of
``n_per_set`` scenarios each (slot ``j`` of every set has footprint ``j``, so sets are paired across the same platforms):

- ``fixed``: fixed points, drawn exactly like training (``ReferenceTrajectories``).
- ``sine``: Lissajous sines at full difficulty, drawn like training.
- ``poly``: quintic spawn -> A -> B (2.5 s + 2.5 s), drawn like training.
- ``poly_far``: generalization. Spawn -> A as in training, then A -> B with B outside the training reach: its x-y distance
  from the arm base is in ``[far_min_radius_xy, far_max_radius_xy]`` (0.9-2.0 m; training stays within 0.8 m of the arm
  base) and its height in the training height range, so it is only reachable once the base has driven toward it. A -> B
  takes ``max(2.5 s, L / ref_speed)``, ``ref_speed`` being the mean speed of the training A -> B legs, so the speed
  profile matches training and only the geometry is new. ``hold_time`` of holding B is appended.
- ``rigid``: straight lines at constant speed (trapezoidal speed ramp at start and end, no quintic) from the spawn pose:
  a third are single straight legs, a third have one 90 degree bend and a third two, with rounded corners
  (``corner_radius``). The TCP orientation is held. The whole path stays inside the training workspace (ball around the
  arm base, outside the inner cylinder, above the height limit). ``hold_time`` of holding the end is appended.

- ``ellipse``: loops around the arm base axis: ellipses (circles are the equal-axes case) with random semi-axes and
  orientation, centred on the arm base's x-y, whose height oscillates along the loop (``ellipse_z_*``). Even scenarios
  are full revolutions, odd ones half revolutions, in a random direction from a random start angle. The TCP gets from the
  spawn pose to the start of the loop with a rest-to-rest quintic (timed at ``ref_speed``), then runs the loop at a
  constant tangential speed (``ellipse_speed``, within the training speeds; trapezoidal ramp from and to rest). The
  orientation is held. The path stays inside the training workspace.

The reference starts at the spawn TCP pose of the arm controller (``SPAWN_POSES``, measured in Isaac: the same for every
env; ``BenchmarkCfg.controller`` picks it, the other sets of numbers are identical). The runner refuses to start if the simulated spawn pose differs from the one stored in the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from hybrid_control.rl.policy_with_controller.trajectories import (
    FIXED, QUINTIC, SINE, ReferenceTrajectories, TrajectoryCfg, quintic_time_scaling,
)
from hybrid_control.sim_interfaces.isaac.ridgeback.params import BaseParams, DomainRandomization
from hybrid_control.utils.torch_rotations import quat_conj, quat_from_rotvec, quat_mul, quat_to_rotvec

SPAWN_QUAT = (0.9925040, 0.0220365, -0.1201792, -0.0026683)  # xyzw, pointing down with a small tilt
# TCP at spawn, env-local [m] (probed in Isaac), per arm controller: the IK/QP tracks panda_hand, the impedance QP the
# fingertip centre, which is 0.11 m further along the hand z axis (same orientation)
SPAWN_POSES = {"ik_qp": ((0.6113457, 0.0, 0.6348850), SPAWN_QUAT),
               "impedance": ((0.5845972, 0.0, 0.5260529), SPAWN_QUAT)}
SPAWN_POS = SPAWN_POSES["ik_qp"][0]
REF_DT = 1.0 / 120.0  # sample period of the reference tables [s] (the physics rate)
SET_NAMES = ("fixed", "sine", "poly", "poly_far", "rigid", "ellipse")
RIGID_TYPES = ("straight", "one bend", "two bends")


@dataclass(frozen=True)
class BenchmarkCfg:
    seed: int = 2026
    controller: str = "ik_qp"  # arm controller the references are for: it fixes the TCP (hence the spawn pose)
    n_per_set: int = 100
    trajectory: TrajectoryCfg = field(default_factory=TrajectoryCfg)  # training sampler for fixed / sine / poly
    # --- poly_far ---
    far_min_radius_xy: float = 0.9  # B is at least this far from the arm base in x-y [m] (> the 0.8 m training ball)
    far_max_radius_xy: float = 2.0  # ... and at most this far [m]
    far_z: tuple[float, float] = (0.38, 1.05)  # B height range [m]: the range training B points cover
    ref_speed: float = 0.25  # mean speed of the training A -> B legs [m/s] (measured: 0.245)
    # --- rigid ---
    rigid_leg: tuple[float, float] = (0.3, 0.6)  # leg length of the bent paths [m]
    rigid_straight_leg: tuple[float, float] = (0.4, 0.9)  # length of the single straight leg [m]
    rigid_speed: tuple[float, float] = (0.2, 0.35)  # cruise speed [m/s]
    rigid_accel: float = 0.5  # start / stop acceleration [m/s^2]
    corner_radius: float = 0.05  # fillet radius of the 90 degree corners [m]
    hold_time: float = 1.0  # hold at the end of poly_far, rigid and ellipse [s]
    # --- ellipse ---
    ellipse_semi_axes: tuple[float, float] = (0.30, 0.65)  # both semi-axes are drawn from this range [m]
    ellipse_z_centre: tuple[float, float] = (0.50, 0.75)  # mean height of the loop [m]
    ellipse_z_amp: tuple[float, float] = (0.0, 0.20)  # amplitude of the height oscillation over one revolution [m]
    ellipse_speed: tuple[float, float] = (0.20, 0.35)  # constant tangential speed [m/s]
    max_draws: int = 10000

    @property
    def spawn(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Spawn TCP position and xyzw quaternion of ``controller``."""
        return SPAWN_POSES[self.controller]


@dataclass
class ScenarioSet:
    name: str
    poses: np.ndarray  # (S, T, 7) float32: position + xyzw quaternion every REF_DT; the last pose is held past the end
    duration: np.ndarray  # (S,) float32 [s], length of each scenario (the episode time to evaluate)
    centre: np.ndarray  # (S, 3) float32 reference centre (only used for the env's centre-distance reward term)
    dynamics: np.ndarray  # (S, 3) float32 base_mass [kg], base_com_z [m], payload [kg]
    info: dict[str, np.ndarray] = field(default_factory=dict)  # kind-specific waypoints / parameters

    def __len__(self) -> int:
        return len(self.poses)


@dataclass
class Benchmark:
    cfg: BenchmarkCfg
    footprints: np.ndarray  # (S, 2) platform length / width of env slot j [m], fixed at spawn
    sets: dict[str, ScenarioSet]

    def params(self, name: str, slots: int | None = None) -> list[BaseParams]:
        """``BaseParams`` of scenarios ``0..slots`` of set ``name`` (footprint of slot j, the scenario's dynamics)."""
        s = self.sets[name]
        return [BaseParams(tuple(map(float, self.footprints[j])), *map(float, s.dynamics[j]))
                for j in range(len(s) if slots is None else slots)]

    def fingerprint(self, names=None) -> str:
        """Short hash of every number in the footprints and the sets ``names`` (default: all): equal fingerprints mean
        identical scenarios. Sets added to the benchmark later do not change the fingerprint of the older ones."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.footprints, dtype=np.float32).tobytes())
        for name in SET_NAMES:
            if name in self.sets and (names is None or name in names):
                s = self.sets[name]
                for a in (s.poses, s.duration, s.centre, s.dynamics):
                    h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        return h.hexdigest()[:12]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"footprints": self.footprints, "meta": np.array(json.dumps(asdict(self.cfg)))}
        for name, s in self.sets.items():
            for key in ("poses", "duration", "centre", "dynamics"):
                arrays[f"{name}/{key}"] = getattr(s, key)
            for key, value in s.info.items():
                arrays[f"{name}/info/{key}"] = value
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Benchmark:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        trajectory = TrajectoryCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta.pop("trajectory").items()})
        cfg = BenchmarkCfg(trajectory=trajectory, **{k: tuple(v) if isinstance(v, list) else v for k, v in meta.items()})
        sets = {}
        for name in SET_NAMES:
            if f"{name}/poses" not in data:
                continue
            prefix = f"{name}/info/"
            info = {k[len(prefix):]: data[k] for k in data.files if k.startswith(prefix)}
            sets[name] = ScenarioSet(name, data[f"{name}/poses"], data[f"{name}/duration"], data[f"{name}/centre"],
                                     data[f"{name}/dynamics"], info)
        return cls(cfg, data["footprints"], sets)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _grid(duration_max: float) -> np.ndarray:
    return np.arange(int(math.ceil(duration_max / REF_DT - 1e-9)) + 1) * REF_DT


def _spawn(cfg: BenchmarkCfg, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    pos, quat = cfg.spawn
    return torch.tensor(pos).expand(n, 3).clone(), torch.tensor(quat).expand(n, 4).clone()


def _poses(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return np.concatenate([pos, quat], axis=-1).astype(np.float32)


def _training_set(cfg: BenchmarkCfg, kind: int, seed: int, n: int) -> dict:
    """fixed / sine / quintic scenarios from the training sampler (global torch RNG, seeded, on the CPU)."""
    torch.manual_seed(seed)
    traj = ReferenceTrajectories(n, "cpu", cfg.trajectory)
    start, start_quat = _spawn(cfg, n)
    traj.reset(torch.arange(n), start, start_quat, kind=torch.full((n,), kind))
    t = torch.tensor(_grid(cfg.trajectory.duration), dtype=torch.float32)
    pos, quat = traj.lookahead_pose(torch.zeros(n), t)
    info = {"centre": traj.centre}
    if kind == SINE:
        info.update(amplitude=traj.amplitude, frequency=traj.frequency, phase=traj.phase)
    if kind == QUINTIC:
        info.update(waypoints=traj.waypoints, waypoint_quat=traj.waypoint_quat)
    return dict(poses=_poses(pos.numpy(), quat.numpy()), duration=np.full(n, cfg.trajectory.duration, np.float32),
                centre=traj.centre.numpy(), info={k: v.numpy().astype(np.float32) for k, v in info.items()})


def quintic_chain(points: torch.Tensor, quats: torch.Tensor, seg_time: torch.Tensor, t: torch.Tensor):
    """Rest-to-rest quintic pose segments through ``points`` (n, 3, 3) / ``quats`` (n, 3, 4): spawn -> A (``seg_time[:, 0]``)
    -> B (``seg_time[:, 1]``), sampled at times ``t`` (T,). Returns positions (n, T, 3) and quaternions (n, T, 4); the last
    pose is held. Same maths as ``ReferenceTrajectories`` with per-scenario segment durations."""
    n = len(points)
    rotvec = quat_to_rotvec(quat_mul(quats[:, 1:], quat_conj(quats[:, :-1])))  # (n, 2, 3), shortest rotation
    tt = t[None, :].expand(n, -1)
    t0 = seg_time[:, :1]
    first = (tt < t0)[..., None]
    s, _, _ = quintic_time_scaling(torch.where(first[..., 0], tt, tt - t0), torch.where(first[..., 0], t0, seg_time[:, 1:]))
    s = s[..., None]
    pick = lambda a, b: torch.where(first, a[:, None], b[:, None])  # noqa: E731
    p0, p1 = pick(points[:, 0], points[:, 1]), pick(points[:, 1], points[:, 2])
    q0, r = pick(quats[:, 0], quats[:, 1]), pick(rotvec[:, 0], rotvec[:, 1])
    return p0 + s * (p1 - p0), quat_mul(quat_from_rotvec(s * r), q0)


def _pad(tables: list[np.ndarray], t_len: int) -> np.ndarray:
    """Stack (T_i, 7) tables, holding the last pose up to ``t_len`` samples."""
    out = np.zeros((len(tables), t_len, 7), np.float32)
    for i, tab in enumerate(tables):
        out[i, : len(tab)] = tab
        out[i, len(tab):] = tab[-1]
    return out


# ----------------------------------------------------------------------------------------------
# poly_far
# ----------------------------------------------------------------------------------------------
def _far_set(cfg: BenchmarkCfg, seed: int, n: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tc = cfg.trajectory
    sampler = ReferenceTrajectories(n, "cpu", tc)
    start, start_quat = _spawn(cfg, n)
    points, quats = sampler._sample_quintic(start, start_quat)  # spawn, A, B (a training B, replaced below) + orientations
    a = points[:, 1]

    # B in the annulus around the arm base, outside the training ball, and the straight A -> B must not cross the robot
    segment_checker = ReferenceTrajectories(n, "cpu", replace(tc, outer_radius=1e3))
    centre_xy = np.array(tc.arm_base[:2])
    b = np.zeros((n, 3), np.float32)
    pending = np.ones(n, bool)
    r0, r1 = cfg.far_min_radius_xy, cfg.far_max_radius_xy
    for _ in range(cfg.max_draws):
        idx = np.flatnonzero(pending)
        if not len(idx):
            break
        r = np.sqrt(rng.uniform(r0**2, r1**2, len(idx)))  # uniform in area
        theta = rng.uniform(0.0, 2 * np.pi, len(idx))
        cand = np.stack([centre_xy[0] + r * np.cos(theta), centre_xy[1] + r * np.sin(theta),
                         rng.uniform(*cfg.far_z, len(idx))], axis=-1).astype(np.float32)
        ok = segment_checker.segment_valid(a[idx], torch.from_numpy(cand)).numpy()
        b[idx[ok]] = cand[ok]
        pending[idx[ok]] = False
    else:
        raise RuntimeError("no valid far B point found; check the BenchmarkCfg radii")
    points = points.clone()
    points[:, 2] = torch.from_numpy(b)

    length = (points[:, 2] - points[:, 1]).norm(dim=-1)
    seg = torch.stack([torch.full((n,), float(tc.segment_time[0])),
                       torch.clamp(length / cfg.ref_speed, min=float(tc.segment_time[1]))], dim=-1)
    duration = seg.sum(-1) + cfg.hold_time
    t = torch.tensor(_grid(float(duration.max())), dtype=torch.float32)
    pos, quat = quintic_chain(points, quats, seg, t)
    centre = 0.5 * (points[:, 1] + points[:, 2])
    return dict(poses=_poses(pos.numpy(), quat.numpy()), duration=duration.numpy().astype(np.float32),
                centre=centre.numpy(),
                info={"waypoints": points.numpy(), "waypoint_quat": quats.numpy(), "segment_time": seg.numpy(),
                      "ab_length": length.numpy()})


# ----------------------------------------------------------------------------------------------
# rigid paths
# ----------------------------------------------------------------------------------------------
def rounded_polyline(vertices: np.ndarray, radius, spacing: float = 1e-3) -> np.ndarray:
    """Dense points (K, 3), ``spacing`` apart, along the polyline through ``vertices`` (m, 3) with each interior corner
    replaced by a circular fillet of ``radius``: straight lines between the fillets, tangent-continuous. ``radius`` is a
    float or one value per vertex; vertices with radius 0 (or a straight continuation) are passed through unrounded."""
    radii = np.broadcast_to(np.asarray(radius, dtype=float), (len(vertices),))
    pieces, cursor = [], vertices[0]
    for k in range(1, len(vertices) - 1):
        d_in = vertices[k] - vertices[k - 1]
        d_in /= np.linalg.norm(d_in)
        d_out = vertices[k + 1] - vertices[k]
        d_out /= np.linalg.norm(d_out)
        turn = math.acos(float(np.clip(d_in @ d_out, -1.0, 1.0)))
        if radii[k] <= 0.0 or turn < 1e-6:
            pieces.append(_line(cursor, vertices[k], spacing))
            cursor = vertices[k]
            continue
        tangent = radii[k] * math.tan(turn / 2.0)
        t1 = vertices[k] - d_in * tangent
        normal = d_out - (d_out @ d_in) * d_in
        normal /= np.linalg.norm(normal)
        centre = t1 + normal * radii[k]
        pieces.append(_line(cursor, t1, spacing))
        angles = np.linspace(0.0, turn, max(int(math.ceil(radii[k] * turn / spacing)), 2) + 1)
        pieces.append(centre + radii[k] * (-normal * np.cos(angles)[:, None] + d_in * np.sin(angles)[:, None]))
        cursor = vertices[k] + d_out * tangent
    pieces.append(_line(cursor, vertices[-1], spacing))
    dense = np.concatenate(pieces)
    return dense[np.r_[True, np.linalg.norm(np.diff(dense, axis=0), axis=-1) > 1e-9]]  # drop duplicated joints


def _line(a: np.ndarray, b: np.ndarray, spacing: float) -> np.ndarray:
    return a + (b - a) * np.linspace(0.0, 1.0, max(int(math.ceil(np.linalg.norm(b - a) / spacing)), 1) + 1)[:, None]


def trapezoid_progress(t: np.ndarray, length: float, speed: float, accel: float) -> tuple[np.ndarray, float]:
    """Path length covered at times ``t`` and the motion time: accelerate at ``accel`` to ``speed``, cruise, decelerate."""
    ramp_t, ramp_s = speed / accel, speed**2 / (2 * accel)
    if length < 2 * ramp_s:
        raise ValueError(f"path of {length:.3f} m is too short to reach {speed} m/s at {accel} m/s^2")
    cruise_t = (length - 2 * ramp_s) / speed
    total = 2 * ramp_t + cruise_t
    s = np.where(t < ramp_t, 0.5 * accel * t**2,
                 np.where(t < ramp_t + cruise_t, ramp_s + speed * (t - ramp_t), length - 0.5 * accel * np.clip(total - t, 0, None) ** 2))
    return np.where(t >= total, length, s), total


def _path_valid(points: np.ndarray, cfg: TrajectoryCfg) -> bool:
    """Whether every point is inside the training workspace: within ``outer_radius`` of the arm base, at least
    ``inner_radius_xy`` from its axis and above ``quintic_min_z``."""
    base = np.array(cfg.arm_base)
    p = points[::5]
    return bool(((p - base) ** 2).sum(-1).max() <= cfg.outer_radius**2
                and np.linalg.norm(p[:, :2] - base[:2], axis=-1).min() >= cfg.inner_radius_xy
                and p[:, 2].min() >= cfg.quintic_min_z)


def _rigid_set(cfg: BenchmarkCfg, seed: int, n: int) -> dict:
    rng = np.random.default_rng(seed)
    start = np.array(cfg.spawn[0])
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    tables, durations, vertices_all, kinds, speeds = [], [], [], [], []
    for i in range(n):
        n_bends = i % 3  # interleaved, so every type is spread over the platforms
        for _ in range(cfg.max_draws):
            lengths = rng.uniform(*(cfg.rigid_leg if n_bends else cfg.rigid_straight_leg), n_bends + 1)
            direction = unit(rng.normal(size=3))
            vertices = [start, start + direction * lengths[0]]
            for length in lengths[1:]:
                perpendicular = unit(rng.normal(size=3))
                direction = unit(perpendicular - (perpendicular @ direction) * direction)  # a 90 degree turn
                vertices.append(vertices[-1] + direction * length)
            vertices = np.array(vertices)
            dense = rounded_polyline(vertices, cfg.corner_radius)
            if _path_valid(dense, cfg.trajectory):
                break
        else:
            raise RuntimeError("no valid rigid path found; check the BenchmarkCfg lengths")
        speed = float(rng.uniform(*cfg.rigid_speed))
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=-1))])
        _, motion_time = trapezoid_progress(np.zeros(1), arc[-1], speed, cfg.rigid_accel)
        duration = motion_time + cfg.hold_time
        t = _grid(duration)
        s, _ = trapezoid_progress(t, arc[-1], speed, cfg.rigid_accel)
        pos = np.stack([np.interp(s, arc, dense[:, k]) for k in range(3)], axis=-1)
        tables.append(_poses(pos, np.broadcast_to(cfg.spawn[1], (len(t), 4))))
        durations.append(duration)
        vertices_all.append(np.pad(vertices, ((0, 4 - len(vertices)), (0, 0)), constant_values=np.nan))
        kinds.append(n_bends)
        speeds.append(speed)
    t_len = max(len(tab) for tab in tables)
    poses = _pad(tables, t_len)
    centre = np.array([np.nanmean(v, axis=0) for v in vertices_all], np.float32)
    return dict(poses=poses, duration=np.array(durations, np.float32), centre=centre,
                info={"vertices": np.array(vertices_all, np.float32), "path_type": np.array(kinds, np.int64),
                      "speed": np.array(speeds, np.float32)})


def _ellipse_curve(centre_xy, a, b, yaw, z0, z_amp, z_phase, theta0, sweep, step=2e-3) -> np.ndarray:
    """Dense points (K, 3) of the loop theta0 -> theta0 + sweep (signed): x-y on the ellipse with semi-axes a, b rotated by
    ``yaw`` about ``centre_xy``, height z0 + z_amp sin(theta + z_phase)."""
    theta = theta0 + np.linspace(0.0, sweep, max(int(abs(sweep) / step), 2) + 1)
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = a * np.cos(theta), b * np.sin(theta)
    return np.stack([centre_xy[0] + c * ex - s * ey, centre_xy[1] + s * ex + c * ey,
                     z0 + z_amp * np.sin(theta + z_phase)], axis=-1)


def _ellipse_set(cfg: BenchmarkCfg, seed: int, n: int) -> dict:
    rng = np.random.default_rng(seed)
    tc = cfg.trajectory
    centre_xy = np.array(tc.arm_base[:2])
    start = np.array(cfg.spawn[0])
    tables, durations, shapes, thetas, speeds, entry_times = [], [], [], [], [], []
    for i in range(n):
        sweep = 2 * math.pi if i % 2 == 0 else math.pi  # alternating full and half revolutions
        for _ in range(cfg.max_draws):
            a, b = rng.uniform(*cfg.ellipse_semi_axes, 2)
            yaw, z_phase, theta0 = rng.uniform(0.0, math.pi), rng.uniform(0.0, 2 * math.pi), rng.uniform(0.0, 2 * math.pi)
            z0, z_amp = rng.uniform(*cfg.ellipse_z_centre), rng.uniform(*cfg.ellipse_z_amp)
            signed_sweep = sweep * rng.choice([-1.0, 1.0])
            curve = _ellipse_curve(centre_xy, a, b, yaw, z0, z_amp, z_phase, theta0, signed_sweep)
            if _path_valid(np.concatenate([_line(start, curve[0], 1e-3), curve]), tc):  # the entry line too
                break
        else:
            raise RuntimeError("no valid ellipse found; check the BenchmarkCfg ellipse ranges")
        speed = float(rng.uniform(*cfg.ellipse_speed))
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=-1))])
        _, loop_time = trapezoid_progress(np.zeros(1), arc[-1], speed, cfg.rigid_accel)
        entry_time = max(float(tc.segment_time[0]), float(np.linalg.norm(curve[0] - start)) / cfg.ref_speed)
        duration = entry_time + loop_time + cfg.hold_time
        t = _grid(duration)
        tau = np.clip(t / entry_time, 0.0, 1.0)
        entry = start + (10 * tau**3 - 15 * tau**4 + 6 * tau**5)[:, None] * (curve[0] - start)
        s_loop, _ = trapezoid_progress(np.maximum(t - entry_time, 0.0), arc[-1], speed, cfg.rigid_accel)
        loop = np.stack([np.interp(s_loop, arc, curve[:, k]) for k in range(3)], axis=-1)
        pos = np.where((t < entry_time)[:, None], entry, loop)
        tables.append(_poses(pos, np.broadcast_to(cfg.spawn[1], (len(t), 4))))
        durations.append(duration)
        shapes.append([a, b, yaw, z0, z_amp, z_phase])
        thetas.append([theta0, signed_sweep])
        speeds.append(speed)
        entry_times.append(entry_time)
    poses = _pad(tables, max(len(tab) for tab in tables))
    centre = np.stack([np.array([*centre_xy, z0]) for _, _, _, z0, _, _ in shapes]).astype(np.float32)
    return dict(poses=poses, duration=np.array(durations, np.float32), centre=centre,
                info={"shape": np.array(shapes, np.float32), "theta": np.array(thetas, np.float32),
                      "speed": np.array(speeds, np.float32), "entry_time": np.array(entry_times, np.float32)})


# ----------------------------------------------------------------------------------------------
def generate_benchmark(cfg: BenchmarkCfg | None = None) -> Benchmark:
    """Build the benchmark: the same ``cfg`` always gives the same scenarios (see ``Benchmark.fingerprint``)."""
    cfg = cfg or BenchmarkCfg()
    n, seed = cfg.n_per_set, cfg.seed
    footprints = np.array([p.footprint for p in DomainRandomization().sample_many(n, seed=seed)], np.float32)
    makers = {
        "fixed": lambda s: _training_set(cfg, FIXED, s, n),
        "sine": lambda s: _training_set(cfg, SINE, s, n),
        "poly": lambda s: _training_set(cfg, QUINTIC, s, n),
        "poly_far": lambda s: _far_set(cfg, s, n),
        "rigid": lambda s: _rigid_set(cfg, s, n),
        "ellipse": lambda s: _ellipse_set(cfg, s, n),
    }
    sets = {}
    for k, name in enumerate(SET_NAMES):
        data = makers[name](seed + 100 * (k + 1))
        dyn = DomainRandomization().sample_many(n, seed=seed + 100 * (k + 1) + 1)  # same distribution as training resets
        dynamics = np.array([[p.base_mass, p.base_com_z, p.payload] for p in dyn], np.float32)
        sets[name] = ScenarioSet(name, dynamics=dynamics, **data)
    return Benchmark(cfg, footprints, sets)
