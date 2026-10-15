"""Reference path that writes a text with the TCP on a vertical wall plane (pure numpy / scipy, no Isaac imports).

The pen plane is either ``plane="wall"``: the vertical plane ``x = x_wall`` (env-local, in front of the robot, which faces
+x) with the TCP pointing at it (its z axis along +x), or ``plane="table"``: the horizontal plane ``z = z_table`` with the
TCP pointing down and the letters' "up" along +x (away from the robot). The text reads left to right as seen from the
robot's side, i.e. it runs toward -y; the first letter is centred in front of the robot. Every letter is one pen-down stroke of a simple stroke font. A stroke is followed at
constant speed with a trapezoidal ramp from and to rest, and its corners are rounded with a small fillet. Between strokes
the TCP lifts ``lift`` metres off the wall, hops to the next letter and lowers again, each as a rest-to-rest quintic, so
the arm stops between strokes. The path starts at the spawn TCP pose (turning the TCP to face the wall), and ends by
parking beyond the last letter so the robot does not hide the text.

``build_text_path`` returns the pose table (sampled every ``REF_DT`` like the benchmark scenarios, playable by
``ScriptedTrajectories``), a pen-down mask and the letter / phase of every sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .scenarios import REF_DT, SPAWN_POSES, rounded_polyline, trapezoid_progress

Q_WALL = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))  # xyzw: rotation by +90 degrees about y, the TCP z axis points along +x
Q_DOWN = (1.0, 0.0, 0.0, 0.0)  # xyzw: rotation by pi about x, the TCP z axis points down (the orientation the policy trained on)
PHASES = ("entry", "lower", "write", "lift", "hop", "park", "hold")  # names of the phase codes of ``TextPath.phase``

# Stroke font in a unit box (height 1, width <= 0.6, v up, u along the reading direction). A glyph is one stroke: a list of
# ("pt", u, v) corner vertices and ("arc", cx, cy, rx, ry, start_deg, end_deg) elliptical arcs (counter-clockwise for
# increasing angles) that continue from the previous point.
GLYPHS: dict[str, list[tuple]] = {
    "M": [("pt", 0, 0), ("pt", 0, 1), ("pt", 0.3, 0.4), ("pt", 0.6, 1), ("pt", 0.6, 0)],
    "P": [("pt", 0, 0), ("pt", 0, 1), ("pt", 0.33, 1), ("arc", 0.33, 0.75, 0.27, 0.25, 90, -90), ("pt", 0, 0.5)],
    "C": [("arc", 0.3, 0.5, 0.3, 0.5, 45, 315)],
    "R": [("pt", 0, 0), ("pt", 0, 1), ("pt", 0.33, 1), ("arc", 0.33, 0.75, 0.27, 0.25, 90, -90), ("pt", 0, 0.5), ("pt", 0.6, 0)],
    "L": [("pt", 0, 1), ("pt", 0, 0), ("pt", 0.5, 0)],
    "2": [("arc", 0.3, 0.72, 0.3, 0.28, 160, -45), ("pt", 0, 0), ("pt", 0.6, 0)],
    "0": [("arc", 0.3, 0.5, 0.3, 0.5, 90, -270)],
    "6": [("arc", 0.3, 0.5, 0.3, 0.5, 62, 270), ("arc", 0.3, 0.3, 0.3, 0.3, 270, 630)],
}


@dataclass(frozen=True)
class TextCfg:
    controller: str = "impedance"  # fixes the TCP spawn pose the path starts from
    text: str = "MPCRL2026"
    plane: str = "wall"  # "wall" (vertical, TCP faces it) or "table" (horizontal, TCP points down)
    x_wall: float = 0.85  # wall plane [m, env-local]: 0.55 m ahead of the arm mount
    z_table: float = 0.75  # table plane height [m]: a sweep found 0.65-0.75 best (lower folds the elbow against its limit)
    x_table: float = 0.58  # x of the middle of the letters on the table [m], where the TCP spawns
    lift: float = 0.05  # pen-up distance from the wall [m]
    height: float = 0.25  # letter height [m]
    pitch: float = 0.20  # distance between letter origins [m]
    z_centre: float = 0.65  # height of the middle of the letters [m] (the shoulder is at 0.61 m)
    y_first: float = 0.0  # y of the middle of the first letter [m]
    write_speed: float = 0.15  # pen-down speed [m/s]
    write_accel: float = 0.3  # start / stop acceleration of a stroke [m/s^2]
    corner_radius: float = 0.02  # fillet radius of the corners inside a stroke [m]
    sample_spacing: float = 0.005  # spacing of the points sampled along arcs [m]
    lower_time: float = 1.0  # time to lift / lower the pen [s]
    hop_speed: float = 0.25  # mean speed of the pen-up hops [m/s]
    hop_min_time: float = 1.5  # shortest hop [s]
    entry_min_time: float = 4.0  # shortest approach from the spawn pose, which also turns the TCP to the wall [s]
    park_distance: float = 0.9  # the TCP parks this far beyond the end of the last letter [m]
    park_standoff: float = 0.35  # ... and this far in front of the wall plane [m]
    park_time: float = 4.0
    end_hold: float = 1.0  # [s]


@dataclass
class TextPath:
    cfg: TextCfg
    poses: np.ndarray  # (T, 7) float32: env-local position + xyzw quaternion every REF_DT
    pen_down: np.ndarray  # (T,) bool: the stroke of a letter is being written
    letter: np.ndarray  # (T,) int: index of the letter being written or hopped to (-1 for entry / park / hold)
    phase: np.ndarray  # (T,) int index into ``PHASES``
    strokes: list[dict] = field(default_factory=list)  # per letter: char, sample range, dense (y, z) points of the stroke

    @property
    def duration(self) -> float:
        return (len(self.poses) - 1) * REF_DT

    @property
    def board(self) -> tuple[float, float, float, float]:
        """Centre y, centre of the other in-plane axis (z for the wall, x for the table), and the board's extent along y and
        along that other axis [m]."""
        c = self.cfg
        ys = np.concatenate([s["yz"][:, 0] for s in self.strokes])
        return (float(0.5 * (ys.min() + ys.max())), c.z_centre if c.plane == "wall" else c.x_table,
                float(ys.max() - ys.min()) + 0.3, c.height + 0.2)


def glyph_points(char: str, spacing: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Unit-box vertices (K, 2) of the stroke of ``char`` and a bool (K,) marking the corner vertices (``spacing`` in unit
    coordinates between the points sampled along arcs)."""
    pts, corner = [], []
    for item in GLYPHS[char]:
        if item[0] == "pt":
            pts.append((item[1], item[2]))
            corner.append(True)
        else:
            _, cx, cy, rx, ry, a0, a1 = item
            angles = np.radians(np.linspace(a0, a1, max(int(math.ceil(abs(math.radians(a1 - a0)) * max(rx, ry) / spacing)), 2) + 1))
            arc = np.stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)], axis=-1)
            for p in arc[1:] if pts else arc:  # the arc continues from the previous point
                pts.append(tuple(p))
                corner.append(False)
    return np.array(pts), np.array(corner)


def stroke_yz(char: str, y_centre: float, cfg: TextCfg) -> np.ndarray:
    """Dense (K, 2) wall-plane points (y, z) of the stroke of ``char`` whose box is centred on ``y_centre``, with its
    corners rounded; 1 mm apart."""
    uv, corner = glyph_points(char, cfg.sample_spacing / cfg.height)
    width = uv[:, 0].max()
    # reading direction is -y: u grows toward -y; centre the glyph box on y_centre
    y = y_centre + (width / 2 - uv[:, 0]) * cfg.height
    z = (cfg.z_centre if cfg.plane == "wall" else cfg.x_table) + (uv[:, 1] - 0.5) * cfg.height  # z (wall) or x (table)
    vertices = np.stack([np.zeros(len(y)), y, z], axis=-1)
    radius = np.where(corner, cfg.corner_radius, 0.0)
    radius[0] = radius[-1] = 0.0
    dense = rounded_polyline(vertices, radius)
    return dense[:, 1:]


def _quintic_move(p0, q0, p1, q1, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Rest-to-rest quintic pose move sampled every REF_DT (end excluded): positions (n, 3) and xyzw quaternions (n, 4)."""
    n = max(int(round(duration / REF_DT)), 1)
    tau = np.arange(n) / n
    s = (10 * tau**3 - 15 * tau**4 + 6 * tau**5)[:, None]
    r0 = Rotation.from_quat(q0)
    rotvec = (Rotation.from_quat(q1) * r0.inv()).as_rotvec()  # shortest rotation
    quat = (Rotation.from_rotvec(s * rotvec) * r0).as_quat()
    return np.asarray(p0) + s * (np.asarray(p1) - np.asarray(p0)), quat


def build_text_path(cfg: TextCfg | None = None) -> TextPath:
    cfg = cfg or TextCfg()
    spawn_pos, spawn_quat = SPAWN_POSES[cfg.controller]
    chars = list(cfg.text)
    unknown = sorted(set(chars) - set(GLYPHS))
    if unknown:
        raise ValueError(f"no glyph for {unknown}; available: {sorted(GLYPHS)}")

    if cfg.plane not in ("wall", "table"):
        raise ValueError(f"plane must be 'wall' or 'table', got {cfg.plane!r}")
    wall_plane = cfg.plane == "wall"
    pos, quat, pen, letter, phase, strokes = [], [], [], [], [], []

    def add(p, q, is_pen, k, code):
        pos.append(np.asarray(p)), quat.append(np.asarray(q))
        pen.append(np.full(len(p), is_pen)), letter.append(np.full(len(p), k)), phase.append(np.full(len(p), PHASES.index(code)))

    def to3(ab, lifted=False):
        """3D point(s) of in-plane coordinates ``ab`` (..., 2) = (y, z for the wall or x for the table), pen up if ``lifted``."""
        ab = np.asarray(ab, dtype=float)
        if wall_plane:
            plane = np.full(ab.shape[:-1] + (1,), cfg.x_wall - (cfg.lift if lifted else 0.0))
            return np.concatenate([plane, ab], axis=-1)
        plane = np.full(ab.shape[:-1] + (1,), cfg.z_table + (cfg.lift if lifted else 0.0))
        return np.concatenate([ab[..., 1:2], ab[..., 0:1], plane], axis=-1)

    hover = lambda ab: to3(ab, True)  # noqa: E731
    wall = lambda ab: to3(ab, False)  # noqa: E731
    q_wall, here_pos, here_quat = np.array(Q_WALL if wall_plane else Q_DOWN), np.array(spawn_pos), np.array(spawn_quat)
    for k, char in enumerate(chars):
        yz = stroke_yz(char, cfg.y_first - k * cfg.pitch, cfg)
        # approach: from the spawn pose (first letter) or hop from above the previous stroke end
        target = hover(yz[0])
        dist = float(np.linalg.norm(target - here_pos))
        if k == 0:
            move = _quintic_move(here_pos, here_quat, target, q_wall, max(cfg.entry_min_time, dist / cfg.hop_speed))
            add(*move, False, -1, "entry")
        else:
            move = _quintic_move(here_pos, q_wall, target, q_wall, max(cfg.hop_min_time, dist / cfg.hop_speed))
            add(*move, False, k, "hop")
        # lower, write, lift
        move = _quintic_move(target, q_wall, wall(yz[0]), q_wall, cfg.lower_time)
        add(*move, False, k, "lower")
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(yz, axis=0), axis=-1))])
        _, motion = trapezoid_progress(np.zeros(1), arc[-1], cfg.write_speed, cfg.write_accel)
        t = np.arange(max(int(round(motion / REF_DT)), 1)) * REF_DT
        s, _ = trapezoid_progress(t, arc[-1], cfg.write_speed, cfg.write_accel)
        yz_t = np.stack([np.interp(s, arc, yz[:, 0]), np.interp(s, arc, yz[:, 1])], axis=-1)
        start = sum(len(p) for p in pos)
        add(to3(yz_t), np.tile(q_wall, (len(t), 1)), True, k, "write")
        strokes.append({"char": char, "start": start, "stop": start + len(t), "yz": yz})
        move = _quintic_move(wall(yz[-1]), q_wall, hover(yz[-1]), q_wall, cfg.lower_time)
        add(*move, False, k, "lift")
        here_pos, here_quat = hover(yz[-1]), q_wall

    # park beyond the end of the text, in front of the wall, then hold
    y_end = min(s["yz"][:, 0].min() for s in strokes)
    if wall_plane:
        park = np.array([cfg.x_wall - cfg.park_standoff, y_end - cfg.park_distance, cfg.z_centre])
    else:
        park = np.array([cfg.x_table, y_end - cfg.park_distance, cfg.z_table + 0.15])
    add(*_quintic_move(here_pos, q_wall, park, q_wall, cfg.park_time), False, -1, "park")
    hold = max(int(round(cfg.end_hold / REF_DT)), 1)
    add(np.tile(park, (hold, 1)), np.tile(q_wall, (hold, 1)), False, -1, "hold")

    pos, quat = np.concatenate(pos), np.concatenate(quat)
    # the last pose once more so the table ends exactly on the park pose
    pos, quat = np.concatenate([pos, pos[-1:]]), np.concatenate([quat, quat[-1:]])
    flag = lambda a, v: np.concatenate([np.concatenate(a), [v]])  # noqa: E731
    return TextPath(cfg, np.concatenate([pos, quat], axis=1).astype(np.float32), flag(pen, False).astype(bool),
                    flag(letter, -1).astype(int), flag(phase, PHASES.index("hold")).astype(int), strokes)
