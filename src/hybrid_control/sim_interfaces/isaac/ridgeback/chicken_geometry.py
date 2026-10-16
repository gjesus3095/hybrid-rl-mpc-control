"""Geometry of the cartoon chicken skin: primitive meshes, OBJ I/O, the overlay URDF and the procedural neck tube.

Pure numpy (no Isaac imports), so the assets can be generated and tested without starting Kit. The Isaac side lives
in ``chicken.py``; the assets are written by ``scripts/make_chicken_assets.py``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "chicken"
CHICKEN_URDF = ASSET_DIR / "ridgeback_franka_chicken.urdf"


# ---------------------------------------------------------------------------
# Primitive meshes: (vertices (V, 3), vertex normals (V, 3), faces (F, 3) triangles)
# ---------------------------------------------------------------------------
Mesh = tuple[np.ndarray, np.ndarray, np.ndarray]


def _grid_faces(rows: int, cols: int, wrap: bool) -> np.ndarray:
    """Two triangles per quad of a ``rows`` x ``cols`` vertex grid (columns wrap around if ``wrap``)."""
    faces = []
    for i in range(rows - 1):
        for j in range(cols if wrap else cols - 1):
            a, b = i * cols + j, i * cols + (j + 1) % cols
            c, d = a + cols, b + cols
            faces += [(a, c, b), (b, c, d)]
    return np.array(faces, dtype=np.int64)


def uv_sphere(n_lat: int = 24, n_lon: int = 32) -> Mesh:
    """Unit sphere (scale it non-uniformly for ellipsoids)."""
    theta = np.linspace(0.0, np.pi, n_lat + 1)[:, None]  # from +z down to -z
    phi = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)[None, :]
    v = np.stack(np.broadcast_arrays(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), -1)
    v = v.reshape(-1, 3)
    return v, v.copy(), _grid_faces(n_lat + 1, n_lon, wrap=True)


def cone(n: int = 32) -> Mesh:
    """Unit cone along +z: base circle of radius 1 at z = 0 (capped), tip at z = 1."""
    phi = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], -1)
    side_n = np.stack([np.cos(phi), np.sin(phi), np.ones(n)], -1) / np.sqrt(2.0)
    # side: base ring + one tip vertex per segment (sharp tip, smooth sides); cap: own ring + centre
    tip = np.tile([0.0, 0.0, 1.0], (n, 1))
    v = np.concatenate([ring, tip, ring, [[0.0, 0.0, 0.0]]])
    nrm = np.concatenate([side_n, side_n, np.tile([0.0, 0.0, -1.0], (n + 1, 1))])
    j = np.arange(n)
    side = np.stack([j, (j + 1) % n, n + j], -1)
    cap = np.stack([2 * n + (j + 1) % n, 2 * n + j, np.full(n, 3 * n)], -1)
    return v, nrm, np.concatenate([side, cap])


def torus(major: float, minor: float, n_major: int = 48, n_minor: int = 16) -> Mesh:
    """Torus around +z in the xy plane [m]."""
    u = np.linspace(0.0, 2 * np.pi, n_major, endpoint=False)[:, None]
    w = np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False)[None, :]
    nrm = np.stack(np.broadcast_arrays(np.cos(w) * np.cos(u), np.cos(w) * np.sin(u), np.sin(w) + 0 * u), -1)
    centre = np.stack(np.broadcast_arrays(major * np.cos(u), major * np.sin(u), 0 * u + 0 * w), -1)
    v = centre + minor * nrm
    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            a, b = i * n_minor + j, i * n_minor + (j + 1) % n_minor
            c, d = ((i + 1) % n_major) * n_minor + j, ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            faces += [(a, c, b), (b, c, d)]
    return v.reshape(-1, 3), nrm.reshape(-1, 3), np.array(faces, dtype=np.int64)


def write_obj(path: Path, mesh: Mesh):
    v, n, f = mesh
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in n]
    lines += ["f " + " ".join(f"{i + 1}//{i + 1}" for i in face) for face in f]
    Path(path).write_text("\n".join(lines) + "\n")


def read_obj(path: Path) -> Mesh:
    """Read an OBJ written by ``write_obj`` (shared vertex/normal indices, triangles)."""
    v, n, f = [], [], []
    for line in Path(path).read_text().splitlines():
        tag, *rest = line.split()
        if tag == "v":
            v.append([float(x) for x in rest])
        elif tag == "vn":
            n.append([float(x) for x in rest])
        elif tag == "f":
            f.append([int(x.split("/")[0]) - 1 for x in rest])
    return np.array(v), np.array(n), np.array(f, dtype=np.int64)


# ---------------------------------------------------------------------------
# Overlay URDF: per-link visuals (link frame offset, mesh, scale, colour)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChickenVisual:
    name: str
    link: str
    mesh: Path
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    scale: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


def _floats(text: str | None, default: str) -> tuple[float, ...]:
    return tuple(float(x) for x in (text or default).split())


def load_chicken_visuals(urdf: Path = CHICKEN_URDF) -> list[ChickenVisual]:
    """All mesh visuals of the overlay URDF; mesh paths resolve relative to the URDF."""
    root = ET.parse(urdf).getroot()
    colors = {m.get("name"): m.find("color").get("rgba") for m in root.findall("material") if m.find("color") is not None}
    out = []
    for link in root.findall("link"):
        for k, vis in enumerate(link.findall("visual")):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            origin = vis.find("origin")
            material = vis.find("material")
            color = material.find("color") if material is not None else None
            rgba = color.get("rgba") if color is not None else colors[material.get("name")]
            out.append(ChickenVisual(
                name=vis.get("name") or f"{link.get('name')}_{k}",
                link=link.get("name"),
                mesh=(Path(urdf).parent / mesh.get("filename")).resolve(),
                xyz=_floats(origin.get("xyz") if origin is not None else None, "0 0 0"),
                rpy=_floats(origin.get("rpy") if origin is not None else None, "0 0 0"),
                scale=_floats(mesh.get("scale"), "1 1 1"),
                rgba=_floats(rgba, "1 1 1 1"),
            ))
    return out


def rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """URDF fixed-axis roll-pitch-yaw (R = Rz(yaw) Ry(pitch) Rx(roll)) as an xyzw quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


# ---------------------------------------------------------------------------
# Procedural neck: a tapered tube along a spline through the arm's link origins
# ---------------------------------------------------------------------------
def catmull_rom(points: np.ndarray, n_samples: int, alpha: float = 0.5) -> np.ndarray:
    """(n_samples, 3) points on a centripetal Catmull-Rom spline through ``points`` (K, 3), K >= 2.

    The curve passes through every control point; the ends are extended by reflection.
    """
    p = np.asarray(points, dtype=float)
    p = np.concatenate([2 * p[:1] - p[1:2], p, 2 * p[-1:] - p[-2:-1]])
    seg = np.maximum(np.linalg.norm(np.diff(p, axis=0), axis=1), 1e-6) ** alpha
    knots = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(knots[1], knots[-2], n_samples)
    i = np.clip(np.searchsorted(knots, t, side="right") - 1, 1, len(p) - 3)
    t0, t1, t2, t3 = (knots[i + k][:, None] for k in (-1, 0, 1, 2))
    p0, p1, p2, p3 = (p[i + k] for k in (-1, 0, 1, 2))
    tt = t[:, None]
    a1 = (t1 - tt) / (t1 - t0) * p0 + (tt - t0) / (t1 - t0) * p1
    a2 = (t2 - tt) / (t2 - t1) * p1 + (tt - t1) / (t2 - t1) * p2
    a3 = (t3 - tt) / (t3 - t2) * p2 + (tt - t2) / (t3 - t2) * p3
    b1 = (t2 - tt) / (t2 - t0) * a1 + (tt - t0) / (t2 - t0) * a2
    b2 = (t3 - tt) / (t3 - t1) * a2 + (tt - t1) / (t3 - t1) * a3
    return (t2 - tt) / (t2 - t1) * b1 + (tt - t1) / (t2 - t1) * b2


def tube_topology(n_rings: int, n_sides: int) -> tuple[np.ndarray, np.ndarray]:
    """USD ``faceVertexCounts`` and ``faceVertexIndices`` of an open tube (quads); fixed for a given size."""
    faces = []
    for i in range(n_rings - 1):
        for j in range(n_sides):
            a, b = i * n_sides + j, i * n_sides + (j + 1) % n_sides
            faces.append((a, b, b + n_sides, a + n_sides))  # counter-clockwise seen from outside
    faces = np.array(faces, dtype=np.int32)
    return np.full(len(faces), 4, dtype=np.int32), faces.reshape(-1)


def tube(centerline: np.ndarray, radii: np.ndarray, n_sides: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and outward normals, each (len(centerline) * n_sides, 3), of a tube with per-ring ``radii``.

    Ring frames are parallel-transported along the curve, so the tube does not twist.
    """
    c = np.asarray(centerline, dtype=float)
    tangent = np.gradient(c, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    seed = np.array([0.0, 0.0, 1.0]) if abs(tangent[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    normal = np.cross(tangent[0], seed)
    normal /= np.linalg.norm(normal)
    normals = [normal]
    for k in range(1, len(c)):
        n = normals[-1] - np.dot(normals[-1], tangent[k]) * tangent[k]  # project onto the new ring plane
        normals.append(n / max(np.linalg.norm(n), 1e-9))
    normals = np.array(normals)
    binormals = np.cross(tangent, normals)
    phi = np.linspace(0.0, 2 * np.pi, n_sides, endpoint=False)
    radial = np.cos(phi)[None, :, None] * normals[:, None] + np.sin(phi)[None, :, None] * binormals[:, None]
    points = c[:, None] + np.asarray(radii, dtype=float)[:, None, None] * radial
    return points.reshape(-1, 3), radial.reshape(-1, 3)
