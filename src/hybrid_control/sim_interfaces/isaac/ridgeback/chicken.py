"""Cartoon chicken skin for the free-base Ridgeback-Franka (visual only; physics, RL and QP are unchanged).

The body sits in a straw nest on the platform (``base_link``), the head is on ``panda_hand`` and the two beak halves
are on the fingers, so the beak opens with the gripper. These parts come from the overlay URDF
``assets/chicken/ridgeback_franka_chicken.urdf`` and are attached under the matching link prims. The neck is a
tapered tube along a spline through the arm's link origins, regenerated every frame by ``update``.

Import this only after the launcher has started Kit (it imports ``pxr`` and ``isaaclab.sim``). Create the skin
after the robot is spawned and before ``sim.reset()``:

    skin = ChickenSkin(["/World/envs/env_0/Robot"])
    ...
    robot.update(dt); skin.update(robot)
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from .chicken_geometry import CHICKEN_URDF, catmull_rom, load_chicken_visuals, read_obj, rpy_to_quat_xyzw, tube, tube_topology

# Neck spline through these link origins (the elbow and the start inside the body are derived from them)
NECK_BODIES = ("panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_hand")
NECK_RADIUS = (0.085, 0.06)  # at the body, at the head [m]
CRATE_COLOR = (0.42, 0.25, 0.11)  # the platform box becomes a wooden crate
ROBOT_LINKS = ("arm_mount_link", *(f"panda_link{i}" for i in range(8)), "panda_hand", "panda_leftfinger",
               "panda_rightfinger")
LOOKS = "/World/Looks/Chicken"


def _material(name: str, rgb) -> str:
    path = f"{LOOKS}/{name}"
    if not sim_utils.get_current_stage().GetPrimAtPath(path).IsValid():
        cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(rgb[:3]), roughness=0.65, metallic=0.0)
        cfg.func(path, cfg)
    return path


@cache
def _obj(path: Path):
    return read_obj(path)


def _define_mesh(stage: Usd.Stage, path: str, points, normals, counts, indices) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(counts, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(indices, dtype=np.int32).reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    _set_points(mesh, points, normals)
    return mesh


def _set_points(mesh: UsdGeom.Mesh, points: np.ndarray, normals: np.ndarray):
    points = np.ascontiguousarray(points, dtype=np.float32)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, dtype=np.float32)))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0), points.max(0)])))


class ChickenSkin:
    """Chicken visuals on one or more robot instances (``robot_prims[k]`` is env k of the articulation)."""

    def __init__(self, robot_prims: list[str], hide_robot: bool = True, urdf: Path = CHICKEN_URDF,
                 n_rings: int = 64, n_sides: int = 24):
        stage = sim_utils.get_current_stage()
        self.n_rings, self.n_sides = n_rings, n_sides
        self.radii = np.linspace(*NECK_RADIUS, n_rings)
        visuals = load_chicken_visuals(urdf)
        neck_rgba = next(v.rgba for v in visuals if v.name == "head")  # same feathers as the head
        counts, indices = tube_topology(n_rings, n_sides)
        self.necks, self.to_local = [], []
        for root in robot_prims:
            for v in visuals:
                xform = UsdGeom.Xform.Define(stage, f"{root}/{v.link}/chicken/{v.name}")
                qx, qy, qz, qw = rpy_to_quat_xyzw(*v.rpy)
                xform.AddTranslateOp().Set(Gf.Vec3d(*v.xyz))
                xform.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
                xform.AddScaleOp().Set(Gf.Vec3f(*v.scale))
                points, normals, faces = _obj(v.mesh)
                _define_mesh(stage, f"{xform.GetPath()}/mesh", points, normals, np.full(len(faces), 3), faces)
                sim_utils.bind_visual_material(str(xform.GetPath()), _material(_color_name(v.rgba), v.rgba))
            # The neck lives in the robot prim's frame; points are written in world coordinates mapped into it.
            neck_path = f"{root}/chicken_neck"
            centre = np.zeros((n_rings, 3))
            centre[:, 2] = np.linspace(0.3, 0.9, n_rings)
            neck = _define_mesh(stage, neck_path, *tube(centre, self.radii, n_sides), counts, indices)
            sim_utils.bind_visual_material(neck_path, _material(_color_name(neck_rgba), neck_rgba))
            self.necks.append(neck)
            world = UsdGeom.Xformable(stage.GetPrimAtPath(root)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            self.to_local.append(np.array(world.GetInverse()).T)  # column-vector convention

            platform = f"{root}/base_link/platform_visual"
            if stage.GetPrimAtPath(platform).IsValid():
                sim_utils.bind_visual_material(platform, _material("crate", CRATE_COLOR))
            if hide_robot:
                for link in ROBOT_LINKS:
                    prim = stage.GetPrimAtPath(f"{root}/{link}/visuals")
                    if prim.IsValid():
                        UsdGeom.Imageable(prim).MakeInvisible()
        self._body_ids = None

    def neck_control_points(self, p: np.ndarray) -> np.ndarray:
        """(5, 3) spline control points from the ``NECK_BODIES`` origins ``p`` (6, 3).

        Starts inside the body (30 % of the way from the mount to the shoulder) and averages the two elbow frames,
        which sit 0.0825 m apart on either side of the elbow. ``panda_link7`` is skipped: its origin is offset
        sideways from the wrist and would put a bump behind the head.
        """
        link0, link2, link3, link4, link5, hand = p
        return np.stack([link0 + 0.3 * (link2 - link0), link2, 0.5 * (link3 + link4), link5, hand])

    def update(self, robot: Articulation):
        """Rebuild every neck from the current link poses (call after ``robot.update``)."""
        if self._body_ids is None:
            self._body_ids = robot.find_bodies(list(NECK_BODIES), preserve_order=True)[0]
        pos = robot.data.body_pos_w.torch[:, self._body_ids].cpu().numpy()
        for neck, to_local, p in zip(self.necks, self.to_local, pos):
            centre = catmull_rom(self.neck_control_points(p), self.n_rings)
            centre = centre @ to_local[:3, :3].T + to_local[:3, 3]
            _set_points(neck, *tube(centre, self.radii, self.n_sides))


def _color_name(rgba) -> str:
    return "c_" + "_".join(f"{round(c * 255):02x}" for c in rgba[:3])
