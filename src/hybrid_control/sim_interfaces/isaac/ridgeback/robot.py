"""Ridgeback-Franka asset handling for Isaac Lab 3.0: free base, per-env footprint and mass model.

Import this only after the launcher arguments have been parsed (it imports ``isaaclab.sim``). ``pxr`` is
imported lazily inside the functions that need it because it must load after Kit has started.

The stock USD pins ``base_link`` to the world through three dummy joints (x, y, yaw), so it can never tilt.
``free_and_resize_base`` deactivates them, which makes ``base_link`` a floating root, and reshapes the
platform. ``MassModel`` then applies per-env masses at runtime.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils import clone
from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from .params import BaseParams

STOCK_FOOTPRINT = (0.9325, 0.7900)  # base collision size in x, y [m] (centred on the base frame)
STOCK_HEIGHT = 0.28  # top plate height [m]; the arm mounts here
MOUNT_X = 0.30  # arm mount ahead of the base frame [m]; the footprint is centred here
TCP_Z = 0.1034  # TCP distance along the hand z axis [m]
PANDA_RATED_PAYLOAD = 3.0  # [kg]
ROOT_HEIGHT = 0.01  # initial base_link height above the ground [m]
Q_STRETCHED = {"panda_joint2": 1.5, "panda_joint4": -0.1, "panda_joint6": 1.6}


def build_robot_cfg(prim_path: str, base_mass: float) -> "ArticulationCfg":  # noqa: F821
    """Stock config minus the dummy-joint base drive. ``prim_path`` may be a regex over several envs."""
    cfg = clone(RIDGEBACK_FRANKA_PANDA_CFG)
    cfg.prim_path = prim_path
    cfg.actuators.pop("base")  # the dummy joints no longer exist
    for name in [k for k in cfg.init_state.joint_pos if k.startswith("dummy")]:
        cfg.init_state.joint_pos.pop(name)
    cfg.init_state.pos = (0.0, 0.0, ROOT_HEIGHT)
    cfg.articulation_root_prim_path = "/base_link"  # the free root, see free_and_resize_base()
    cfg.spawn.mass_props = {"/base_link": [sim_utils.MassCfg(mass=base_mass)]}
    return cfg


def free_and_resize_base(prim_path: str, footprint: tuple[float, float], friction: float = 0.0, fixed_base: bool = False):
    """Deactivate the dummy world anchor, re-shape the platform under the arm mount, set its material.

    ``prim_path`` is one robot instance (no regex). Must run before ``sim.reset()``. ``fixed_base`` welds ``base_link``
    to the world at its spawn pose with a ``FixedJoint`` (the articulation then has no floating-base DoFs).
    """
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = sim_utils.get_current_stage()
    for name in ("world", "dummy_base_x", "dummy_base_y"):
        stage.GetPrimAtPath(f"{prim_path}/{name}").SetActive(False)
    # Make base_link the articulation root. Left on the parent Xform, PhysX would pick an arbitrary link
    # as the floating root, so root-pose writes would move the wrong body.
    stage.GetPrimAtPath(prim_path).RemoveAPI(UsdPhysics.ArticulationRootAPI)
    base_prim = stage.GetPrimAtPath(f"{prim_path}/base_link")
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(base_prim).CreateEnabledSelfCollisionsAttr(False)
    if fixed_base:
        # body0 left empty = the world; the joint frame is base_link's spawn pose, so nothing moves at the first step
        weld = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/base_weld")
        weld.CreateBody1Rel().SetTargets([base_prim.GetPath()])
        pose = UsdGeom.Xformable(base_prim).ComputeLocalToWorldTransform(0.0)
        weld.CreateLocalPos0Attr().Set(Gf.Vec3f(pose.ExtractTranslation()))
        weld.CreateLocalRot0Attr().Set(Gf.Quatf(pose.ExtractRotationQuat()))
        weld.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        weld.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    scale = (footprint[0] / STOCK_FOOTPRINT[0], footprint[1] / STOCK_FOOTPRINT[1], 1.0)
    for child in base_prim.GetChildren():
        if child.GetName() not in ("visuals", "collisions"):
            continue
        # Untyped prims ignore transform ops, so type them first. Order: scale about the stock
        # centre, then translate the footprint under the arm mount.
        if not child.GetTypeName():
            child.SetTypeName("Xform")
        xform = UsdGeom.Xformable(child)
        xform.AddTranslateOp().Set(Gf.Vec3d(MOUNT_X, 0.0, 0.0))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        if child.GetName() == "visuals":
            UsdGeom.Imageable(child).MakeInvisible()  # replaced by the platform box below

    # The stock laser scanners (0.13 kg each) sit at the stock corners, outside a smaller footprint, so they
    # would float in the air. Hide their visuals; their small masses stay in the platform model.
    for name in ("front_laser", "rear_laser"):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/{name}")).MakeInvisible()

    # Opaque platform box (visual only): shows the footprint.
    box = sim_utils.CuboidCfg(
        size=(footprint[0], footprint[1], STOCK_HEIGHT),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.40, 0.50), roughness=0.5, metallic=0.2),
    )
    box.func(f"{prim_path}/base_link/platform_visual", box, translation=(MOUNT_X, 0.0, STOCK_HEIGHT / 2))

    # A free box would slide with Coulomb friction (stalling the velocity tracker at ~mu*g*t). Use a
    # low-friction platform material; "min" outranks the ground's default "average" combine mode.
    material = f"{prim_path}/platform_material"
    sim_utils.spawn_rigid_body_material(
        material,
        PhysxRigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction, friction_combine_mode="min"),
    )
    sim_utils.bind_physics_material(f"{prim_path}/base_link/collisions", material)


class MassModel:
    """Per-env box COM/inertia for the platform and a point payload at the TCP. The arm itself is untouched.

    The nominal mass properties are captured on construction, so ``set_params`` can be called repeatedly
    (e.g. at resets) without the payload accumulating on the hand.
    """

    def __init__(self, robot: Articulation):
        self.robot = robot
        self.base_id = robot.find_bodies("base_link")[0][0]
        self.hand_id = robot.find_bodies("panda_hand")[0][0]
        self._masses = robot.data.body_mass.torch.clone()
        self._com_pos = robot.data.body_com_pos_b.torch.clone()
        self._com_quat = robot.data.body_com_quat_b.torch.clone()
        self._inertia = robot.data.body_inertia.torch.clone()

    def set_params(self, params: list[BaseParams], env_ids: torch.Tensor | None = None):
        """Write the mass model of ``env_ids`` (all envs if None); ``params[k]`` belongs to ``env_ids[k]``."""
        device = self.robot.device
        ids = torch.arange(self.robot.num_instances, device=device) if env_ids is None else env_ids.to(device).long()
        if len(params) != len(ids):
            raise ValueError(f"got {len(params)} params for {len(ids)} envs")
        masses, com_pos, inertia = self._masses[ids].clone(), self._com_pos[ids].clone(), self._inertia[ids].clone()
        for k, p in enumerate(params):
            length, width = p.footprint
            m, h = p.base_mass, STOCK_HEIGHT
            masses[k, self.base_id] = m
            com_pos[k, self.base_id] = torch.tensor([MOUNT_X, 0.0, p.base_com_z], device=device)
            inertia[k, self.base_id] = 0.0
            inertia[k, self.base_id, 0] = m * (width**2 + h**2) / 12.0
            inertia[k, self.base_id, 4] = m * (length**2 + h**2) / 12.0
            inertia[k, self.base_id, 8] = m * (length**2 + width**2) / 12.0
            if p.payload > 0.0:
                m_hand = masses[k, self.hand_id].clone()
                total = m_hand + p.payload
                com_pos[k, self.hand_id, 2] = (com_pos[k, self.hand_id, 2] * m_hand + TCP_Z * p.payload) / total
                inertia[k, self.hand_id] += torch.eye(3, device=device).flatten() * 1e-3 * p.payload
                masses[k, self.hand_id] = total
        env_ids = ids.to(torch.int32)
        self.robot.set_masses_index(masses=masses, env_ids=env_ids)
        self.robot.set_coms_index(coms=torch.cat([com_pos, self._com_quat[ids]], dim=-1), env_ids=env_ids)
        self.robot.set_inertias_index(inertias=inertia, env_ids=env_ids)


def franka_tcp_offset() -> "Pose":  # noqa: F821
    """``panda_hand -> panda_fingertip_centered`` from Isaac Lab's Factory Franka (franka_mimic.usd), xyzw.

    Same source as the TCP of the base-disturbance demos. Needs Kit running (imports ``pxr``).
    """
    import numpy as np
    from pxr import Usd, UsdGeom

    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

    from hybrid_control.robots.state import Pose

    hand, tcp = "panda_hand", "panda_fingertip_centered"
    stage = Usd.Stage.Open(retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/Factory/franka_mimic.usd"))
    cache = UsdGeom.XformCache()
    world = {p.GetName(): cache.GetLocalToWorldTransform(p) for p in stage.Traverse() if p.GetName() in (hand, tcp)}
    relative = world[tcp] * world[hand].GetInverse()  # Gf row-vector convention: TCP in hand
    rotation = relative.ExtractRotationQuat()
    return Pose(np.array(relative.ExtractTranslation()), np.array([*rotation.GetImaginary(), rotation.GetReal()]))


def tilt(quat_xyzw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched (roll, pitch, yaw) [rad] of xyzw quaternions (Isaac Lab 3.0 order), each of shape (N,)."""
    x, y, z, w = quat_xyzw.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


__all__ = [
    "MOUNT_X", "PANDA_RATED_PAYLOAD", "Q_STRETCHED", "ROOT_HEIGHT", "STOCK_HEIGHT", "MassModel",
    "build_robot_cfg", "franka_tcp_offset", "free_and_resize_base", "tilt",
]
