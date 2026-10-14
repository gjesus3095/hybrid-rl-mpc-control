"""Probe how to change the Ridgeback-Franka mobile base: platform mass, robot mass, platform size.

Three independent knobs (all optional; with none given the stock robot is reported):

    --base_mass KG         set mass of ``base_link`` at SPAWN time (cfg ``mass_props``)
    --robot_mass_scale S   multiply the mass of every body at RUNTIME (``set_masses_index``)
    --base_scale SX SY SZ  scale base_link collision + visual geometry at spawn (footprint/height)

Inertia: ``--no_inertia_rescale`` leaves inertia untouched; by default it is scaled with mass
(uniform-density assumption, same as ``randomize_rigid_body_mass``). Geometry scaling does NOT
move the arm mount or change mass; combine with --base_mass if you want consistent density.

Usage (from the Isaac Lab folder, headless):

    uv run --extra isaacsim python ~/hybrid-rl-mpc-control/scripts/isaac/ridgeback_base_probe.py \
        --base_mass 250 --robot_mass_scale 1.2 --base_scale 1.5 1.0 1.0
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Change Ridgeback base mass / robot mass / base size.")
parser.add_argument("--base_mass", type=float, default=None, help="Mass of base_link [kg].")
parser.add_argument("--robot_mass_scale", type=float, default=1.0, help="Scale factor on all body masses.")
parser.add_argument("--base_scale", type=float, nargs=3, default=None, metavar=("SX", "SY", "SZ"),
                    help="Scale of the base_link geometry in the base frame.")
parser.add_argument("--no_inertia_rescale", action="store_true", help="Do not scale inertia with runtime mass.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils import clone

from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG  # isort:skip


def scale_base_geometry(prim_path: str, scale: tuple[float, float, float]):
    """Multiply the scale of the visual and collision Xforms directly under ``prim_path``.

    Only geometry changes: the joints (arm mount, lasers) keep their frames and masses are
    untouched. Must run before ``sim.reset()`` so the physics engine cooks the new shapes.
    """
    from pxr import Gf, Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    base = stage.GetPrimAtPath(prim_path)
    for child in base.GetChildren():
        if child.GetName() not in ("visuals", "collisions"):
            continue
        # ``visuals``/``collisions`` are untyped prims in the USD, which USD does not treat as
        # transformable: a scale op authored on them is silently ignored. Type them as Xform first.
        if not child.GetTypeName():
            child.SetTypeName("Xform")
        xform = UsdGeom.Xformable(child)
        ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
        op = ops.get("xformOp:scale") or xform.AddScaleOp()
        old = op.Get() or Gf.Vec3f(1, 1, 1)
        op.Set(Gf.Vec3f(old[0] * scale[0], old[1] * scale[1], old[2] * scale[2]))


def base_collision_size(prim_path: str):
    """World-aligned size [m] of the base_link collision geometry."""
    from pxr import Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(f"{prim_path}/collisions")).ComputeAlignedRange()
    return tuple(round(v, 3) for v in rng.GetSize())


def report(tag: str, robot: Articulation, base_id: int):
    mass = robot.data.body_mass.torch[0]
    print(f"[PROBE] {tag}: base_link={mass[base_id].item():.2f} kg, total={mass.sum().item():.2f} kg, "
          f"base inertia diag={robot.data.body_inertia.torch[0, base_id][[0, 4, 8]].tolist()}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)

        robot_cfg = clone(RIDGEBACK_FRANKA_PANDA_CFG)
        robot_cfg.prim_path = "/World/Robot"
        if args_cli.base_mass is not None:
            # Keys are regex suffixes under the spawn prim (/World/Robot); the USD default prim is
            # /panda_mobile, so base_link is "/base_link". Inertia is NOT rescaled by this route.
            robot_cfg.spawn.mass_props = {"/base_link": [sim_utils.MassCfg(mass=args_cli.base_mass)]}
        robot = Articulation(cfg=robot_cfg)  # spawns the USD

        print("[PROBE] base collision size before:", base_collision_size("/World/Robot/base_link"))
        if args_cli.base_scale is not None:
            scale_base_geometry("/World/Robot/base_link", args_cli.base_scale)
        print("[PROBE] base collision size after :", base_collision_size("/World/Robot/base_link"))

        sim.reset()
        base_id = robot.find_bodies("base_link")[0][0]
        report("spawn", robot, base_id)

        if args_cli.robot_mass_scale != 1.0:
            masses = robot.data.body_mass.torch.clone() * args_cli.robot_mass_scale
            robot.set_masses_index(masses=masses, full_data=True)
            if not args_cli.no_inertia_rescale:
                inertias = robot.data.body_inertia.torch.clone() * args_cli.robot_mass_scale
                robot.set_inertias_index(inertias=inertias)
            sim.step()
            robot.update(sim.get_physics_dt())
            report("runtime-scaled", robot, base_id)

        print("[PROBE] bodies:", dict(zip(robot.body_names, [round(m, 3) for m in robot.data.body_mass.torch[0].tolist()])))


if __name__ == "__main__":
    main()
