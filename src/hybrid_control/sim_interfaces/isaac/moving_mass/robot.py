"""Primitive floating-platform articulation. Import after Isaac launcher setup."""
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

JOINT_NAMES = ('roll', 'pitch', 'lift')
ROOT_HEIGHT = 0.01


def spawn_mechanism(path, params, cfg, friction=0.05):
    """Author a tree with explicit positive mass/inertia and collision on every rigid body.

    USD angular limits are degrees; the runtime interface uses radians. Body
    origins are joint pivots. All authored joints start at zero; reset puts the mast upright.
    """
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, path)
    length, width = params.footprint
    length_mast = cfg.mast_length
    z = cfg.platform_height + 0.08

    def body(name, position, size, mass, com, collision=True, color=(0.35, 0.4, 0.5)):
        xform = UsdGeom.Xform.Define(stage, f'{path}/{name}')
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        sx, sy, sz = size
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(mass * (sy*sy + sz*sz)/12,
                                                   mass * (sx*sx + sz*sz)/12,
                                                   mass * (sx*sx + sy*sy)/12))
        cube = UsdGeom.Cube.Define(stage, f'{path}/{name}/visual')
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*( (0, 0, cfg.platform_height/2) if name == 'base_link' else com)))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            contact = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            contact.CreateContactOffsetAttr(0.003)
            contact.CreateRestOffsetAttr(0.0)
        return prim

    root = body('base_link', (0, 0, 0), (length, width, cfg.platform_height), params.base_mass,
                (0, 0, params.base_com_z), True)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(True)
    body('carriage', (0, 0, z), (0.06, 0.06, 0.08), params.carriage_mass, (0, 0, 0))
    body('roll_hub', (0, 0, z), (0.06, 0.06, 0.06), params.gimbal_mass, (0, 0, 0), color=(0.2, 0.55, 0.8))
    body('mast', (0, 0, z), (0.035, 0.035, length_mast), params.mast_mass,
         (0, 0, length_mast/2), color=(0.2, 0.7, 0.6))
    body('terminal', (0, 0, z+length_mast+0.04), (0.08, 0.08, 0.08), params.terminal_mass,
         (0, 0, 0), color=(0.9, 0.5, 0.15))
    # The compact two-axis bearing shares a pivot: its housing must not collide
    # with the mast passing through it. All other nonadjacent self contacts remain.
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f'{path}/carriage')).CreateFilteredPairsRel().SetTargets(
        [f'{path}/mast'])

    def joint(name, kind, parent, child, anchor, index=None):
        j = kind.Define(stage, f'{path}/{name}')
        j.CreateBody0Rel().SetTargets([f'{path}/{parent}'])
        j.CreateBody1Rel().SetTargets([f'{path}/{child}'])
        j.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateCollisionEnabledAttr(False)  # connected bodies share their joint pivot
        if index is not None:
            j.CreateAxisAttr(('X', 'Y', 'Z')[index])
            scale = 180/math.pi if index < 2 else 1.0
            j.CreateLowerLimitAttr(cfg.lower[index] * scale)
            j.CreateUpperLimitAttr(cfg.upper[index] * scale)
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'angular' if index < 2 else 'linear')
            drive.CreateTypeAttr('force')
            drive.CreateMaxForceAttr(cfg.effort_limit[index])
        return j

    joint('lift', UsdPhysics.PrismaticJoint, 'base_link', 'carriage', (0, 0, z), 2)
    joint('roll', UsdPhysics.RevoluteJoint, 'carriage', 'roll_hub', (0, 0, 0), 0)
    joint('pitch', UsdPhysics.RevoluteJoint, 'roll_hub', 'mast', (0, 0, 0), 1)
    joint('terminal_fixed', UsdPhysics.FixedJoint, 'mast', 'terminal', (0, 0, length_mast+0.04))
    material = f'{path}/platform_material'
    sim_utils.spawn_rigid_body_material(material, PhysxRigidBodyMaterialCfg(
        static_friction=friction, dynamic_friction=friction, friction_combine_mode='min'))
    sim_utils.bind_physics_material(f'{path}/base_link/visual', material)
    upper_material = f'{path}/upper_material'
    sim_utils.spawn_rigid_body_material(upper_material, PhysxRigidBodyMaterialCfg(
        static_friction=0.6, dynamic_friction=0.5, restitution=0.0))
    for name in ('carriage', 'roll_hub', 'mast', 'terminal'):
        sim_utils.bind_physics_material(f'{path}/{name}/visual', upper_material)


def build_robot_cfg(path, cfg):
    return ArticulationCfg(
        prim_path=path, spawn=None, articulation_root_prim_path='/base_link',
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, ROOT_HEIGHT), joint_pos=dict(zip(JOINT_NAMES, cfg.initial))),
        actuators={name: ImplicitActuatorCfg(
            joint_names_expr=[name], stiffness=cfg.stiffness[i], damping=cfg.damping[i],
            effort_limit_sim=cfg.effort_limit[i], velocity_limit_sim=cfg.velocity_limit[i],
        ) for i, name in enumerate(JOINT_NAMES)},
    )


class MassModel:
    def __init__(self, robot, cfg):
        self.robot, self.cfg = robot, cfg
        self.base_id = robot.find_bodies('base_link')[0][0]
        self.terminal_id = robot.find_bodies('terminal')[0][0]
        self.upper_ids = {name: robot.find_bodies(body)[0][0] for name, body in
                          (('carriage_mass', 'carriage'), ('gimbal_mass', 'roll_hub'), ('mast_mass', 'mast'))}
        self.mass = robot.data.body_mass.torch.clone()
        self.com = robot.data.body_com_pos_b.torch.clone()
        self.quat = robot.data.body_com_quat_b.torch.clone()
        self.inertia = robot.data.body_inertia.torch.clone()

    def set_params(self, params, env_ids=None):
        r = self.robot
        ids = torch.arange(r.num_instances, device=r.device) if env_ids is None else env_ids.long()
        if len(params) != len(ids):
            raise ValueError('one dynamics record is required per reset environment')
        mass, com, inertia = self.mass[ids].clone(), self.com[ids].clone(), self.inertia[ids].clone()
        for k, p in enumerate(params):
            length, width = p.footprint
            h, m = self.cfg.platform_height, p.base_mass
            mass[k, self.base_id] = m
            com[k, self.base_id] = torch.tensor([0, 0, p.base_com_z], device=r.device)
            inertia[k, self.base_id] = torch.tensor([m*(width**2+h**2)/12, 0, 0, 0,
                m*(length**2+h**2)/12, 0, 0, 0, m*(length**2+width**2)/12], device=r.device)
            for name, body_id in self.upper_ids.items():
                value = getattr(p, name)
                mass[k, body_id] = value
                # Scale the spawn inertia, never the previous episode's inertia.
                inertia[k, body_id] = self.inertia[ids[k], body_id] * (value / self.mass[ids[k], body_id])
            mass[k, self.terminal_id] = p.terminal_mass
            inertia[k, self.terminal_id] = torch.eye(3, device=r.device).flatten() * (p.terminal_mass * 0.08**2 / 6)
        idx = ids.to(torch.int32)
        r.set_masses_index(masses=mass, env_ids=idx)
        r.set_coms_index(coms=torch.cat((com, self.quat[ids]), -1), env_ids=idx)
        r.set_inertias_index(inertias=inertia, env_ids=idx)


def tilt(quat):
    """xyzw quaternion -> roll, pitch, yaw [rad]."""
    x, y, z, w = quat.unbind(-1)
    return (torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            torch.asin((2*(w*y-z*x)).clamp(-1, 1)),
            torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
