"""Vectorized floating platforms with a spline-driven three-joint moving mass."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from .params import MechanismCfg, MovingMassParams
from .robot import JOINT_NAMES, ROOT_HEIGHT, MassModel, build_robot_cfg, spawn_mechanism, tilt


TIP_ANGLE = 0.5  # roll/pitch [rad] beyond which the robot counts as tipped


class MovingMassEnv:
    def __init__(
        self,
        params: list[MovingMassParams],
        sim: SimulationContext,
        spacing: float = 2.5,
        num_cols: int | None = None,
        friction: float = 0.05,
        tau: float = 0.15,
        max_force: float = 1000.0,
        mechanism: MechanismCfg | None = None,
    ):
        self.mechanism = mechanism or MechanismCfg()
        if not params:
            raise ValueError("at least one environment is required")
        self.sim, self.params = sim, list(params)
        self.num_envs = len(params)
        self.tau, self.max_force = tau, max_force
        self.dt = sim.get_physics_dt()

        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)
        light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

        cols = num_cols or math.ceil(math.sqrt(self.num_envs))
        origins = [((i % cols) * spacing, -(i // cols) * spacing) for i in range(self.num_envs)]
        centre = (sum(o[0] for o in origins) / len(origins), sum(o[1] for o in origins) / len(origins))
        self.origins = torch.tensor([(x - centre[0], y - centre[1], 0.0) for x, y in origins], device=sim.device)
        for i, origin in enumerate(self.origins.tolist()):
            sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=origin)

        for i, p in enumerate(params):
            spawn_mechanism(f"/World/envs/env_{i}/Robot", p, self.mechanism, friction)
        # Mobile bases may leave their spawn cells; neighboring episodes must not collide.
        from isaaclab.cloner import filter_collisions
        filter_collisions(sim.stage, sim.cfg.physics_prim_path, '/World/collisions',
                          [f'/World/envs/env_{i}' for i in range(self.num_envs)],
                          global_paths=['/World/defaultGroundPlane'])
        self.robot = Articulation(cfg=build_robot_cfg("/World/envs/env_.*/Robot", self.mechanism))
        self.half_length = torch.tensor([0.5 * p.footprint[0] for p in params], device=sim.device)
        self.half_width = torch.tensor([0.5 * p.footprint[1] for p in params], device=sim.device)

    def start(self):
        """Reset the simulation, put every robot in its default pose and apply the mass model."""
        self.sim.reset()
        robot = self.robot
        self.base_id = robot.find_bodies("base_link")[0][0]
        self.mechanism_body_ids = [i for i, n in enumerate(robot.body_names) if n != 'base_link']
        self.platform_body_ids = [self.base_id]
        self.joint_ids, _ = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if len(self.joint_ids) != 3 or len(robot.joint_names) != 3:
            raise RuntimeError('moving-mass articulation must have exactly three joints')
        self.mass_model = MassModel(robot, self.mechanism)
        self.mass_model.set_params(self.params)
        self.masses = robot.data.body_mass.torch.clone()
        self.mass_total = torch.zeros(self.num_envs, device=robot.device)
        self.mass_mechanism = torch.zeros_like(self.mass_total)
        self.inertia_z = torch.zeros_like(self.mass_total)
        self._refresh_mass_cache()
        self.reset()

    def _refresh_mass_cache(self, env_ids: torch.Tensor | None = None):
        """Recompute the per-env masses and yaw inertia used by the velocity tracker (after a mass-model write)."""
        robot = self.robot
        ids = slice(None) if env_ids is None else env_ids.long()
        self.masses[ids] = robot.data.body_mass.torch[ids]
        m = self.masses[ids]
        self.mass_total[ids] = m.sum(1)
        self.mass_mechanism[ids] = m[:, self.mechanism_body_ids].sum(1)
        self.inertia_z[ids] = robot.data.body_inertia.torch[ids, self.base_id, 8]

    def reset(self, env_ids: torch.Tensor | None = None, dynamics: list[MovingMassParams] | None = None):
        """Back to the upright mechanism at the env origins; the tipped latch is cleared.

        env_ids: envs to reset (all if None).
        dynamics: optional new ``MovingMassParams`` per reset env (``dynamics[k]`` for ``env_ids[k]``). Only mass, COM
            height and terminal mass are taken from them; the footprint is fixed at spawn.
        """
        robot = self.robot
        ids = torch.arange(self.num_envs, device=robot.device) if env_ids is None else env_ids.to(robot.device).long()
        if dynamics is not None:
            if len(dynamics) != len(ids):
                raise ValueError(f"got {len(dynamics)} dynamics for {len(ids)} envs")
            for i, new in zip(ids.tolist(), dynamics):
                self.params[i] = self.params[i].with_dynamics(new)
            self.mass_model.set_params([self.params[i] for i in ids.tolist()], env_ids=ids)
            self._refresh_mass_cache(ids)
        idx = ids.to(torch.int32)
        pose = robot.data.default_root_pose.torch[ids].clone()
        pose[:, :3] = self.origins[ids] + torch.tensor([0.0, 0.0, ROOT_HEIGHT], device=robot.device)
        pose[:, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=robot.device)  # xyzw identity
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=idx)
        robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch[ids].clone(), env_ids=idx)
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch[ids].clone(), env_ids=idx)
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch[ids].clone(), env_ids=idx)
        robot.reset(ids.tolist())
        if env_ids is None:
            self.tipped = torch.zeros(self.num_envs, dtype=torch.bool, device=robot.device)
            self.left_footprint = torch.zeros_like(self.tipped)
            self.time = 0.0
        else:
            self.tipped[ids] = False
            self.left_footprint[ids] = False
        self._update_state()

    def step(self, joint_position: torch.Tensor, base_cmd: torch.Tensor, joint_velocity: torch.Tensor):
        """Advance one physics step. All inputs (N, 3).

        Joints ordered roll/pitch/lift: rad/rad/m and rad/s/rad/s/m/s.
        Base commands are yaw-frame vx/vy/wz, in m/s/m/s/rad/s.
        """
        robot = self.robot
        robot.actuators.target_command.set_position_index(value=joint_position, joint_ids=self.joint_ids)
        robot.actuators.target_command.set_velocity_index(value=joint_velocity, joint_ids=self.joint_ids)

        # velocity tracker -> planar force + yaw torque at ground height (off once tipped)
        vel = robot.data.body_com_lin_vel_w.torch[:, self.base_id]
        ang = robot.data.body_com_ang_vel_w.torch[:, self.base_id]
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        v_des = torch.stack([c * base_cmd[:, 0] - s * base_cmd[:, 1], s * base_cmd[:, 0] + c * base_cmd[:, 1]], dim=-1)
        force = torch.zeros(self.num_envs, 3, device=robot.device)
        torque = torch.zeros_like(force)
        force[:, :2] = self.mass_total[:, None] * (v_des - vel[:, :2]) / self.tau
        scale = (self.max_force / force[:, :2].norm(dim=-1).clamp_min(1e-9)).clamp_max(1.0)
        force[:, :2] *= scale[:, None]
        torque[:, 2] = self.inertia_z * (base_cmd[:, 2] - ang[:, 2]) / self.tau
        active = (~self.tipped).float()[:, None]
        force, torque = force * active, torque * active
        pos = robot.data.body_com_pos_w.torch[:, self.base_id]
        point = torch.cat([pos[:, :2], torch.zeros(self.num_envs, 1, device=robot.device)], dim=-1)
        robot.permanent_wrench_composer.reset()
        robot.permanent_wrench_composer.add_forces_and_torques_index(
            forces=force[:, None], torques=torque[:, None], positions=point[:, None], body_ids=[self.base_id],
            is_global=True,
        )

        robot.write_data_to_sim()
        self.sim.step()
        robot.update(self.dt)
        self.time += self.dt
        self._update_state()

    def _update_state(self):
        robot = self.robot
        self.roll, self.pitch, self.yaw = tilt(robot.data.body_quat_w.torch[:, self.base_id])
        body_com = robot.data.body_com_pos_w.torch
        m = self.masses

        def group_com(ids):
            return (body_com[:, ids] * m[:, ids, None]).sum(1) / m[:, ids].sum(1, keepdim=True)

        self.com_platform = group_com(self.platform_body_ids)
        self.com_mechanism = group_com(self.mechanism_body_ids)
        self.com_total = (body_com * m[:, :, None]).sum(1) / self.mass_total[:, None]
        # Footprint is centred on the floating root.
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        base_pos = robot.data.body_pos_w.torch[:, self.base_id]
        centre = base_pos[:, :2]
        rel = self.com_total[:, :2] - centre
        dx, dy = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]  # base-frame offsets
        self.com_offset = torch.maximum(dx.abs() / self.half_length, dy.abs() / self.half_width) * self.half_length
        self.left_footprint = self.com_offset > self.half_length
        self.tipped |= torch.maximum(self.roll.abs(), self.pitch.abs()) > TIP_ANGLE
        # Aggregate yaw inertia about the base CoM, including moving masses.
        x, y, z, w = robot.data.body_com_quat_w.torch.unbind(-1)
        axis = torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
        local_inertia = robot.data.body_inertia.torch.reshape(self.num_envs, -1, 3, 3)
        rotated_izz = torch.einsum('nbi,nbij,nbj->nb', axis, local_inertia, axis)
        distance2 = (body_com[:, :, :2] - body_com[:, self.base_id:self.base_id+1, :2]).square().sum(-1)
        self.inertia_z = (rotated_izz + m * distance2).sum(-1)
        self.base_pos = base_pos - self.origins  # env-local
