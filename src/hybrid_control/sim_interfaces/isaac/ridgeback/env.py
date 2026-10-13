"""Vectorized free-base Ridgeback-Franka environments with per-env domain randomization.

Every env is its own spawned robot (``/World/envs/env_<i>/Robot``), so the platform footprint can differ
between envs; masses, payload and COM height are applied at runtime through ``MassModel``.

The base is driven by a virtual velocity actuator -- a saturated first-order velocity tracker -- that applies
a planar force and yaw torque to ``base_link`` at ground height. Once an env has tipped the actuator is off.
Failure criterion: the whole-robot COM leaves the platform footprint (``com_offset > half_length``), and
``tipped`` latches when roll or pitch exceeds ``TIP_ANGLE``.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from .params import BaseParams
from .robot import (
    MOUNT_X, Q_STRETCHED, ROOT_HEIGHT, MassModel, build_robot_cfg, free_and_resize_base, tilt,
)

TIP_ANGLE = 0.5  # roll/pitch [rad] beyond which the robot counts as tipped


class RidgebackDREnv:
    def __init__(
        self,
        params: list[BaseParams],
        sim: SimulationContext,
        spacing: float = 2.5,
        num_cols: int | None = None,
        friction: float = 0.0,
        tau: float = 0.15,
        max_force: float = 1000.0,
    ):
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

        # Spawns one robot under every env_<i>, then each is reshaped with its own footprint.
        self.robot = Articulation(cfg=build_robot_cfg("/World/envs/env_.*/Robot", params[0].base_mass))
        for i, p in enumerate(params):
            free_and_resize_base(f"/World/envs/env_{i}/Robot", p.footprint, friction)
        self.half_length = torch.tensor([0.5 * p.footprint[0] for p in params], device=sim.device)
        self.half_width = torch.tensor([0.5 * p.footprint[1] for p in params], device=sim.device)

    def start(self):
        """Reset the simulation, put every robot in its default pose and apply the mass model."""
        self.sim.reset()
        robot = self.robot
        self.base_id = robot.find_bodies("base_link")[0][0]
        self.arm_body_ids = [i for i, n in enumerate(robot.body_names) if n.startswith("panda_")]
        self.platform_body_ids = [i for i, n in enumerate(robot.body_names) if not n.startswith("panda_")]
        self.arm_ids, arm_names = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
        self.finger_ids, _ = robot.find_joints("panda_finger_joint.*")
        self.q_fold = robot.data.default_joint_pos.torch[:, self.arm_ids].clone()
        self.q_stretched = self.q_fold.clone()
        for name, value in Q_STRETCHED.items():
            self.q_stretched[:, arm_names.index(name)] = value

        self.mass_model = MassModel(robot)
        self.mass_model.set_params(self.params)
        self.masses = robot.data.body_mass.torch.clone()
        self.mass_total = self.masses.sum(1)
        self.mass_arm = self.masses[:, self.arm_body_ids].sum(1)
        self.inertia_z = robot.data.body_inertia.torch[:, self.base_id, 8] + 0.5 * (
            self.mass_total - torch.tensor([p.base_mass for p in self.params], device=robot.device)
        ) * MOUNT_X**2
        self.reset()

    def reset(self):
        """Back to the folded arm at the env origins; the tipped latch is cleared."""
        robot = self.robot
        pose = robot.data.default_root_pose.torch.clone()
        pose[:, :3] = self.origins + torch.tensor([0.0, 0.0, ROOT_HEIGHT], device=robot.device)
        pose[:, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=robot.device)  # xyzw identity
        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
        robot.reset()
        self.tipped = torch.zeros(self.num_envs, dtype=torch.bool, device=robot.device)
        self.left_footprint = torch.zeros_like(self.tipped)
        self.time = 0.0
        self._update_state()

    def step(self, arm_q_target: torch.Tensor, base_cmd: torch.Tensor):
        """Advance one physics step.

        arm_q_target: (N, 7) arm joint position targets [rad].
        base_cmd: (N, 3) base-frame velocity command (vx, vy, wz) [m/s, m/s, rad/s].
        """
        robot = self.robot
        robot.actuators.target_command.set_position_index(value=arm_q_target, joint_ids=self.arm_ids)
        robot.actuators.target_command.set_position_index(
            value=torch.zeros(self.num_envs, len(self.finger_ids), device=robot.device), joint_ids=self.finger_ids
        )

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
        self.com_arm = group_com(self.arm_body_ids)
        self.com_total = (body_com * m[:, :, None]).sum(1) / self.mass_total[:, None]
        # footprint centre (yaw frame, under the arm mount) and the total COM relative to it
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        base_pos = robot.data.body_pos_w.torch[:, self.base_id]
        centre = base_pos[:, :2] + torch.stack([c * MOUNT_X, s * MOUNT_X], dim=-1)
        rel = self.com_total[:, :2] - centre
        dx, dy = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]  # base-frame offsets
        self.com_offset = torch.maximum(dx.abs() / self.half_length, dy.abs() / self.half_width) * self.half_length
        self.left_footprint |= self.com_offset > self.half_length
        self.tipped |= torch.maximum(self.roll.abs(), self.pitch.abs()) > TIP_ANGLE
        self.base_pos = base_pos - self.origins  # env-local
