"""Batched CPU snapshot of the Franka arm of every env in a ``RidgebackDREnv``, as input for the per-env arm QPs.

``ArmBatch`` reads all envs with a single GPU->CPU copy per refresh: joint state, the end-effector pose and world
Jacobian, the arm root (``panda_link0``) pose and, with ``dynamics=True``, the joint-space mass matrix, gravity +
Coriolis forces and a backward-difference Jacobian derivative (what the torque-level impedance QP needs). The
end-effector may be a fixed offset from a body (e.g. the TCP from ``panda_hand``). Everything is in the Isaac world
frame, which includes the env origin offset. A floating base moves the end-effector in this frame, so a world-frame
target makes the QP compensate for base motion on its own.

Needs Kit running (``env`` is an Isaac env), but this module does not import Isaac itself.
"""

from __future__ import annotations

import numpy as np
import torch

from hybrid_control.robots.state import Pose
from hybrid_control.utils.torch_rotations import quat_apply, quat_mul

EE_FRAME = "panda_hand"
ROOT_FRAME = "panda_link0"
N_BASE_DOFS = 6  # floating-base columns that precede the joint columns of the Isaac Lab Jacobian and dynamics


class ArmBatch:
    """CPU snapshot of the arm state of all envs."""

    def __init__(self, env, ee_frame: str = EE_FRAME, ee_body: str | None = None, ee_offset: Pose | None = None,
                 root_frame: str = ROOT_FRAME, dynamics: bool = False):
        """The end-effector ``ee_frame`` is ``ee_offset`` in body ``ee_body`` (default: the body ``ee_frame`` itself).

        ``dynamics=True`` also reads the mass matrix, nonlinear forces and Jacobian derivative (impedance QP).
        """
        self.env = env
        robot = env.robot
        self.joint_names = tuple(robot.joint_names[i] for i in env.arm_ids)
        self.ee_frame, self.root_frame, self.dynamics = ee_frame, root_frame, dynamics
        self.ee_id = robot.find_bodies(ee_body or ee_frame)[0][0]
        self.root_id = robot.find_bodies(root_frame)[0][0]
        dev = robot.device
        offset = ee_offset or Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
        self._off_pos = torch.tensor(np.asarray(offset.position), dtype=torch.float32, device=dev)
        self._off_quat = torch.tensor(np.asarray(offset.quaternion), dtype=torch.float32, device=dev)
        self.dof_cols = [N_BASE_DOFS + i for i in env.arm_ids]
        lim = robot.data.joint_pos_limits.torch[0, env.arm_ids].cpu().numpy()
        self.limits = (lim[:, 0].copy(), lim[:, 1].copy())
        self.time = 0.0
        self._jac_prev = self._time_prev = None
        self.refresh()

    def _ee_pose_t(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.env.robot.data
        pos, quat = data.body_link_pos_w.torch[:, self.ee_id], data.body_link_quat_w.torch[:, self.ee_id]
        return pos + quat_apply(quat, self._off_pos.expand_as(pos)), quat_mul(quat, self._off_quat.expand_as(quat))

    def refresh(self):
        """Copy the current state of all envs to the CPU. Call after every physics change and before the QP."""
        env, robot = self.env, self.env.robot
        data = robot.data
        n = len(env.arm_ids)
        cols = self.dof_cols
        ee_pos, ee_quat = self._ee_pose_t()
        jac = data.body_link_jacobian_w.torch[:, self.ee_id][:, :, cols].clone()
        r = ee_pos - data.body_link_pos_w.torch[:, self.ee_id]  # v_ee = v_body + w x r
        jac[:, :3] += torch.cross(jac[:, 3:], r[:, :, None].expand(-1, -1, n), dim=1)
        parts = [
            data.joint_pos.torch[:, env.arm_ids], data.joint_vel.torch[:, env.arm_ids], ee_pos, ee_quat,
            jac.flatten(1), data.body_link_pose_w.torch[:, self.root_id], env.com_total,
        ]
        if self.dynamics:
            import warp as wp  # only needed inside Isaac Lab

            coriolis = wp.to_torch(robot.root_view.get_coriolis_and_centrifugal_compensation_forces())
            h = (data.gravity_compensation_forces.torch + coriolis)[:, cols]
            parts += [data.mass_matrix.torch[:, cols][:, :, cols].flatten(1), h]
        packed = torch.cat(parts, dim=1).cpu().numpy().astype(np.float64)
        sizes = [n, n, 3, 4, 6 * n, 7, 3] + ([n * n, n] if self.dynamics else [])
        chunks = np.split(packed, np.cumsum(sizes)[:-1], axis=1)
        self.q, self.qd, self.ee_pos, self.ee_quat, jac_np, self.root_pose, self.com = chunks[:7]
        self.jacobian = jac_np.reshape(-1, 6, n)
        self.time = env.time
        if self.dynamics:
            self.mass_matrix = chunks[7].reshape(-1, n, n)
            self.nonlinear_forces = chunks[8]
            if self._jac_prev is None or self.time <= self._time_prev:
                self.jacobian_dot = np.zeros_like(self.jacobian)
            else:
                self.jacobian_dot = (self.jacobian - self._jac_prev) / (self.time - self._time_prev)
            self._jac_prev, self._time_prev = self.jacobian.copy(), self.time

    def reset_history(self, env_ids: np.ndarray):
        """Restart the Jacobian derivative of ``env_ids`` (after a reset teleports their arms)."""
        if self.dynamics and self._jac_prev is not None:
            self._jac_prev[env_ids] = self.jacobian[env_ids]
            self.jacobian_dot[env_ids] = 0.0

    @property
    def ee_pose(self) -> np.ndarray:
        """(N, 7) end-effector world poses, position + xyzw quaternion."""
        return np.concatenate([self.ee_pos, self.ee_quat], axis=1)

    @property
    def limits_array(self) -> np.ndarray:
        """(n, 2) joint position limits."""
        return np.stack(self.limits, axis=-1)

    def ee_pose_t(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Live end-effector position (N, 3) and xyzw quaternion (N, 4) in the world frame, on the simulation device."""
        return self._ee_pose_t()

    def ee_pos_t(self) -> torch.Tensor:
        """Live end-effector positions (N, 3) in the world frame, on the simulation device (no refresh needed)."""
        return self._ee_pose_t()[0]
