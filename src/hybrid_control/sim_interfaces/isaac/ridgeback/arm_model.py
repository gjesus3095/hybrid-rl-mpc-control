"""``RobotModel`` views of the Franka arm of every env in a ``RidgebackDREnv``, so the NumPy IK/QP controller
can drive each env's arm unchanged.

``ArmBatch`` owns one CPU snapshot of all envs (a single GPU->CPU copy per refresh); ``IsaacArmModel`` is the
``RobotModel`` for env ``i`` and reads that snapshot. Everything is in the Isaac world frame, which includes the
env origin offset. A floating base moves the end-effector in this frame, so a world-frame target makes the QP
compensate for base motion on its own.

Needs Kit running (``env`` is an Isaac env), but this module does not import Isaac itself.
"""

from __future__ import annotations

import numpy as np
import torch

from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import CenterOfMassState, JointPositionCommand, JointState, Pose

EE_FRAME = "panda_hand"
N_BASE_DOFS = 6  # floating-base columns that precede the joint columns of the Isaac Lab Jacobian


class ArmBatch:
    """CPU snapshot of the arm state of all envs plus the command buffer the QP writes into."""

    def __init__(self, env, ee_frame: str = EE_FRAME, track_orientation: bool = True):
        """``track_orientation=False`` zeroes the angular Jacobian rows, so the QP only sees the position task.

        The QP weights only scale the desired twist, not the Jacobian rows, so a nonzero orientation row always
        constrains the arm (even at weight 0); dropping the rows is how to track position alone.
        """
        self.env = env
        self.track_orientation = track_orientation
        robot = env.robot
        self.joint_names = tuple(robot.joint_names[i] for i in env.arm_ids)
        self.ee_frame = ee_frame
        self.ee_id = robot.find_bodies(ee_frame)[0][0]
        self.jac_cols = [N_BASE_DOFS + i for i in env.arm_ids]
        lim = robot.data.joint_pos_limits.torch[0, env.arm_ids].cpu().numpy()
        self.limits = (lim[:, 0].copy(), lim[:, 1].copy())
        n = env.num_envs
        self.q_cmd = np.zeros((n, len(env.arm_ids)))
        self.time = 0.0
        self.refresh()

    def refresh(self):
        """Copy the current state of all envs to the CPU. Call after every physics change and before the QP."""
        env, robot = self.env, self.env.robot
        q = robot.data.joint_pos.torch[:, env.arm_ids]
        qd = robot.data.joint_vel.torch[:, env.arm_ids]
        pos = robot.data.body_link_pos_w.torch[:, self.ee_id]
        quat = robot.data.body_link_quat_w.torch[:, self.ee_id]
        jac = robot.data.body_link_jacobian_w.torch[:, self.ee_id][:, :, self.jac_cols]
        packed = torch.cat([q, qd, pos, quat, jac.flatten(1), env.com_total], dim=1).cpu().numpy().astype(np.float64)
        n = len(env.arm_ids)
        self.q, self.qd = packed[:, :n], packed[:, n : 2 * n]
        self.ee_pos, self.ee_quat = packed[:, 2 * n : 2 * n + 3], packed[:, 2 * n + 3 : 2 * n + 7]
        self.jacobian = packed[:, 2 * n + 7 : 2 * n + 7 + 6 * n].reshape(-1, 6, n)
        self.com = packed[:, 2 * n + 7 + 6 * n :]
        self.time = env.time

    def ee_pos_t(self) -> torch.Tensor:
        """Live end-effector positions (N, 3) in the world frame, on the simulation device (no refresh needed)."""
        return self.env.robot.data.body_link_pos_w.torch[:, self.ee_id]

    def set_command(self, env_ids, q: np.ndarray):
        self.q_cmd[env_ids] = q

    def models(self) -> list["IsaacArmModel"]:
        return [IsaacArmModel(self, i) for i in range(self.env.num_envs)]


class IsaacArmModel(RobotModel):
    """The arm of env ``index``, read from an ``ArmBatch`` snapshot."""

    def __init__(self, batch: ArmBatch, index: int):
        self._batch, self._i = batch, index

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._batch.joint_names

    @property
    def position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._batch.limits[0].copy(), self._batch.limits[1].copy()

    def get_state(self) -> JointState:
        b = self._batch
        return JointState(b.time, b.q[self._i], b.qd[self._i])

    def get_com_state(self) -> CenterOfMassState:
        b = self._batch
        return CenterOfMassState(b.time, b.com[self._i])

    def get_pose(self, frame_name: str) -> Pose:
        self._check_frame(frame_name)
        b = self._batch
        return Pose(b.ee_pos[self._i], b.ee_quat[self._i])

    def get_jacobian(self, frame_name: str) -> np.ndarray:
        self._check_frame(frame_name)
        jacobian = self._batch.jacobian[self._i].copy()
        if not self._batch.track_orientation:
            jacobian[3:] = 0.0
        return jacobian

    def apply_command(self, command: JointPositionCommand) -> None:
        if tuple(command.joint_names) != self.joint_names:
            raise ValueError("command joint names must match the model's joint names")
        self._batch.set_command(self._i, command.positions)

    def _check_frame(self, frame_name: str):
        if frame_name != self._batch.ee_frame:
            raise KeyError(f"only the end-effector frame {self._batch.ee_frame!r} is available, got {frame_name!r}")
