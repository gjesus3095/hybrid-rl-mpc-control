"""Whether the arm can follow a timed TCP reference from the spawn pose with a base that cannot move (pinocchio, no Isaac).

``ArmReach.trajectory_ok`` runs differential IK along the sampled reference, seeded by the previous solution and starting
from the folded spawn configuration, so a pass means a *continuous* joint path exists: no restarts, no branch jumps. A
reference passes only if, at every sample,

- the TCP pose is met (position and orientation within ``pos_tol`` / ``ori_tol``),
- every joint stays at least ``joint_margin`` inside its limits, and
- no joint needs more than ``max_speed_fraction`` of the Panda velocity limit between two samples.

This is a kinematic necessary condition, not a promise about a particular controller (dynamics, payload, singularity
handling), so the evaluation of a fixed base on the sets built with it is still the final check.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import pinocchio as pin

URDF = files("hybrid_control.sim_interfaces.isaac.assets.franka") / "lula_franka_gen.urdf"
Q_FOLD = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 2.0, 0.741])  # arm spawn configuration (Isaac Lab Ridgeback-Franka)
JOINT_SPEED_LIMIT = np.array([2.175] * 4 + [2.61] * 3)  # Panda joint velocity limits [rad/s]


@dataclass(frozen=True)
class ReachCfg:
    pos_tol: float = 5e-4  # [m]
    ori_tol: float = 5e-3  # [rad]
    joint_margin: float = 0.05  # [rad] kept away from the joint limits
    max_speed_fraction: float = 0.7  # of ``JOINT_SPEED_LIMIT``
    stride: int = 4  # reference samples (REF_DT) between IK solves: 4 = the 30 Hz policy step
    max_iters: int = 100


class ArmReach:
    """Panda on the arm mount ``arm_base`` (env-local), TCP defined by the spawn pose the benchmark starts from."""

    def __init__(self, spawn: tuple, arm_base=(0.30, 0.0, 0.28), cfg: ReachCfg | None = None):
        self.cfg = cfg or ReachCfg()
        self.model = pin.buildModelFromUrdf(str(URDF))
        self.arm_base = np.asarray(arm_base, float)
        self.lo = self.model.lowerPositionLimit[:7] + self.cfg.joint_margin
        self.hi = self.model.upperPositionLimit[:7] - self.cfg.joint_margin
        hand = self.model.getFrameId("panda_hand")
        data = self.model.createData()
        full = np.zeros(self.model.nq)
        full[:7] = Q_FOLD
        pin.forwardKinematics(self.model, data, full)
        pin.updateFramePlacements(self.model, data)
        spawn_pose = self._se3(np.asarray(spawn[0], float), np.asarray(spawn[1], float))
        offset = data.oMf[hand].inverse() * spawn_pose  # TCP in the hand frame, so the spawn pose is met exactly at Q_FOLD
        frame = self.model.frames[hand]
        parent_joint = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
        self.tcp = self.model.addFrame(pin.Frame("tcp", parent_joint, hand, frame.placement * offset, pin.FrameType.OP_FRAME))
        self.data = self.model.createData()

    def _se3(self, position: np.ndarray, quat_xyzw: np.ndarray) -> pin.SE3:
        """Env-local pose -> pose in the arm-base frame."""
        quat = np.asarray(quat_xyzw, float)
        return pin.SE3(pin.Quaternion(quat / np.linalg.norm(quat)).toRotationMatrix(), position - self.arm_base)

    def _solve(self, target: pin.SE3, q: np.ndarray) -> tuple[np.ndarray, bool]:
        c = self.cfg
        full = np.zeros(self.model.nq)
        for _ in range(c.max_iters):
            full[:7] = q
            pin.forwardKinematics(self.model, self.data, full)
            pin.updateFramePlacements(self.model, self.data)
            err = pin.log6(self.data.oMf[self.tcp].inverse() * target).vector  # twist in the TCP frame
            if np.linalg.norm(err[:3]) < c.pos_tol and np.linalg.norm(err[3:]) < c.ori_tol:
                return q, True
            jac = pin.computeFrameJacobian(self.model, self.data, full, self.tcp, pin.ReferenceFrame.LOCAL)[:, :7]
            q = np.clip(q + 0.7 * jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err), self.lo, self.hi)
        return q, False

    def trajectory_ok(self, poses: np.ndarray, dt: float) -> bool:
        """``poses`` (T, 7) env-local TCP position + xyzw quaternion sampled every ``dt`` seconds."""
        c = self.cfg
        q = np.clip(Q_FOLD, self.lo, self.hi)
        idx = list(range(0, len(poses), c.stride))
        if idx[-1] != len(poses) - 1:
            idx.append(len(poses) - 1)
        previous, previous_k = q, 0
        for k in idx:
            q, ok = self._solve(self._se3(poses[k, :3].astype(float), poses[k, 3:].astype(float)), previous)
            if not ok:
                return False
            if k > previous_k and (np.abs(q - previous) / ((k - previous_k) * dt) / JOINT_SPEED_LIMIT).max() > c.max_speed_fraction:
                return False
            previous, previous_k = q, k
        return True
