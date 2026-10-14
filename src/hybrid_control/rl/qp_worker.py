"""Worker process of ``QPPool``: solves the IK/QP of a slice of envs per request. Not meant to be run by hand."""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import resource_tracker, shared_memory

import numpy as np

from hybrid_control.controllers.ik_qp.controller import IKQPConfig, InverseKinematicsController, PoseTarget
from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import CenterOfMassState, JointPositionCommand, JointState, Pose
from hybrid_control.rl.qp_pool import views


class _SnapshotModel(RobotModel):
    """The arm of env ``self.i``, read from the shared block (only what the QP controller asks for)."""

    def __init__(self, v, joint_names, limits):
        self.v, self.i, self._names = v, 0, tuple(joint_names)
        self._limits = (np.array(limits[0]), np.array(limits[1]))

    @property
    def joint_names(self):
        return self._names

    @property
    def position_limits(self):
        return self._limits

    def get_state(self):
        return JointState(0.0, self.v["q"][self.i], np.zeros(len(self._names)))

    def get_com_state(self):
        return CenterOfMassState(0.0, np.zeros(3))

    def get_pose(self, frame_name):
        return Pose(self.v["ee_pos"][self.i], self.v["ee_quat"][self.i])

    def get_jacobian(self, frame_name):
        return self.v["jac"][self.i].copy()

    def apply_command(self, command: JointPositionCommand):
        raise NotImplementedError


def main():
    cfg = json.loads(sys.argv[1])
    # OSQP prints to stdout on every solve; keep the real stdout for the protocol and send fd 1 to /dev/null.
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)

    shm = shared_memory.SharedMemory(name=cfg["shm"])
    try:  # the parent owns the block; stop this process's resource tracker from unlinking it (py < 3.13)
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    n_dof = len(cfg["joint_names"])
    v = views(shm.buf, cfg["num_envs"], n_dof)
    model = _SnapshotModel(v, cfg["joint_names"], cfg["limits"])
    ik = cfg["ik"]
    ik["velocity_lower"], ik["velocity_upper"] = np.array(ik["velocity_lower"]), np.array(ik["velocity_upper"])
    controller = InverseKinematicsController(model, IKQPConfig(**ik))
    start, stop, dt, frame = cfg["start"], cfg["stop"], cfg["dt"], cfg["frame"]

    out.write(b"r\n")  # ready
    for line in sys.stdin.buffer:
        if line.startswith(b"q"):
            break
        for i in range(start, stop):
            v["failed"][i] = 0.0
            v["q_cmd"][i] = v["q"][i]
            if not v["active"][i]:
                continue
            model.i = i
            try:
                target = PoseTarget(frame, Pose(v["target"][i], v["ori"][i]))
                v["q_cmd"][i] = controller.compute(target, dt).positions
            except RuntimeError:  # OSQP failure: hold position
                v["failed"][i] = 1.0
        out.write(b"d\n")
    del v
    shm.close()


if __name__ == "__main__":
    main()
