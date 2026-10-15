"""Per-env arm QP solves for ``QPPool``, and the worker process that runs them on a slice of envs.

``make_solver`` builds the solver of one controller type for envs ``start:stop``; ``solve(v)`` reads that slice of
the arrays ``v`` (see ``qp_pool.fields``) and writes ``command`` and ``failed``. Run as a module it is a ``QPPool``
worker; not meant to be run by hand.
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import resource_tracker, shared_memory

import numpy as np

from hybrid_control.utils.control_utils import IKQPConfig, TaskSpaceTarget
from hybrid_control.robots.model import ControlMode, RobotModel
from hybrid_control.robots.state import CenterOfMassState, JointState, Pose
from hybrid_control.rl.policy_with_controller.qp_pool import views


def _pose(row: np.ndarray) -> Pose:
    return Pose(row[:3], row[3:])


class _PinkSlice:
    """``PinkInverseKinematicsController`` over envs ``start:stop``; inactive/failed envs hold ``q``."""

    def __init__(self, cfg: dict, start: int, stop: int):
        from hybrid_control.controllers.ik_qp.ik_qp import PinkInverseKinematicsController
        from hybrid_control.controllers.ik_qp.parallel import ArrayRobotModel

        ik = dict(cfg["ik"])
        ik["velocity_lower"], ik["velocity_upper"] = np.array(ik["velocity_lower"]), np.array(ik["velocity_upper"])
        self.robot = ArrayRobotModel(cfg["joint_names"], cfg["root_frame"])
        self.controller = PinkInverseKinematicsController(
            self.robot, IKQPConfig(**ik), root_frame=cfg["root_frame"], solver=cfg["qp_solver"],
        )
        self.frame, self.dt, self.start, self.stop = cfg["frame"], cfg["dt"], start, stop

    def solve(self, v: dict[str, np.ndarray]):
        from pink.exceptions import NoSolutionFound

        robot = self.robot
        robot.q, robot.limits, robot.root_pose = v["q"], v["limits"], v["root_pose"]
        for i in range(self.start, self.stop):
            v["failed"][i] = 0.0
            v["command"][i] = v["q"][i]
            if not v["active"][i]:
                continue
            robot.env = i
            try:
                v["command"][i] = self.controller.compute(TaskSpaceTarget(self.frame, _pose(v["target"][i])), self.dt).positions
            except NoSolutionFound:  # hold position
                v["failed"][i] = 1.0


class _ArrayTorqueModel(RobotModel):
    """Torque-mode ``RobotModel`` view of env ``self.i`` in the pool arrays."""

    @property
    def control_mode(self) -> ControlMode:
        return "torque"

    def __init__(self, v, joint_names, frame, torque_limit):
        super().__init__()
        self.v, self.i, self._names, self.frame = v, 0, tuple(joint_names), frame
        self._torque_limits = (-np.array(torque_limit), np.array(torque_limit))

    @property
    def joint_names(self):
        return self._names

    @property
    def position_limits(self):
        return self.v["limits"][self.i, :, 0], self.v["limits"][self.i, :, 1]

    @property
    def torque_limits(self):
        return self._torque_limits

    def get_state(self):
        return JointState(0.0, self.v["q"][self.i], self.v["dq"][self.i])

    def get_com_state(self):
        return CenterOfMassState(0.0, np.zeros(3))

    def get_pose(self, frame_name):
        self._check(frame_name)
        return _pose(self.v["ee_pose"][self.i])

    def get_jacobian(self, frame_name):
        self._check(frame_name)
        return self.v["jac"][self.i]

    def get_jacobian_derivative(self, frame_name):
        self._check(frame_name)
        return self.v["djac"][self.i]

    def get_mass_matrix(self):
        return self.v["mass"][self.i]

    def get_nonlinear_forces(self):
        return self.v["h"][self.i]

    def apply_command(self, command):
        raise NotImplementedError

    def _check(self, frame_name):
        if frame_name != self.frame:
            raise KeyError(f"only the end-effector frame {self.frame!r} is available, got {frame_name!r}")


class _ImpedanceSlice:
    """One ``ImpedanceController`` per env in ``start:stop`` (each keeps its own warm start and posture state).

    Inactive and failed envs get ``h`` (gravity/Coriolis compensation), as in the impedance demo.
    """

    def __init__(self, cfg: dict, start: int, stop: int):
        from hybrid_control.controllers.impedance.impedance_control import ImpedanceController, ImpedanceQPConfig

        qp = {k: np.array(x) if isinstance(x, list) else x for k, x in cfg["impedance"].items()}
        self.v = None
        self.robots = {}
        self.controllers = {}
        for i in range(start, stop):
            robot = _ArrayTorqueModel(None, cfg["joint_names"], cfg["frame"], cfg["torque_limit"])
            robot.i = i
            self.robots[i] = robot
            self.controllers[i] = ImpedanceController(robot, ImpedanceQPConfig(**qp))
        self.frame, self.dt, self.start, self.stop = cfg["frame"], cfg["dt"], start, stop

    def solve(self, v: dict[str, np.ndarray]):
        from hybrid_control.controllers.impedance.impedance_control import TaskSpaceTarget

        for i in range(self.start, self.stop):
            self.robots[i].v = v
            if v["reset"][i]:
                self.controllers[i].reset()
            v["failed"][i] = 0.0
            v["command"][i] = v["h"][i]
            if not v["active"][i]:
                continue
            target = TaskSpaceTarget(self.frame, _pose(v["target"][i]), v["target_vel"][i], v["target_acc"][i])
            try:
                v["command"][i] = self.controllers[i].compute(target, self.dt).torques
            except RuntimeError:  # OSQP failure: gravity/Coriolis compensation only
                v["failed"][i] = 1.0


def make_solver(controller: str, cfg: dict, start: int, stop: int):
    if controller == "ik_qp":
        return _PinkSlice(cfg, start, stop)
    if controller == "impedance":
        return _ImpedanceSlice(cfg, start, stop)
    raise ValueError(f"unknown controller {controller!r}")


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
    v = views(shm.buf, cfg["controller"], cfg["num_envs"], cfg["n_dof"])
    solver = make_solver(cfg["controller"], cfg["solver"], cfg["start"], cfg["stop"])

    out.write(b"r\n")  # ready
    for line in sys.stdin.buffer:
        if line.startswith(b"q"):
            break
        solver.solve(v)
        out.write(b"d\n")
    del v, solver
    shm.close()


if __name__ == "__main__":
    main()
