"""Pink IK/QP for many environments, solved in parallel by CPU worker processes.

Each worker owns one ``PinkInverseKinematicsController`` and a contiguous slice of the environments,
which it solves sequentially. Per call the parent sends every worker its slice of the batched inputs
through a pipe, then collects the joint position commands; the slices are solved concurrently.
Processes rather than threads: Pink, the Pinocchio bindings and qpsolvers hold the GIL.

Workers are forked, so create the pool *before* the simulator starts: forking a running Isaac Sim
process (many threads, CUDA context) can deadlock. ``configure`` may follow later, e.g. once a TCP
offset read from USD is known. ``num_workers=0`` solves in the calling process (sequential baseline).
"""

import ctypes
import multiprocessing as mp
import os
import signal
import time
from collections.abc import Mapping, Sequence

import numpy as np

from hybrid_control.controllers.ik_qp.config import IKQPConfig, PoseTarget
from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import JointState, Pose


class ArrayRobotModel(RobotModel):
    """``RobotModel`` view of environment ``env`` in batched arrays: joint positions (N, n),
    position limits (N, n, 2) and the root-frame pose (N, 7, xyzw). Only what the Pink IK reads."""

    def __init__(self, joint_names: Sequence[str], root_frame: str):
        self._joint_names = tuple(joint_names)
        self.root_frame = root_frame
        self.env = 0
        self.q = self.limits = self.root_pose = None

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.limits[self.env, :, 0], self.limits[self.env, :, 1]

    def get_state(self) -> JointState:
        q = self.q[self.env]
        return JointState(0.0, q, np.zeros_like(q))

    def get_pose(self, frame_name: str) -> Pose:
        if frame_name != self.root_frame:
            raise ValueError(f"Only the root frame '{self.root_frame}' is available, not '{frame_name}'")
        pose = self.root_pose[self.env]
        return Pose(pose[:3], pose[3:])

    def get_com_state(self):
        raise NotImplementedError

    def get_jacobian(self, frame_name: str):
        raise NotImplementedError

    def apply_command(self, command):
        raise NotImplementedError


class _Solver:
    """One controller solving a slice of environments in turn (runs in a worker or in-process)."""

    def __init__(self, joint_names, config, root_frame, solver, extra_frames, urdf_path):
        from hybrid_control.controllers.ik_qp.ik_qp import DEFAULT_URDF, PinkInverseKinematicsController

        self.robot = ArrayRobotModel(joint_names, root_frame)
        self.controller = PinkInverseKinematicsController(
            self.robot, config, urdf_path=urdf_path or DEFAULT_URDF, root_frame=root_frame, solver=solver,
            extra_frames=extra_frames)

    def solve(self, q, limits, root_pose, target_pose, frame_name, dt):
        from pink.exceptions import NoSolutionFound

        t0 = time.perf_counter()
        robot = self.robot
        robot.q, robot.limits, robot.root_pose = q, limits, root_pose
        q_cmd, ok = q.copy(), np.ones(len(q), dtype=bool)
        for env in range(len(q)):
            robot.env = env
            target = PoseTarget(frame_name, Pose(target_pose[env, :3], target_pose[env, 3:]))
            try:
                q_cmd[env] = self.controller.compute(target, dt).positions
            except NoSolutionFound:  # caller decides what to hold; q_cmd stays at q
                ok[env] = False
        return q_cmd, ok, time.perf_counter() - t0


def _exit_with_parent() -> None:
    """Get SIGTERM when the parent dies: Isaac Sim exits without running Python cleanup (close())."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG (Linux)
    except (OSError, AttributeError):
        pass


def _worker(conn, cpu, inherited):
    _exit_with_parent()
    for other in inherited:  # earlier workers' parent ends: closing them lets those pipes report EOF
        other.close()
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    solver = None
    while True:
        try:
            message = conn.recv()
        except EOFError:  # parent gone
            break
        if message is None:
            break
        kind, payload = message
        try:
            if kind == "configure":
                solver = _Solver(*payload)
                conn.send(("ok", None))
            else:
                conn.send(("ok", solver.solve(*payload)))
        except Exception as error:  # report to the parent instead of dying silently
            conn.send(("error", repr(error)))
    conn.close()


class ParallelPinkIK:
    """Batched front end to ``PinkInverseKinematicsController`` with ``num_workers`` processes.

    ``cpus`` optionally pins worker i to CPU ``cpus[i]`` (e.g. one per physical core).
    """

    def __init__(self, num_workers: int, cpus: Sequence[int] | None = None):
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if cpus is not None and len(cpus) < num_workers:
            raise ValueError("Need one CPU per worker")
        self.num_workers = num_workers
        self._local = None
        self._conns, self._procs = [], []
        context = mp.get_context("fork")
        for i in range(num_workers):
            parent, child = context.Pipe()
            proc = context.Process(target=_worker, daemon=True,
                                   args=(child, None if cpus is None else cpus[i], tuple(self._conns) + (parent,)))
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self.worker_times = np.zeros(num_workers)  # solve time of each worker in the last call, s

    def configure(self, joint_names: Sequence[str], config: IKQPConfig, root_frame: str = "panda_link0",
                  solver: str = "daqp", extra_frames: Mapping[str, tuple[str, Pose]] | None = None,
                  urdf_path=None) -> None:
        payload = (tuple(joint_names), config, root_frame, solver, dict(extra_frames or {}), urdf_path)
        if not self.num_workers:
            self._local = _Solver(*payload)
            return
        for conn in self._conns:
            conn.send(("configure", payload))
        for conn in self._conns:
            self._receive(conn)

    def compute(self, q: np.ndarray, position_limits: np.ndarray, root_pose: np.ndarray,
                target_pose: np.ndarray, frame_name: str, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Batched ``PinkInverseKinematicsController.compute``.

        q (N, n); position_limits (N, n, 2); root_pose and target_pose (N, 7) world poses, xyzw.
        Returns position commands (N, n) and ok (N,) bool; failed envs (NoSolutionFound) get q.
        """
        q = np.asarray(q, dtype=float)
        args = (q, np.asarray(position_limits, dtype=float), np.asarray(root_pose, dtype=float),
                np.asarray(target_pose, dtype=float))
        if not self.num_workers:
            if self._local is None:
                raise RuntimeError("Call configure() first")
            q_cmd, ok, elapsed = self._local.solve(*args, frame_name, dt)
            self.worker_times = np.array([elapsed])
            return q_cmd, ok
        chunks = np.array_split(np.arange(len(q)), self.num_workers)
        busy = []
        for conn, idx in zip(self._conns, chunks):
            if len(idx):
                conn.send(("solve", (*(a[idx[0]:idx[-1] + 1] for a in args), frame_name, dt)))
                busy.append((conn, idx))
        q_cmd, ok = np.empty_like(q), np.empty(len(q), dtype=bool)
        self.worker_times = np.zeros(self.num_workers)
        for i, (conn, idx) in enumerate(busy):
            q_cmd[idx], ok[idx], self.worker_times[i] = self._receive(conn)
        return q_cmd, ok

    @staticmethod
    def _receive(conn):
        status, result = conn.recv()
        if status != "ok":
            raise RuntimeError(f"IK worker failed: {result}")
        return result

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._conns, self._procs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def physical_cpus() -> list[int]:
    """One logical CPU per physical core this process may run on (first hyperthread sibling)."""
    allowed, seen, cpus = os.sched_getaffinity(0), set(), []
    for cpu in sorted(allowed):
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                siblings = f.read().strip()
        except OSError:
            siblings = str(cpu)
        if siblings not in seen:
            seen.add(siblings)
            cpus.append(cpu)
    return cpus
