"""Run the per-env arm QPs (Pink IK or impedance) in parallel worker processes (NumPy only, no Isaac/torch imports).

``QPPool`` holds one set of per-env input/output arrays (``QPPool.v``) whose fields depend on the controller
(``FIELDS``). The caller fills the inputs, calls ``solve()`` and reads ``command`` (joint position targets for
``ik_qp``, joint torques for ``impedance``). Each env's solve is done by ``qp_worker.make_solver``, so the serial
path (``workers=0``) and the parallel path give the same results.

Workers are plain subprocesses (``python -m hybrid_control.rl.qp_worker``) rather than ``multiprocessing`` children:
they never fork a process that holds CUDA/Kit state and never re-import the training script, so the pool can be
created after the simulator has started. Inputs and outputs travel through one shared-memory block; a one-line
message over the worker's pipes signals each solve.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

CONTROLLERS = ("ik_qp", "impedance")


def fields(controller: str, n_dof: int) -> dict[str, tuple[int, ...]]:
    """Per-env field name -> shape (without the env axis). Poses are (position, xyzw quaternion), world frame."""
    common = {"q": (n_dof,), "limits": (n_dof, 2), "target": (7,), "active": (), "reset": ()}
    if controller == "ik_qp":
        specific = {"root_pose": (7,)}
    elif controller == "impedance":
        specific = {
            "dq": (n_dof,), "ee_pose": (7,), "jac": (6, n_dof), "djac": (6, n_dof), "mass": (n_dof, n_dof),
            "h": (n_dof,), "target_vel": (6,), "target_acc": (6,),
        }
    else:
        raise ValueError(f"controller must be one of {CONTROLLERS}, got {controller!r}")
    return {**common, **specific, "command": (n_dof,), "failed": ()}


def layout(controller: str, num_envs: int, n_dof: int) -> tuple[dict[str, tuple[int, tuple[int, ...]]], int]:
    """Field name -> (float64 offset, shape) inside the shared block, and the block size in float64."""
    out, offset = {}, 0
    for name, shape in fields(controller, n_dof).items():
        shape = (num_envs, *shape)
        out[name] = (offset, shape)
        offset += int(np.prod(shape))
    return out, offset


def views(buf, controller: str, num_envs: int, n_dof: int) -> dict[str, np.ndarray]:
    spec, total = layout(controller, num_envs, n_dof)
    flat = np.ndarray((total,), dtype=np.float64, buffer=buf)
    return {k: flat[o : o + int(np.prod(s))].reshape(s) for k, (o, s) in spec.items()}


class QPPool:
    def __init__(self, controller: str, num_envs: int, n_dof: int, solver_cfg: dict, workers: int):
        """``solver_cfg`` is the JSON-serializable controller setup read by ``qp_worker.make_solver``."""
        from .qp_worker import make_solver

        self.controller, self.num_envs, self.n_dof = controller, num_envs, n_dof
        _, size = layout(controller, num_envs, n_dof)
        self._shm, self._procs, self._local = None, [], None
        if workers <= 0:  # solve in this process
            self.v = views(bytearray(8 * size), controller, num_envs, n_dof)
            self._local = make_solver(controller, solver_cfg, 0, num_envs)
            return
        self._shm = shared_memory.SharedMemory(create=True, size=8 * size)
        self.v = views(self._shm.buf, controller, num_envs, n_dof)
        for arr in self.v.values():
            arr[...] = 0.0
        workers = max(1, min(workers, num_envs))
        cfg = {"shm": self._shm.name, "controller": controller, "num_envs": num_envs, "n_dof": n_dof,
               "solver": solver_cfg}
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        src = str(Path(__file__).resolve().parents[2])  # make `hybrid_control` importable in the workers
        env["PYTHONPATH"] = os.pathsep.join([src] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        for chunk in np.array_split(np.arange(num_envs), workers):
            wcfg = dict(cfg, start=int(chunk[0]), stop=int(chunk[-1]) + 1)
            self._procs.append(subprocess.Popen(
                [sys.executable, "-m", "hybrid_control.rl.qp_worker", json.dumps(wcfg)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
            ))
        for p in self._procs:  # wait until every worker has attached and imported
            self._read(p)

    def _read(self, proc):
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"QP worker exited unexpectedly (code {proc.poll()})")
        return line

    def solve(self) -> tuple[np.ndarray, int]:
        """Solve every active env from the inputs in ``self.v``. Returns ``(command (N, n), number of failed solves)``.

        Inactive and failed envs get the controller's fallback (hold ``q`` for ``ik_qp``, gravity/Coriolis
        compensation ``h`` for ``impedance``). ``reset`` flags are consumed (cleared) by the solve.
        """
        v = self.v
        if self._local is not None:
            self._local.solve(v)
        else:
            for p in self._procs:
                p.stdin.write(b"s\n")
                p.stdin.flush()
            for p in self._procs:
                self._read(p)
        v["reset"][:] = 0.0
        return v["command"].copy(), int(v["failed"].sum())

    def close(self):
        for p in self._procs:
            try:
                p.stdin.write(b"q\n")
                p.stdin.flush()
                p.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        for p in self._procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self._procs = []
        if self._shm is not None:
            self.v = {}
            self._shm.close()
            self._shm.unlink()
            self._shm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
