"""Run the per-env IK/QP solves in parallel worker processes (NumPy only, no Isaac/torch imports).

The unchanged ``InverseKinematicsController`` is used inside each worker, so results equal the serial loop. Workers
are plain subprocesses (``python -m hybrid_control.rl.qp_worker``) rather than ``multiprocessing`` children: they
never fork a process that holds CUDA/Kit state and never re-import the training script. Inputs and outputs travel
through one shared-memory block; a one-line message over the worker's pipes signals each solve.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from hybrid_control.controllers.ik_qp.controller import IKQPConfig

N_QUAT = 4


def layout(num_envs: int, n_dof: int) -> dict[str, tuple[int, tuple[int, ...]]]:
    """Field name -> (float64 offset, shape) inside the shared block."""
    fields = {
        "q": (num_envs, n_dof), "ee_pos": (num_envs, 3), "ee_quat": (num_envs, N_QUAT),
        "jac": (num_envs, 6, n_dof), "target": (num_envs, 3), "ori": (num_envs, N_QUAT), "active": (num_envs,),
        "q_cmd": (num_envs, n_dof), "failed": (num_envs,),
    }
    out, offset = {}, 0
    for name, shape in fields.items():
        out[name] = (offset, shape)
        offset += int(np.prod(shape))
    out["_size"] = (offset, ())
    return out


def views(buf, num_envs: int, n_dof: int) -> dict[str, np.ndarray]:
    spec = layout(num_envs, n_dof)
    total = spec.pop("_size")[0]
    flat = np.ndarray((total,), dtype=np.float64, buffer=buf)
    return {k: flat[o : o + int(np.prod(s))].reshape(s) for k, (o, s) in spec.items()}


class QPPool:
    def __init__(
        self, num_envs: int, joint_names: tuple[str, ...], limits: tuple[np.ndarray, np.ndarray], config: IKQPConfig,
        dt: float, ee_frame: str, workers: int,
    ):
        self.num_envs, self.n_dof = num_envs, len(joint_names)
        spec = layout(num_envs, self.n_dof)
        self._shm = shared_memory.SharedMemory(create=True, size=8 * spec["_size"][0])
        self.v = views(self._shm.buf, num_envs, self.n_dof)
        for arr in self.v.values():
            arr[...] = 0.0
        workers = max(1, min(workers, num_envs))
        cfg = {
            "shm": self._shm.name, "num_envs": num_envs, "joint_names": list(joint_names), "frame": ee_frame, "dt": dt,
            "limits": [np.asarray(limits[0]).tolist(), np.asarray(limits[1]).tolist()],
            "ik": {
                "velocity_lower": config.velocity_lower.tolist(), "velocity_upper": config.velocity_upper.tolist(),
                "kp_pos": config.kp_pos, "kp_ori": config.kp_ori, "position_weight": config.position_weight,
                "orientation_weight": config.orientation_weight, "damping": config.damping,
            },
        }
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        src = str(Path(__file__).resolve().parents[2])  # make `hybrid_control` importable in the workers
        env["PYTHONPATH"] = os.pathsep.join([src] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        self._procs = []
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

    def solve(self, q, ee_pos, ee_quat, jacobian, target, ori, active) -> tuple[np.ndarray, int]:
        """Solve every active env. Returns ``(q_cmd (N, n), number of failed solves)``; inactive/failed hold ``q``."""
        v = self.v
        v["q"][:], v["ee_pos"][:], v["ee_quat"][:], v["jac"][:] = q, ee_pos, ee_quat, jacobian
        v["target"][:], v["ori"][:], v["active"][:] = target, ori, active
        for p in self._procs:
            p.stdin.write(b"s\n")
            p.stdin.flush()
        for p in self._procs:
            self._read(p)
        return v["q_cmd"].copy(), int(v["failed"].sum())

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
