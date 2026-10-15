"""ParallelPinkIK must reproduce PinkInverseKinematicsController env by env, for any worker count."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pin = pytest.importorskip("pinocchio")
pytest.importorskip("pink")
pytest.importorskip("daqp")

from hybrid_control.utils.control_utils import IKQPConfig, TaskSpaceTarget
from hybrid_control.controllers.ik_qp.ik_qp import PinkInverseKinematicsController
from hybrid_control.controllers.ik_qp.parallel import ParallelPinkIK
from hybrid_control.robots.state import Pose
from test_ik_qp_pink import ARM_JOINTS, FRAME, Q_HOME, PinocchioRobot

DT = 0.02
CONFIG = IKQPConfig(np.full(7, -0.5), np.full(7, 0.5), kp_pos=50.0, kp_ori=100.0, damping=1e5)
NUM_ENVS = 13  # not a multiple of the worker counts


def batch(rng):
    """Robots on different bases and their batched inputs; some targets are far enough to hit bounds."""
    robots, targets = [], []
    for _ in range(NUM_ENVS):
        base = pin.SE3(Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix(), np.r_[rng.normal(size=2), 0.3])
        robot = PinocchioRobot(Q_HOME + rng.uniform(-0.2, 0.2, 7), base)
        lower, upper = robot.position_limits
        robot = PinocchioRobot(np.clip(robot.get_state().positions, lower + 1e-3, upper - 1e-3), base)
        pose = robot.get_pose(FRAME)
        scale = rng.choice([1e-3, 5e-2])
        rotation = Rotation.from_rotvec(rng.normal(size=3) * scale) * Rotation.from_quat(np.array(pose.quaternion))
        robots.append(robot)
        targets.append(Pose(pose.position + rng.normal(size=3) * scale, rotation.as_quat()))
    q = np.array([r.get_state().positions for r in robots])
    limits = np.array([np.stack(r.position_limits, axis=1) for r in robots])
    root = np.array([np.r_[r.get_pose("panda_link0").position, r.get_pose("panda_link0").quaternion] for r in robots])
    target = np.array([np.r_[t.position, t.quaternion] for t in targets])
    return robots, targets, (q, limits, root, target)


@pytest.mark.parametrize("workers", [0, 1, 3, 20])
def test_matches_sequential_controller(workers):
    robots, targets, inputs = batch(np.random.default_rng(0))
    expected = np.array([PinkInverseKinematicsController(r, CONFIG, solver="daqp").compute(TaskSpaceTarget(FRAME, t), DT)
                         .positions for r, t in zip(robots, targets)])
    with ParallelPinkIK(workers) as pool:
        pool.configure(ARM_JOINTS, CONFIG, solver="daqp")
        q_cmd, ok = pool.compute(*inputs, FRAME, DT)
    assert ok.all()
    np.testing.assert_allclose(q_cmd, expected, rtol=0, atol=1e-12)
    assert np.any(np.isclose(np.abs(q_cmd - inputs[0]), 0.5 * DT))  # some velocity bounds active


def test_worker_errors_are_raised():
    with ParallelPinkIK(2) as pool:
        with pytest.raises(RuntimeError, match="IK worker failed"):
            pool.configure(("not_a_joint",) * 7, CONFIG, solver="daqp")


def test_workers_exit_when_parent_dies_without_cleanup():
    """Isaac Sim ends the process with os._exit, so close() never runs; the workers must still go."""
    import os
    import subprocess
    import sys
    import time

    code = ("import os\nfrom hybrid_control.controllers.ik_qp.parallel import ParallelPinkIK\n"
            "pool = ParallelPinkIK(3)\nprint(*(p.pid for p in pool._procs), flush=True)\nos._exit(0)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60)
    pids = [int(pid) for pid in out.stdout.split()]
    assert len(pids) == 3

    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    deadline = time.monotonic() + 5
    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(alive(pid) for pid in pids)
