import numpy as np
import pytest
import subprocess
import sys

pytest.importorskip("pink")

from hybrid_control.controllers.impedance.impedance_control import (  # noqa: E402
    ImpedanceController, ImpedanceQPConfig, TaskSpaceTarget,
)
from hybrid_control.utils.control_utils import IKQPConfig  # noqa: E402
from hybrid_control.controllers.ik_qp.ik_qp import PinkInverseKinematicsController  # noqa: E402
from hybrid_control.controllers.ik_qp.parallel import ArrayRobotModel  # noqa: E402
from hybrid_control.rl.policy_with_controller.qp_pool import QPPool  # noqa: E402
from hybrid_control.robots.state import Pose  # noqa: E402
from hybrid_control.rl.policy_with_controller.qp_worker import _ArrayTorqueModel  # noqa: E402

N_ENVS, N_DOF = 7, 7
JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
LIMITS = np.array([[-2.9, 2.9]] * N_DOF)
DT = 1.0 / 60.0
ACTIVE = np.array([1, 1, 0, 1, 1, 1, 0], dtype=float)
QUAT = np.array([0.0, 0.0, 0.0, 1.0])

IK = {"velocity_lower": [-2.0] * N_DOF, "velocity_upper": [2.0] * N_DOF, "kp_pos": 10.0, "kp_ori": 5.0,
      "position_weight": 1.0, "orientation_weight": 1.0, "damping": 1e5}
IK_CFG = {"joint_names": JOINTS, "frame": "panda_hand", "root_frame": "panda_link0", "dt": DT, "qp_solver": "osqp",
          "ik": IK}

TAU = [87.0] * 4 + [12.0] * 3
IMPEDANCE = {"acceleration_lower": [-15.0] * N_DOF, "acceleration_upper": [15.0] * N_DOF, "kp_pos": 400.0,
             "kp_ori": 200.0, "posture_kp": 0.0, "posture_kd": 0.0, "nullspace_posture": [0.0] * N_DOF}
IMPEDANCE_CFG = {"joint_names": JOINTS, "frame": "tcp", "dt": DT, "torque_limit": TAU, "impedance": IMPEDANCE}


def test_worker_module_is_discoverable_without_editable_install(monkeypatch, tmp_path):
    """The pool must supply the source root after moving into policy_with_controller/."""
    popen = subprocess.Popen

    def checked_popen(args, **kwargs):
        # -S excludes site-packages/editable installs; use the worker's actual PYTHONPATH.
        with popen(
            [sys.executable, "-S", "-c",
             "import importlib.util, sys; assert importlib.util.find_spec(sys.argv[1])", args[2]],
            env=kwargs["env"], cwd=tmp_path,
        ) as probe:
            assert probe.wait(timeout=10) == 0
        return popen(args, **kwargs)

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(subprocess, "Popen", checked_popen)
    pool = QPPool("ik_qp", 1, N_DOF, IK_CFG, 1)
    pool.close()


def _run(pool, fill):
    try:
        fill(pool.v)
        first = pool.solve()
        fill(pool.v)
        second = pool.solve()  # a second request works too
    finally:
        pool.close()
    return first, second


@pytest.mark.parametrize("workers", [0, 3])
def test_ik_pool_matches_serial_pink_controller_and_respects_active_mask(workers):
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, (N_ENVS, N_DOF)) + np.array([0, 0, 0, -1.5, 0, 1.9, 0.8])
    root = np.tile(np.r_[0.1, 0.2, 0.3, QUAT], (N_ENVS, 1))
    target = np.c_[rng.uniform(0.3, 0.6, (N_ENVS, 3)), np.tile(QUAT, (N_ENVS, 1))]

    def fill(v):
        v["q"][:], v["limits"][:], v["root_pose"][:], v["target"][:], v["active"][:] = q, LIMITS, root, target, ACTIVE

    (cmd, failures), (cmd_again, _) = _run(QPPool("ik_qp", N_ENVS, N_DOF, IK_CFG, workers), fill)
    assert failures == 0
    assert np.allclose(cmd, cmd_again)
    robot = ArrayRobotModel(JOINTS, "panda_link0")
    robot.q, robot.limits, robot.root_pose = q, np.tile(LIMITS, (N_ENVS, 1, 1)), root
    serial = PinkInverseKinematicsController(robot, IKQPConfig(**{k: np.array(x) if isinstance(x, list) else x
                                                                   for k, x in IK.items()}), solver="osqp")
    for i in range(N_ENVS):
        if not ACTIVE[i]:
            assert np.array_equal(cmd[i], q[i])
            continue
        robot.env = i
        expected = serial.compute(TaskSpaceTarget("panda_hand", Pose(target[i, :3], target[i, 3:])), DT).positions
        assert np.allclose(cmd[i], expected, atol=1e-6)


@pytest.mark.parametrize("workers", [0, 3])
def test_impedance_pool_matches_serial_controller_and_falls_back_to_h(workers):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(N_ENVS, N_DOF, N_DOF))
    data = {
        "q": rng.uniform(-1, 1, (N_ENVS, N_DOF)), "dq": rng.uniform(-0.2, 0.2, (N_ENVS, N_DOF)),
        "ee_pose": np.c_[rng.uniform(0.3, 0.8, (N_ENVS, 3)), np.tile(QUAT, (N_ENVS, 1))],
        "jac": rng.normal(size=(N_ENVS, 6, N_DOF)), "djac": 0.1 * rng.normal(size=(N_ENVS, 6, N_DOF)),
        "mass": a @ a.transpose(0, 2, 1) + N_DOF * np.eye(N_DOF), "h": rng.uniform(-5, 5, (N_ENVS, N_DOF)),
        "limits": np.tile(LIMITS, (N_ENVS, 1, 1)), "target_vel": 0.1 * rng.normal(size=(N_ENVS, 6)),
        "target_acc": 0.1 * rng.normal(size=(N_ENVS, 6)), "active": ACTIVE,
    }
    data["target"] = data["ee_pose"] + np.c_[rng.uniform(-0.05, 0.05, (N_ENVS, 3)), np.zeros((N_ENVS, 4))]

    def fill(v):
        for k, x in data.items():
            v[k][:] = x
        v["reset"][:] = 1.0

    (cmd, failures), (cmd_again, _) = _run(QPPool("impedance", N_ENVS, N_DOF, IMPEDANCE_CFG, workers), fill)
    assert failures == 0
    assert np.allclose(cmd, cmd_again, atol=1e-5)  # reset flag -> same (cold-start) solve
    config = ImpedanceQPConfig(**{k: np.array(x) if isinstance(x, list) else x for k, x in IMPEDANCE.items()})
    for i in range(N_ENVS):
        if not ACTIVE[i]:
            assert np.array_equal(cmd[i], data["h"][i])
            continue
        robot = _ArrayTorqueModel(data, JOINTS, "tcp", TAU)
        robot.i = i
        target = TaskSpaceTarget("tcp", Pose(data["target"][i, :3], QUAT), data["target_vel"][i], data["target_acc"][i])
        expected = ImpedanceController(robot, config).compute(target, DT).torques
        assert np.allclose(cmd[i], expected, atol=1e-5)

