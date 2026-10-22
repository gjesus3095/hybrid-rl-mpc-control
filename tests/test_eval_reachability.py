import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pinocchio")

from hybrid_control.rl.evaluation.reachability import ArmReach
from hybrid_control.rl.evaluation.scenarios import REF_DT, SPAWN_POSES, BenchmarkCfg, generate_benchmark

CFG = BenchmarkCfg(n_per_set=20, controller="impedance")


@pytest.fixture(scope="module")
def reach():
    return ArmReach(CFG.spawn, CFG.trajectory.arm_base, CFG.reach)


def _table(positions, quat=SPAWN_POSES["impedance"][1]):
    return np.concatenate([positions, np.broadcast_to(quat, (len(positions), 4))], axis=1).astype(np.float32)


def test_holding_the_spawn_pose_is_reachable(reach):
    assert reach.trajectory_ok(_table(np.tile(SPAWN_POSES["impedance"][0], (30, 1))), REF_DT)


def test_a_point_two_metres_away_is_not_reachable(reach):
    spawn = np.array(SPAWN_POSES["impedance"][0])
    ramp = spawn + np.linspace(0.0, 1.0, 120)[:, None] * np.array([1.5, 0.0, 0.0])
    assert not reach.trajectory_ok(_table(ramp), REF_DT)


def test_a_jump_that_needs_too_fast_joints_is_rejected(reach):
    spawn = np.array(SPAWN_POSES["impedance"][0])
    jump = np.tile(spawn, (20, 1))
    jump[4:] += np.array([0.0, 0.25, 0.0])  # 25 cm within one 1/30 s step, the pose itself is reachable
    assert reach.trajectory_ok(_table(jump[:4]), REF_DT)
    assert not reach.trajectory_ok(_table(jump), REF_DT)


@pytest.mark.parametrize("controller", ["ik_qp", "impedance"])
def test_fixed_sine_and_rigid_sets_are_all_reachable_by_a_fixed_base(controller):
    cfg = BenchmarkCfg(n_per_set=20, controller=controller)
    benchmark = generate_benchmark(cfg)
    checker = ArmReach(cfg.spawn, cfg.trajectory.arm_base, cfg.reach)
    for name in ("fixed", "sine", "rigid"):
        s = benchmark.sets[name]
        for i in range(len(s)):
            assert checker.trajectory_ok(s.poses[i, : int(round(s.duration[i] / REF_DT)) + 1], REF_DT), (name, i)
