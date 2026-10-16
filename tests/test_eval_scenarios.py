import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hybrid_control.rl.evaluation.metrics import failure_steps, valid_mask
from hybrid_control.rl.evaluation.scenarios import (
    REF_DT, SET_NAMES, SPAWN_POS, SPAWN_POSES, SPAWN_QUAT, Benchmark, BenchmarkCfg, generate_benchmark, rounded_polyline,
    trapezoid_progress,
)
from hybrid_control.rl.evaluation.scripted import ScriptedTrajectories

ARM_BASE = np.array(BenchmarkCfg().trajectory.arm_base)


@pytest.fixture(scope="module")
def benchmark():
    return generate_benchmark(BenchmarkCfg(n_per_set=30))


def test_benchmark_is_deterministic(benchmark):
    assert generate_benchmark(BenchmarkCfg(n_per_set=30)).fingerprint() == benchmark.fingerprint()
    assert generate_benchmark(BenchmarkCfg(n_per_set=30, seed=1)).fingerprint() != benchmark.fingerprint()


def test_save_load_roundtrip(benchmark, tmp_path):
    benchmark.save(tmp_path / "b.npz")
    loaded = Benchmark.load(tmp_path / "b.npz")
    assert loaded.fingerprint() == benchmark.fingerprint()
    assert loaded.cfg == benchmark.cfg
    assert loaded.sets["rigid"].info["path_type"].tolist() == benchmark.sets["rigid"].info["path_type"].tolist()


def test_every_scenario_starts_at_the_spawn_pose_and_has_dynamics(benchmark):
    for name in SET_NAMES:
        s = benchmark.sets[name]
        assert len(s) == 30 and s.poses.shape[2] == 7
        assert np.allclose(s.poses[:, 0, :3], SPAWN_POS, atol=1e-5)
        assert np.allclose(np.abs((s.poses[:, 0, 3:] * np.array(SPAWN_QUAT)).sum(-1)), 1.0, atol=1e-5)
        assert np.allclose(np.linalg.norm(s.poses[..., 3:], axis=-1), 1.0, atol=1e-4)
        assert np.all(s.dynamics[:, 0] > 0) and np.all(s.dynamics[:, 2] >= 0)
    p = benchmark.params("poly", 30)
    assert [x.footprint for x in p] == [tuple(map(float, f)) for f in benchmark.footprints]  # slot j = footprint j


def test_far_targets_are_outside_the_training_reach_and_inside_two_metres(benchmark):
    s = benchmark.sets["poly_far"]
    b = s.info["waypoints"][:, 2]
    radius = np.linalg.norm(b[:, :2] - ARM_BASE[:2], axis=-1)
    assert radius.min() >= 0.9 > benchmark.cfg.trajectory.outer_radius and radius.max() <= 2.0
    assert b[:, 2].min() >= benchmark.cfg.far_z[0] - 1e-6 and b[:, 2].max() <= benchmark.cfg.far_z[1] + 1e-6
    # A -> B takes max(2.5 s, L / ref_speed): the mean speed never exceeds the training one
    t_ab, length = s.info["segment_time"][:, 1], s.info["ab_length"]
    assert np.all(t_ab >= 2.5 - 1e-6) and np.all(length / t_ab <= benchmark.cfg.ref_speed + 1e-6)
    assert np.allclose(s.duration, 2.5 + t_ab + benchmark.cfg.hold_time, atol=1e-5)
    assert np.allclose(s.poses[np.arange(30), -1, :3], b, atol=1e-5)  # B is reached and held
    # the straight A -> B never goes through the robot's own body
    a = s.info["waypoints"][:, 1]
    for i in range(30):
        u = np.linspace(0, 1, 200)[:, None]
        xy = a[i, :2] + u * (b[i, :2] - a[i, :2])
        assert np.linalg.norm(xy - ARM_BASE[:2], axis=-1).min() >= benchmark.cfg.trajectory.inner_radius_xy - 1e-4


def test_rigid_paths_are_straight_lines_with_right_angle_bends_inside_the_workspace(benchmark):
    s = benchmark.sets["rigid"]
    cfg = benchmark.cfg
    assert set(s.info["path_type"].tolist()) == {0, 1, 2}
    for i in range(len(s)):
        v = s.info["vertices"][i]
        v = v[~np.isnan(v).any(1)]
        assert len(v) == s.info["path_type"][i] + 2
        legs = np.diff(v, axis=0)
        legs /= np.linalg.norm(legs, axis=1, keepdims=True)
        assert np.allclose((legs[:-1] * legs[1:]).sum(1), 0.0, atol=1e-6)  # 90 degree bends
        pos = s.poses[i, : int(s.duration[i] / REF_DT) + 1, :3]
        d = np.linalg.norm(pos - ARM_BASE, axis=-1)
        assert d.max() <= cfg.trajectory.outer_radius + 1e-3 and pos[:, 2].min() >= cfg.trajectory.quintic_min_z - 1e-3
        assert np.linalg.norm(pos[:, :2] - ARM_BASE[:2], axis=-1).min() >= cfg.trajectory.inner_radius_xy - 1e-3
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=-1) / REF_DT
        assert speed.max() <= s.info["speed"][i] * 1.02  # constant cruise speed, no overshoot
        assert np.allclose(s.poses[i, :, 3:], SPAWN_QUAT, atol=1e-6)  # orientation held


def test_rounded_polyline_keeps_the_straight_legs_and_rounds_the_corner():
    v = np.array([[0.0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
    dense = rounded_polyline(v, 0.05)
    assert np.allclose(dense[0], v[0]) and np.allclose(dense[-1], v[-1])
    assert np.abs(np.linalg.norm(np.diff(dense, axis=0), axis=-1) - 1e-3).max() < 1e-3 + 1e-9  # evenly spaced
    corner_gap = np.linalg.norm(dense - v[1], axis=-1).min()
    assert corner_gap == pytest.approx(0.05 * (np.sqrt(2) - 1), abs=2e-3)  # a fillet cuts the corner by r (sqrt2 - 1)
    straight = dense[(dense[:, 0] < 0.4)]
    assert np.allclose(straight[:, 1:], 0.0)


def test_trapezoid_progress_accelerates_cruises_and_stops():
    t = np.linspace(0, 4, 4001)
    s, total = trapezoid_progress(t, 0.6, 0.3, 0.5)
    assert s[0] == 0 and s[-1] == pytest.approx(0.6)
    v = np.diff(s) / np.diff(t)
    assert v.max() == pytest.approx(0.3, abs=1e-3) and np.all(np.diff(s) >= -1e-12)
    assert total == pytest.approx(0.6 / 0.3 + 0.3 / 0.5)  # L / v + v / a


def test_scripted_trajectories_interpolate_the_table(benchmark):
    s = benchmark.sets["poly_far"]
    traj = ScriptedTrajectories(30, "cpu")
    traj.load(s)
    t = torch.full((30,), 3.1234)
    pos, quat = traj.pose(t)
    k = 3.1234 / REF_DT
    i0 = int(k)
    expect = s.poses[:, i0, :3] + (k - i0) * (s.poses[:, i0 + 1, :3] - s.poses[:, i0, :3])
    assert torch.allclose(pos, torch.from_numpy(expect).float(), atol=1e-5)
    assert torch.allclose(quat.norm(dim=-1), torch.ones(30), atol=1e-5)
    late = traj.position(torch.full((30,), 1e3))
    assert torch.allclose(late, torch.from_numpy(s.poses[:, -1, :3]), atol=1e-6)  # held past the end
    ahead = traj.lookahead(torch.zeros(30), torch.tensor([0.0, 0.5, 1.0]))
    assert ahead.shape == (30, 3, 3)
    # reset keeps the scenarios and only checks the spawn pose
    traj.reset(torch.arange(30), torch.tensor(SPAWN_POS).expand(30, 3), torch.tensor(SPAWN_QUAT).expand(30, 4))
    with pytest.raises(ValueError, match="spawn pose"):
        traj.reset(torch.arange(30), torch.tensor(SPAWN_POS).expand(30, 3) + 0.05)


def test_failure_steps_and_valid_mask():
    flags = np.zeros((3, 10), bool)
    flags[0, 4:] = True  # fails at step 4
    flags[1, 8:] = True  # fails after its own 6 steps: not a failure
    n_steps = np.array([10, 6, 10])
    fail = failure_steps(flags, n_steps)
    assert fail.tolist() == [4, -1, -1]
    valid = valid_mask(n_steps, fail, 10)
    assert valid[0].sum() == 5 and valid[1].sum() == 6 and valid[2].sum() == 10


def test_impedance_benchmark_starts_at_the_fingertip_pose_with_the_same_platforms_and_dynamics():
    ik = generate_benchmark(BenchmarkCfg(n_per_set=10))
    imp = generate_benchmark(BenchmarkCfg(n_per_set=10, controller="impedance"))
    assert imp.fingerprint() != ik.fingerprint()
    for name in SET_NAMES:
        assert np.allclose(imp.sets[name].poses[:, 0, :3], SPAWN_POSES["impedance"][0], atol=1e-5)
        assert np.allclose(imp.sets[name].dynamics, ik.sets[name].dynamics)
    assert np.allclose(imp.footprints, ik.footprints)
    b = imp.sets["poly_far"].info["waypoints"][:, 2]  # the far-B constraints do not depend on the start
    assert np.linalg.norm(b[:, :2] - ARM_BASE[:2], axis=-1).min() >= 0.9


def test_ellipse_loops_are_centred_on_the_arm_base_inside_the_workspace_at_training_speeds(benchmark):
    s = benchmark.sets["ellipse"]
    cfg = benchmark.cfg
    theta = s.info["theta"]
    assert np.allclose(np.abs(theta[:, 1]), np.pi)  # half revolutions only: none goes all the way around the base
    assert set(np.sign(theta[:, 1])) == {-1.0, 1.0}  # both directions
    assert np.allclose(s.centre[:, :2], ARM_BASE[:2])
    a, b, _, z0, z_amp, _ = s.info["shape"].T
    assert a.min() >= cfg.ellipse_semi_axes[0] - 1e-6 and b.max() <= cfg.ellipse_semi_axes[1] + 1e-6
    for i in range(len(s)):
        entry = s.info["entry_time"][i]
        pos = s.poses[i, : int(s.duration[i] / REF_DT) + 1, :3]
        loop = pos[int(np.ceil(entry / REF_DT)):]
        d = np.linalg.norm(pos - ARM_BASE, axis=-1)
        assert d.max() <= cfg.trajectory.outer_radius + 1e-3 and pos[:, 2].min() >= cfg.trajectory.quintic_min_z - 1e-3
        assert np.linalg.norm(pos[:, :2] - ARM_BASE[:2], axis=-1).min() >= cfg.trajectory.inner_radius_xy - 1e-3
        # on the ellipse: (x'/a)^2 + (y'/b)^2 = 1 in the ellipse frame
        psi = s.info["shape"][i, 2]
        c, sn = np.cos(psi), np.sin(psi)
        x, y = loop[:, 0] - ARM_BASE[0], loop[:, 1] - ARM_BASE[1]
        assert np.allclose(((c * x + sn * y) / a[i]) ** 2 + ((-sn * x + c * y) / b[i]) ** 2, 1.0, atol=2e-3)
        speed = np.linalg.norm(np.diff(loop, axis=0), axis=-1) / REF_DT
        assert speed.max() <= s.info["speed"][i] * 1.02 and s.info["speed"][i] <= cfg.ellipse_speed[1]
        assert (loop[:, 2].max() - loop[:, 2].min()) <= 2 * z_amp[i] + 1e-3  # the height oscillation is bounded by its amplitude
    assert np.allclose(s.poses[:, 0, :3], SPAWN_POS, atol=1e-5)
    assert np.all(s.duration > 7.0) and np.all(s.duration < 25.0)


def test_adding_sets_does_not_change_the_fingerprint_of_the_older_ones(benchmark):
    older = tuple(n for n in SET_NAMES if n != "ellipse")
    assert benchmark.fingerprint(older) != benchmark.fingerprint()
    assert benchmark.fingerprint(older) == generate_benchmark(BenchmarkCfg(n_per_set=30)).fingerprint(older)


@pytest.mark.parametrize("order", [2, 4])
def test_poly_sets_follow_bezier_paths_of_the_set_order(order):
    from hybrid_control.rl.evaluation.scenarios import QUINTIC, _far_set, _training_set
    from hybrid_control.rl.policy_with_controller.trajectories import bezier_eval

    cfg = BenchmarkCfg.with_spatial_order(order, n_per_set=12)
    tc = cfg.trajectory
    for name, data in (("poly", _training_set(cfg, QUINTIC, 7, 12)), ("poly_far", _far_set(cfg, 7, 12))):
        control, wp = data["info"]["control_points"], data["info"]["waypoints"]
        assert control.shape == (12, 2, order + 1, 3), name
        assert np.allclose(control[:, :, 0], wp[:, :2], atol=1e-6) and np.allclose(control[:, :, -1], wp[:, 1:], atol=1e-6)
        dense = bezier_eval(torch.from_numpy(control[:, 1]), torch.linspace(0, 1, 400)[None].expand(12, -1)).numpy()
        straight = wp[:, 1:2] + np.linspace(0, 1, 400)[None, :, None] * (wp[:, 2:3] - wp[:, 1:2])
        assert np.linalg.norm(dense - straight, axis=-1).max() > 1e-3, name  # actually curved
        assert np.linalg.norm(dense[..., :2] - ARM_BASE[:2], axis=-1).min() >= tc.inner_radius_xy - 1e-4, name
        assert dense[..., 2].min() >= tc.quintic_min_z - 1e-4, name
        # the reference table runs along the curves and holds B
        seg = data["info"].get("segment_time", np.tile(tc.segment_time, (12, 1)))
        k = np.round((seg[:, 0] + 0.5 * seg[:, 1]) / REF_DT).astype(int)  # quintic s = 1/2 halfway through A -> B
        mid = bezier_eval(torch.from_numpy(control[:, 1]), torch.full((12, 1), 0.5)).numpy()[:, 0]
        assert np.allclose(data["poses"][np.arange(12), k, :3], mid, atol=2e-3), name
        assert np.allclose(data["poses"][:, -1, :3], wp[:, 2], atol=1e-5), name
    far = _far_set(cfg, 7, 12)["info"]
    assert np.all(far["ab_length"] > np.linalg.norm(far["waypoints"][:, 2] - far["waypoints"][:, 1], axis=-1))  # arc length


def test_spatial_order_one_keeps_the_straight_benchmark():
    straight, explicit = BenchmarkCfg(n_per_set=4), BenchmarkCfg.with_spatial_order(1, n_per_set=4)
    assert straight == explicit
    with pytest.raises(ValueError):
        BenchmarkCfg.with_spatial_order(0)
