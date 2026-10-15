import math

import pytest

torch = pytest.importorskip("torch")

from hybrid_control.rl.policy_with_controller.observations import ObservationCfg, Snapshot, com_offset_norm, observe, to_base_frame
from hybrid_control.rl.policy_with_controller.rewards import RewardCfg, compute_rewards
from hybrid_control.rl.policy_with_controller.trajectories import ReferenceTrajectories, TrajectoryCfg


def make_trajs(n=16, **cfg):
    torch.manual_seed(0)
    trajs = ReferenceTrajectories(n, "cpu", TrajectoryCfg(**cfg))
    start = torch.tensor([0.61, 0.0, 0.635]).expand(n, 3)  # panda_hand at spawn in Isaac
    trajs.reset(torch.arange(n), start)
    return trajs, start


def test_trajectory_starts_at_initial_position():
    trajs, start = make_trajs()
    assert torch.allclose(trajs.position(torch.zeros(16)), start, atol=1e-6)


def test_trajectory_reaches_reference_and_fixed_points_stay():
    trajs, _ = make_trajs(p_fixed=1.0, p_sine=0.0, p_quintic=0.0)
    late = trajs.position(torch.full((16,), 10.0))
    assert torch.allclose(late, trajs.centre, atol=1e-6)
    assert torch.allclose(trajs.position(torch.full((16,), 20.0)), late, atol=1e-6)


def test_all_trajectories_are_reachable_from_the_spawn_pose():
    n = 400
    torch.manual_seed(1)
    trajs = ReferenceTrajectories(n, "cpu", TrajectoryCfg(p_quintic=0.0))
    trajs.reset(torch.arange(n), torch.tensor([0.5, 0.0, 0.5]).expand(n, 3))
    cfg = trajs.cfg
    shoulder = torch.tensor(cfg.shoulder)
    for t in torch.linspace(cfg.blend_time, cfg.duration, 80):  # after the blend-in the raw trajectory is followed
        p = trajs.position(torch.full((n,), float(t)))
        d = (p - shoulder).norm(dim=-1)
        assert (d <= cfg.reach + 1e-4).all() and (d >= cfg.min_radius - 1e-4).all()
        assert (p[:, 2] >= cfg.min_z - 1e-4).all()
    assert (trajs.amplitude.norm(dim=-1) > 0).float().mean() > 0.4  # most are real motions, not fallbacks to points


def test_sines_are_redrawn_until_reachable_never_replaced_by_a_point():
    from hybrid_control.rl.policy_with_controller.trajectories import SINE

    # a thin reach shell: most first draws fail, yet every env keeps a reachable sine
    trajs, _ = make_trajs(n=200, p_fixed=0.0, p_sine=1.0, p_quintic=0.0, min_radius=0.55, reach=0.65)
    assert (trajs.kind == SINE).all()
    assert trajs.reachable(trajs.centre, trajs.amplitude, trajs.frequency, trajs.phase).all()
    assert (trajs.amplitude.norm(dim=-1) > 0).all()
    with pytest.raises(RuntimeError):  # an empty region is reported instead of looping forever
        make_trajs(n=4, p_fixed=0.0, p_sine=1.0, p_quintic=0.0, min_radius=0.7, reach=0.6, max_draws=20)


def test_trajectory_stays_in_amplitude_box_and_difficulty_zero_is_fixed():
    trajs, _ = make_trajs(p_fixed=0.0, p_quintic=0.0, difficulty=0.0)
    assert torch.all(trajs.amplitude == 0)
    trajs, _ = make_trajs(p_fixed=0.0, p_quintic=0.0, difficulty=1.0)
    ts = torch.linspace(5.0, 25.0, 50)
    pos = torch.stack([trajs.position(torch.full((16,), float(t))) for t in ts])
    assert torch.all((pos - trajs.centre).abs() <= trajs.amplitude + 1e-5)


def test_lookahead_matches_position():
    trajs, _ = make_trajs()
    t = torch.full((16,), 1.0)
    look = trajs.lookahead(t, torch.tensor([0.0, 0.5, 2.0]))
    assert look.shape == (16, 3, 3)
    assert torch.allclose(look[:, 1], trajs.position(t + 0.5), atol=1e-6)


SPAWN_QUAT = torch.nn.functional.normalize(torch.tensor([0.92388, -0.38268, 0.0, 0.0]), dim=0)  # TCP orientation != Q_DOWN


def make_quintic(n=500, seed=2, **cfg):
    from hybrid_control.rl.policy_with_controller.trajectories import QUINTIC

    torch.manual_seed(seed)
    trajs = ReferenceTrajectories(n, "cpu", TrajectoryCfg(p_fixed=0.0, p_sine=0.0, p_quintic=1.0, **cfg))
    start = torch.tensor([0.61, 0.0, 0.635]).expand(n, 3)  # panda_hand at spawn in Isaac
    trajs.reset(torch.arange(n), start, SPAWN_QUAT.expand(n, 4))
    assert (trajs.kind == QUINTIC).all()  # no fallbacks to fixed points
    return trajs, start


def quat_dist(a, b):
    return 1.0 - (a * b).sum(-1).abs()  # 0 for the same rotation (q and -q included)


def test_torch_quintic_matches_numpy_quintic_trajectory():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from hybrid_control.robots.state import Pose
    from hybrid_control.utils.trajectories import QuinticTrajectory

    trajs, _ = make_quintic(n=8)
    t0, t1 = trajs.cfg.segment_time
    for i in range(8):
        w, q = trajs.waypoints[i].double().numpy(), trajs.waypoint_quat[i].double().numpy()
        segments = [(QuinticTrajectory(Pose(w[0], q[0]), Pose(w[1], q[1]), t0), 0.0),
                    (QuinticTrajectory(Pose(w[1], q[1]), Pose(w[2], q[2]), t1), t0)]
        for seg, offset in segments:
            for tau in np.linspace(0.0, seg.duration * 0.999, 7):
                ref = seg.sample(tau)
                pos, quat = trajs.pose(torch.full((8,), offset + tau))
                assert np.allclose(pos[i].numpy(), ref.pose.position, atol=1e-5)
                assert quat_dist(quat[i].double(), torch.tensor(ref.pose.quaternion)) < 1e-5


def test_quintic_passes_through_waypoints_and_holds_b():
    trajs, start = make_quintic()
    n, (t0, t1) = len(start), trajs.cfg.segment_time
    for t, k in ((0.0, 0), (t0, 1), (t0 + t1, 2), (t0 + t1 + 3.0, 2)):
        pos, quat = trajs.pose(torch.full((n,), t))
        assert torch.allclose(pos, trajs.waypoints[:, k], atol=1e-5)
        assert (quat_dist(quat, trajs.waypoint_quat[:, k]) < 1e-5).all()
    assert torch.allclose(trajs.waypoints[:, 0], start) and (quat_dist(trajs.waypoint_quat[:, 0], SPAWN_QUAT) < 1e-5).all()


def test_quintic_segments_satisfy_all_constraints_on_a_fine_grid():
    trajs, _ = make_quintic()
    cfg, n = trajs.cfg, trajs.num_envs
    centre = torch.tensor(cfg.arm_base)
    pos = trajs.lookahead(torch.zeros(n), torch.linspace(0.0, sum(cfg.segment_time), 2000))  # (n, 2000, 3)
    d, d_xy = (pos - centre).norm(dim=-1), (pos - centre)[..., :2].norm(dim=-1)
    assert (d_xy >= cfg.inner_radius_xy - 1e-5).all() and (d <= cfg.outer_radius + 1e-5).all()
    assert (pos[..., 2] >= cfg.quintic_min_z - 1e-5).all()


def test_segment_valid_matches_dense_sampling():
    trajs = ReferenceTrajectories(1, "cpu", TrajectoryCfg())
    cfg = trajs.cfg
    torch.manual_seed(3)
    centre = torch.tensor(cfg.arm_base)
    p0 = centre + 0.5 * torch.randn(4000, 3)
    p1 = torch.where(torch.rand(4000, 1) < 0.5, 2 * centre - p0 + 0.05 * torch.randn(4000, 3), centre + 0.5 * torch.randn(4000, 3))
    s = torch.linspace(0.0, 1.0, 1000)[None, :, None]
    pts = p0[:, None] + s * (p1 - p0)[:, None]
    d, d_xy = (pts - centre).norm(dim=-1), (pts - centre)[..., :2].norm(dim=-1)
    dense = (d_xy >= cfg.inner_radius_xy).all(1) & (d <= cfg.outer_radius).all(1) & (pts[..., 2] >= cfg.quintic_min_z).all(1)
    exact = trajs.segment_valid(p0, p1)
    assert (exact & ~dense).sum() == 0  # never accepts a segment that leaves the region
    assert (exact != dense).float().mean() < 1e-3  # disagreements only within the grid spacing of the inner ball
    assert dense.any() and (~dense).any()


def test_quintic_orientations_point_down_within_the_tilt_limit():
    from hybrid_control.utils.torch_rotations import quat_apply

    trajs, _ = make_quintic(tilt_max_deg=25.0)
    z = quat_apply(trajs.waypoint_quat[:, 1:].reshape(-1, 4), torch.tensor([0.0, 0.0, 1.0]).expand(2 * trajs.num_envs, 3))
    tilt = torch.rad2deg(torch.acos((-z[:, 2]).clamp(-1.0, 1.0)))
    assert tilt.max() <= 25.0 + 1e-3 and tilt.max() > 15.0


def make_sines(n=300, **cfg):
    torch.manual_seed(5)
    trajs = ReferenceTrajectories(n, "cpu", TrajectoryCfg(p_fixed=0.0, p_sine=1.0, p_quintic=0.0, **cfg))
    trajs.reset(torch.arange(n), torch.tensor([0.61, 0.0, 0.635]).expand(n, 3), SPAWN_QUAT.expand(n, 4))
    return trajs


def test_sine_orientation_starts_at_spawn_and_oscillates_about_a_down_centre():
    from hybrid_control.utils.torch_rotations import quat_angle, quat_apply

    trajs = make_sines()
    n, cfg = trajs.num_envs, trajs.cfg
    quat0 = trajs.pose(torch.zeros(n))[1]
    assert (quat_dist(quat0, SPAWN_QUAT) < 1e-5).all()  # no orientation jump at the start
    z = quat_apply(trajs.centre_quat, torch.tensor([0.0, 0.0, 1.0]).expand(n, 3))
    assert (torch.rad2deg(torch.acos((-z[:, 2]).clamp(-1, 1))) <= cfg.tilt_max_deg + 1e-3).all()  # centre points down
    assert (trajs.ori_amplitude <= math.radians(cfg.ori_amplitude_deg) + 1e-6).all() and (trajs.ori_amplitude > 0).any()
    # after the blend: q(t) = exp(theta(t)) q_centre, so its angle to the centre is |theta(t)| <= |amplitude|
    quat = trajs.lookahead_pose(torch.zeros(n), torch.linspace(cfg.blend_time, cfg.duration, 200))[1]
    angle = quat_angle(quat, trajs.centre_quat[:, None].expand_as(quat))
    assert (angle <= trajs.ori_amplitude.norm(dim=-1)[:, None] + 1e-4).all()
    assert (angle.max(dim=1).values > 1e-3).float().mean() > 0.9  # the orientation actually moves
    full = trajs.lookahead_pose(torch.zeros(n), torch.linspace(0.0, cfg.duration, 1001))[1]
    assert ((full[:, 1:] * full[:, :-1]).sum(-1) > 0.99).all()  # continuous, no sign flips


def test_fixed_points_hold_and_difficulty_zero_sines_hold_their_centre_orientation():
    trajs, _ = make_trajs(n=32, p_fixed=1.0, p_sine=0.0, p_quintic=0.0)
    quat = trajs.lookahead_pose(torch.zeros(32), torch.linspace(0.0, 5.0, 50))[1]
    assert torch.allclose(quat, trajs.start_quat[:, None].expand_as(quat), atol=1e-6)
    trajs = make_sines(difficulty=0.0)
    assert (trajs.ori_amplitude == 0).all()
    quat = trajs.lookahead_pose(torch.zeros(trajs.num_envs), torch.linspace(trajs.cfg.blend_time, 5.0, 20))[1]
    assert (quat_dist(quat, trajs.centre_quat[:, None].expand_as(quat)) < 1e-5).all()


def test_kind_mix():
    from hybrid_control.rl.policy_with_controller.trajectories import FIXED, QUINTIC, SINE

    trajs, _ = make_trajs(n=3000, p_fixed=0.2, p_sine=0.4, p_quintic=0.4)
    frac = [(trajs.kind == k).float().mean().item() for k in (FIXED, SINE, QUINTIC)]
    assert all(abs(f - p) < 0.05 for f, p in zip(frac, (0.2, 0.4, 0.4)))


def test_failing_segments_are_redrawn_not_turned_into_fixed_points():
    # a narrow shell: most first draws fail, yet every env ends with a valid quintic trajectory
    trajs, _ = make_quintic(n=300, inner_radius_xy=0.30, outer_radius=0.55)  # spawn: 0.31 m in x-y, 0.47 m in 3D
    cfg = trajs.cfg
    assert trajs.segment_valid(trajs.waypoints[:, 0], trajs.waypoints[:, 1]).all()
    assert trajs.segment_valid(trajs.waypoints[:, 1], trajs.waypoints[:, 2]).all()
    assert ((trajs.waypoints[:, 1:, :2] - torch.tensor(cfg.arm_base[:2])).norm(dim=-1) >= 0.30 - 1e-6).all()
    # a spawn pose inside the cylinder can never start a valid segment: report it instead of looping forever
    with pytest.raises(RuntimeError):
        make_trajs(n=4, p_fixed=0.0, p_sine=0.0, p_quintic=1.0, inner_radius_xy=0.5, max_draws=20)


def test_inner_limit_is_a_vertical_cylinder_not_a_sphere():
    trajs = ReferenceTrajectories(1, "cpu", TrajectoryCfg())
    base = torch.tensor(trajs.cfg.arm_base)
    above = base + torch.tensor([0.05, 0.0, 0.4])  # 0.40 m from the arm base in 3D but only 0.05 m from its axis
    beside = base + torch.tensor([0.25, 0.0, 0.2])
    assert not trajs.segment_valid(above[None], above[None]).item()
    assert trajs.segment_valid(beside[None], beside[None]).item()
    # a segment passing over the arm base crosses the cylinder although both ends are outside it
    p0, p1 = base + torch.tensor([0.3, 0.0, 0.3]), base + torch.tensor([-0.3, 0.0, 0.3])
    assert not trajs.segment_valid(p0[None], p1[None]).item()


def test_lookahead_past_the_end_holds_the_final_pose():
    for kind_cfg in (dict(p_fixed=0.0, p_sine=1.0, p_quintic=0.0), dict(p_fixed=0.0, p_sine=0.0, p_quintic=1.0)):
        trajs, _ = make_trajs(n=64, **kind_cfg)
        end = trajs.cfg.duration
        t = torch.full((64,), end - 0.5)
        pos, quat = trajs.lookahead_pose(t, torch.tensor([0.5, 1.0, 1.5]))
        final_pos, final_quat = trajs.pose(torch.full((64,), end))
        assert torch.allclose(pos, final_pos[:, None].expand_as(pos), atol=1e-6)
        assert torch.allclose(quat, final_quat[:, None].expand_as(quat), atol=1e-6)


@pytest.mark.parametrize("spawn_sign", [1.0, -1.0])
def test_reference_quaternion_is_continuous_for_either_spawn_sign(spawn_sign):
    torch.manual_seed(4)
    n = 500
    trajs = ReferenceTrajectories(n, "cpu", TrajectoryCfg(p_fixed=0.0, p_sine=0.0, p_quintic=1.0))
    trajs.reset(torch.arange(n), torch.tensor([0.61, 0.0, 0.635]).expand(n, 3), (spawn_sign * SPAWN_QUAT).expand(n, 4))
    quat = trajs.lookahead_pose(torch.zeros(n), torch.linspace(0.0, trajs.cfg.duration, 1001))[1]  # (n, 1001, 4)
    assert ((quat[:, 1:] * quat[:, :-1]).sum(-1) > 0.99).all()  # no sign flip anywhere, incl. the segment boundary


def test_to_base_frame_rotates_about_z():
    yaw = torch.tensor([math.pi / 2])
    out = to_base_frame(yaw, torch.tensor([[1.0, 0.0, 0.7]]))
    assert torch.allclose(out, torch.tensor([[0.0, -1.0, 0.7]]), atol=1e-6)


def make_snapshot(n=4, k=5, com_x=0.0):
    z = lambda *s: torch.zeros(*s)  # noqa: E731
    com = z(n, 3)
    com[:, 0] = com_x
    return Snapshot(
        yaw=z(n), roll=z(n), pitch=z(n), centre_xy=z(n, 2), lin_vel_w=z(n, 3), ang_vel_w=z(n, 3), com_total=com,
        com_arm=z(n, 3), ee_pos=z(n, 3), ref=z(n, k, 3), ref_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(n, k, 4),
        # arm_q=z(n, 7), arm_qd=z(n, 7), traj_centre=z(n, 3), time_fraction=z(n),
        half_size=torch.full((n, 2), 0.2), params=z(n, 6), last_action=z(n, 3),
    )


def test_com_offset_norm_is_fraction_of_half_footprint():
    rho = com_offset_norm(make_snapshot(com_x=0.1))
    assert torch.allclose(rho, torch.full((4,), 0.5))


def test_observation_shapes_and_privileged_params():
    snap = make_snapshot()
    actor, critic = observe(snap, ObservationCfg(physical_params_in_actor=True))
    actor2, critic2 = observe(snap, ObservationCfg(physical_params_in_actor=False))
    assert actor.shape[0] == critic.shape[0] == 4
    assert actor2.shape[1] == actor.shape[1] - 6  # actor loses the 6 physical params...
    assert critic2.shape[1] == critic.shape[1]  # ...but the critic keeps them
    assert torch.isfinite(actor).all() and torch.isfinite(critic).all()


def test_com_offset_norm_moves_between_actor_and_critic_only():
    snap = make_snapshot(com_x=0.1)
    actor, critic = observe(snap, ObservationCfg(com_offset_in_actor=True))
    actor_old, critic_old = observe(snap, ObservationCfg(com_offset_in_actor=False))
    assert actor.shape[1] == actor_old.shape[1] + 1  # older runs: actor without rho
    assert critic.shape[1] == critic_old.shape[1]  # the critic sees rho once either way
    assert torch.allclose(actor[:, -1], torch.full((4,), 0.5))


def test_saved_observation_cfg_defaults_old_runs_to_no_com_offset(tmp_path):
    from hybrid_control.rl.config import saved_observation_cfg

    (tmp_path / "params").mkdir()
    ckpt = str(tmp_path / "model_0.pt")
    (tmp_path / "params" / "env.yaml").write_text("observation:\n  lookahead: [0.0, 0.5]\n  physical_params_in_actor: true\n")
    assert saved_observation_cfg(ckpt) == ObservationCfg(lookahead=(0.0, 0.5), com_offset_in_actor=False,
                                                         ref_orientation_in_actor=False)
    (tmp_path / "params" / "env.yaml").write_text("observation:\n  com_offset_in_actor: true\n  ref_orientation_in_actor: true\n")
    assert saved_observation_cfg(ckpt).com_offset_in_actor and saved_observation_cfg(ckpt).ref_orientation_in_actor


def test_reference_orientation_observation():
    from hybrid_control.rl.policy_with_controller.observations import observation_terms

    snap = make_snapshot(k=5)
    actor, critic = observe(snap, ObservationCfg(ref_orientation_in_actor=True))
    actor_old, critic_old = observe(snap, ObservationCfg(ref_orientation_in_actor=False))
    assert actor.shape[1] == actor_old.shape[1] + 30 and critic.shape[1] == critic_old.shape[1] + 30
    terms, _ = observation_terms(snap, ObservationCfg())
    assert torch.allclose(terms["ref_orientation"][0, :6], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))  # identity
    # a base yawed by +90 deg sees a world-aligned reference rotated by -90 deg: x axis -> (0, -1, 0)
    snap.yaw = torch.full((4,), math.pi / 2)
    terms, _ = observation_terms(snap, ObservationCfg())
    assert torch.allclose(terms["ref_orientation"][0, :6], torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0]), atol=1e-6)


def reward(rho, terminated=False, cfg=None, **kw):
    n = len(rho)
    z = torch.zeros(n)
    args = dict(
        roll=z, pitch=z, centre_dist=z, centre_dist_prev=z, dt=1 / 30, base_vel=torch.zeros(n, 2), yaw_rate=z, yaw=z,
        track_err=z, action=torch.zeros(n, 3), last_action=torch.zeros(n, 3),
    )
    args.update(kw)
    return compute_rewards(cfg or RewardCfg(), rho, terminated=torch.full((n,), terminated), **args)


def test_reward_prefers_centred_com_and_penalizes_termination():
    r, _ = reward(torch.tensor([0.6, 0.8, 1.0]))
    assert r[0] > r[1] > r[2]
    r_ok, _ = reward(torch.tensor([1.0]))
    r_fail, _ = reward(torch.tensor([1.0]), terminated=True)
    assert r_fail < r_ok - 5.0


def test_com_reward_is_constant_inside_deadzone():
    _, terms = reward(torch.tensor([0.0, 0.3, 0.6, 0.61]))
    com = terms["com"]
    assert torch.allclose(com[:3], com[2].expand(3)) and com[3] < com[2]
    assert torch.allclose(com[2], torch.tensor(2.0), atol=1e-4)  # plateau scaled to the former maximum


def test_disabled_terms_are_logged_with_zero_weight():
    _, terms = reward(torch.zeros(1), roll=torch.tensor([0.3]), centre_dist=torch.tensor([1.5]),
                      centre_dist_prev=torch.tensor([1.6]), yaw=torch.tensor([0.6]), action=torch.ones(1, 3))
    for name in ("tilt", "centre", "progress", "yaw", "action"):
        assert terms[name].abs().sum() == 0


ENABLED = RewardCfg(w_centre=0.5, w_progress=1.0, w_yaw=-0.3)  # the disabled terms, with their former weights


def test_reward_centre_deadzone_and_action_rate():
    cfg = ENABLED
    near, _ = reward(torch.zeros(1), cfg=cfg, centre_dist=torch.tensor([0.05]), centre_dist_prev=torch.tensor([0.05]))
    mid, _ = reward(torch.zeros(1), cfg=cfg, centre_dist=torch.tensor([0.3]), centre_dist_prev=torch.tensor([0.3]))
    far, _ = reward(torch.zeros(1), cfg=cfg, centre_dist=torch.tensor([1.5]), centre_dist_prev=torch.tensor([1.5]))
    assert torch.allclose(near, mid) and far < mid
    smooth, _ = reward(torch.zeros(1), action=torch.ones(1, 3), last_action=torch.ones(1, 3))
    jerky, _ = reward(torch.zeros(1), action=torch.ones(1, 3), last_action=-torch.ones(1, 3))
    assert jerky < smooth


def test_progress_rewards_approach_and_telescopes_over_a_loop():
    far, nearer = torch.tensor([1.0]), torch.tensor([0.99])
    _, toward = reward(torch.zeros(1), cfg=ENABLED, centre_dist=nearer, centre_dist_prev=far)
    _, away = reward(torch.zeros(1), cfg=ENABLED, centre_dist=far, centre_dist_prev=nearer)
    assert toward["progress"] > 0 > away["progress"]
    # Going out and coming back to the same distance sums to zero (before clamping): no reward for circling.
    _, a = reward(torch.zeros(1), cfg=ENABLED, centre_dist=torch.tensor([0.9]), centre_dist_prev=torch.tensor([0.9 - 0.01]))
    _, b = reward(torch.zeros(1), cfg=ENABLED, centre_dist=torch.tensor([0.9 - 0.01]), centre_dist_prev=torch.tensor([0.9]))
    assert torch.allclose(a["progress"] + b["progress"], torch.zeros(1), atol=1e-6)
    inside, _ = reward(torch.zeros(1), cfg=ENABLED, centre_dist=torch.tensor([0.2]), centre_dist_prev=torch.tensor([0.3]))
    assert reward(torch.zeros(1), cfg=ENABLED, centre_dist=torch.tensor([0.2]), centre_dist_prev=torch.tensor([0.3]))[1]["progress"] == 0


def test_motion_penalties_punish_speed_spin_and_yaw_drift():
    base, _ = reward(torch.zeros(1))
    base_enabled, _ = reward(torch.zeros(1), cfg=ENABLED)
    fast, _ = reward(torch.zeros(1), base_vel=torch.tensor([[0.5, 0.0]]))
    spinning, _ = reward(torch.zeros(1), yaw_rate=torch.tensor([0.6]))
    turned, _ = reward(torch.zeros(1), cfg=ENABLED, yaw=torch.tensor([0.6]))
    assert fast < base and spinning < base and turned < base_enabled
    assert spinning <= fast  # spinning at the yaw-rate scale costs at least as much as moving at the speed scale


def test_tracking_reward_falls_with_end_effector_error():
    good, t_good = reward(torch.zeros(1), track_err=torch.tensor([0.01]))
    bad, t_bad = reward(torch.zeros(1), track_err=torch.tensor([0.20]))
    assert t_good["track"] > t_bad["track"] > 0 and good > bad


def test_reward_term_info_matches_compute_rewards():
    from hybrid_control.rl.policy_with_controller.rewards import RewardCfg, reward_term_info

    _, terms = reward(torch.zeros(3))
    assert [name for name, _, _ in reward_term_info(RewardCfg())] == list(terms)


def test_observation_terms_concatenate_to_observe():
    from hybrid_control.rl.policy_with_controller.observations import observation_terms

    snap = make_snapshot()
    actor, critic = observe(snap, ObservationCfg())
    actor_terms, critic_terms = observation_terms(snap, ObservationCfg())
    assert torch.equal(actor, torch.cat(list(actor_terms.values()), dim=-1))
    assert torch.equal(critic, torch.cat(list(critic_terms.values()), dim=-1))
