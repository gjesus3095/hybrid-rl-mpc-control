import math

import pytest

torch = pytest.importorskip("torch")

from hybrid_control.rl.observations import ObservationCfg, Snapshot, com_offset_norm, observe, to_base_frame
from hybrid_control.rl.rewards import RewardCfg, compute_rewards
from hybrid_control.rl.trajectories import ReferenceTrajectories, TrajectoryCfg


def make_trajs(n=16, **cfg):
    torch.manual_seed(0)
    trajs = ReferenceTrajectories(n, "cpu", TrajectoryCfg(**cfg))
    start = torch.tensor([0.5, 0.0, 0.5]).expand(n, 3)
    trajs.reset(torch.arange(n), start)
    return trajs, start


def test_trajectory_starts_at_initial_position():
    trajs, start = make_trajs()
    assert torch.allclose(trajs.position(torch.zeros(16)), start, atol=1e-6)


def test_trajectory_reaches_reference_and_fixed_points_stay():
    trajs, _ = make_trajs(p_fixed=1.0)
    late = trajs.position(torch.full((16,), 10.0))
    assert torch.allclose(late, trajs.centre, atol=1e-6)
    assert torch.allclose(trajs.position(torch.full((16,), 20.0)), late, atol=1e-6)


def test_all_trajectories_are_reachable_from_the_spawn_pose():
    n = 400
    torch.manual_seed(1)
    trajs = ReferenceTrajectories(n, "cpu", TrajectoryCfg())
    trajs.reset(torch.arange(n), torch.tensor([0.5, 0.0, 0.5]).expand(n, 3))
    cfg = trajs.cfg
    shoulder = torch.tensor(cfg.shoulder)
    for t in torch.linspace(cfg.blend_time, cfg.horizon, 80):  # after the blend-in the raw trajectory is followed
        p = trajs.position(torch.full((n,), float(t)))
        d = (p - shoulder).norm(dim=-1)
        assert (d <= cfg.reach + 1e-4).all() and (d >= cfg.min_radius - 1e-4).all()
        assert (p[:, 2] >= cfg.min_z - 1e-4).all()
    assert (trajs.amplitude.norm(dim=-1) > 0).float().mean() > 0.4  # most are real motions, not fallbacks to points


def test_trajectory_stays_in_amplitude_box_and_difficulty_zero_is_fixed():
    trajs, _ = make_trajs(p_fixed=0.0, difficulty=0.0)
    assert torch.all(trajs.amplitude == 0)
    trajs, _ = make_trajs(p_fixed=0.0, difficulty=1.0)
    ts = torch.linspace(5.0, 25.0, 50)
    pos = torch.stack([trajs.position(torch.full((16,), float(t))) for t in ts])
    assert torch.all((pos - trajs.centre).abs() <= trajs.amplitude + 1e-5)


def test_lookahead_matches_position():
    trajs, _ = make_trajs()
    t = torch.full((16,), 1.0)
    look = trajs.lookahead(t, torch.tensor([0.0, 0.5, 2.0]))
    assert look.shape == (16, 3, 3)
    assert torch.allclose(look[:, 1], trajs.position(t + 0.5), atol=1e-6)


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
        com_arm=z(n, 3), ee_pos=z(n, 3), arm_q=z(n, 7), arm_qd=z(n, 7), ref=z(n, k, 3), traj_centre=z(n, 3),
        half_size=torch.full((n, 2), 0.2), params=z(n, 6), last_action=z(n, 3), time_fraction=z(n),
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


def reward(rho, terminated=False, **kw):
    n = len(rho)
    z = torch.zeros(n)
    args = dict(
        roll=z, pitch=z, centre_dist=z, centre_dist_prev=z, dt=1 / 30, base_vel=torch.zeros(n, 2), yaw_rate=z, yaw=z,
        track_err=z, action=torch.zeros(n, 3), last_action=torch.zeros(n, 3),
    )
    args.update(kw)
    return compute_rewards(RewardCfg(), rho, terminated=torch.full((n,), terminated), **args)


def test_reward_prefers_centred_com_and_penalizes_termination():
    r, _ = reward(torch.tensor([0.0, 0.5, 1.0]))
    assert r[0] > r[1] > r[2]
    r_ok, _ = reward(torch.tensor([1.0]))
    r_fail, _ = reward(torch.tensor([1.0]), terminated=True)
    assert r_fail < r_ok - 5.0


def test_reward_centre_deadzone_and_action_rate():
    near, _ = reward(torch.zeros(1), centre_dist=torch.tensor([0.05]), centre_dist_prev=torch.tensor([0.05]))
    mid, _ = reward(torch.zeros(1), centre_dist=torch.tensor([0.3]), centre_dist_prev=torch.tensor([0.3]))
    far, _ = reward(torch.zeros(1), centre_dist=torch.tensor([1.5]), centre_dist_prev=torch.tensor([1.5]))
    assert torch.allclose(near, mid) and far < mid
    smooth, _ = reward(torch.zeros(1), action=torch.ones(1, 3), last_action=torch.ones(1, 3))
    jerky, _ = reward(torch.zeros(1), action=torch.ones(1, 3), last_action=-torch.ones(1, 3))
    assert jerky < smooth


def test_progress_rewards_approach_and_telescopes_over_a_loop():
    far, nearer = torch.tensor([1.0]), torch.tensor([0.99])
    _, toward = reward(torch.zeros(1), centre_dist=nearer, centre_dist_prev=far)
    _, away = reward(torch.zeros(1), centre_dist=far, centre_dist_prev=nearer)
    assert toward["progress"] > 0 > away["progress"]
    # Going out and coming back to the same distance sums to zero (before clamping): no reward for circling.
    _, a = reward(torch.zeros(1), centre_dist=torch.tensor([0.9]), centre_dist_prev=torch.tensor([0.9 - 0.01]))
    _, b = reward(torch.zeros(1), centre_dist=torch.tensor([0.9 - 0.01]), centre_dist_prev=torch.tensor([0.9]))
    assert torch.allclose(a["progress"] + b["progress"], torch.zeros(1), atol=1e-6)
    inside, _ = reward(torch.zeros(1), centre_dist=torch.tensor([0.2]), centre_dist_prev=torch.tensor([0.3]))
    assert reward(torch.zeros(1), centre_dist=torch.tensor([0.2]), centre_dist_prev=torch.tensor([0.3]))[1]["progress"] == 0


def test_motion_penalties_punish_speed_spin_and_yaw_drift():
    base, _ = reward(torch.zeros(1))
    fast, _ = reward(torch.zeros(1), base_vel=torch.tensor([[0.5, 0.0]]))
    spinning, _ = reward(torch.zeros(1), yaw_rate=torch.tensor([0.6]))
    turned, _ = reward(torch.zeros(1), yaw=torch.tensor([0.6]))
    assert fast < base and spinning < base and turned < base
    assert spinning < fast  # spinning is penalized harder than moving at the same relative speed


def test_tracking_reward_falls_with_end_effector_error():
    good, t_good = reward(torch.zeros(1), track_err=torch.tensor([0.01]))
    bad, t_bad = reward(torch.zeros(1), track_err=torch.tensor([0.20]))
    assert t_good["track"] > t_bad["track"] > 0 and good > bad
