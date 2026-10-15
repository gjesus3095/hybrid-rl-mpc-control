"""CPU tests for the moving-mass task, with no Isaac imports."""
import math
from dataclasses import asdict
import pytest

torch = pytest.importorskip('torch')
from hybrid_control.rl.policy_no_controller.trajectories import JointSplines, TrajectoryCfg
from hybrid_control.rl.policy_no_controller.observations import Snapshot, ObservationCfg, observe, observation_terms
from hybrid_control.rl.policy_no_controller.rewards import RewardCfg, compute_rewards
from hybrid_control.sim_interfaces.isaac.moving_mass.params import MechanismCfg, MovingMassRandomization


def spline(n=4, seed=11, **kwargs):
    return JointSplines(n, 'cpu', 15, MechanismCfg(), TrajectoryCfg(**kwargs), seed)


def test_spline_bounds_derivatives_endpoints_and_continuity():
    s = spline()
    q0, v0, a0 = s.evaluate(torch.zeros(4))
    assert torch.allclose(q0, torch.tensor(MechanismCfg().initial).expand(4, -1))
    assert not v0.any() and not a0.any()
    for time in torch.linspace(0, 15, 201):
        q, v, a = s.evaluate(time.expand(4))
        assert (q >= s.lower-1e-6).all() and (q <= s.upper+1e-6).all()
        assert (v.abs() <= torch.tensor(s.cfg.max_velocity)+1e-5).all()
        assert (a.abs() <= torch.tensor(s.cfg.max_acceleration)+1e-5).all()
    # Sample sufficiently near each knot for the configured faster accelerations.
    for knot in s.knots.unique()[1:-1]:
        left = s.evaluate((knot-1e-6).expand(4))
        right = s.evaluate((knot+1e-6).expand(4))
        for x, y in zip(left, right):
            assert torch.allclose(x, y, atol=1e-4)
    end, ve, ae = s.evaluate(torch.full((4,), 15.0))
    after, va, aa = s.evaluate(torch.full((4,), 18.0))
    assert torch.equal(end, after) and not ve.any() and not ae.any() and not va.any() and not aa.any()
    assert (end-q0).abs().max() > 0.01


def test_spline_matches_independent_scipy_and_analytic_derivatives():
    scipy = pytest.importorskip('scipy.interpolate')
    s = spline()
    times = torch.tensor([1.1, 4.2, 8.3, 13.2])
    result = s.evaluate(times)
    for i in range(4):
        reference = scipy.BSpline(s.knots.numpy(), s.control[i].numpy(), 3)
        for derivative in range(3):
            assert torch.allclose(result[derivative][i], torch.tensor(reference(float(times[i]), nu=derivative), dtype=torch.float32), atol=1e-5)


def test_partial_resets_are_reproducible_and_independent():
    a, b = spline(), spline()
    before = a.control.clone()
    a.reset(torch.tensor([1, 3]))
    assert torch.equal(a.control[[0, 2]], before[[0, 2]])
    assert not torch.equal(a.control[1], before[1])
    b.reset(torch.tensor([3]))
    b.reset(torch.tensor([1]))
    assert torch.equal(a.control, b.control)
    a.reset(torch.tensor([1]))
    a.reset(torch.tensor([0]))
    b.reset(torch.tensor([0]))
    assert torch.equal(a.control[0], b.control[0])


def test_zero_difficulty_is_stationary():
    s = spline(difficulty=0)
    q, v, a = s.evaluate(torch.tensor([1., 4., 8., 14.]))
    assert torch.allclose(q, s.initial.expand(4, -1))
    assert not v.any() and not a.any()


def snapshot():
    z = lambda *s: torch.zeros(*s)
    return Snapshot(yaw=z(2), roll=z(2), pitch=z(2), centre_xy=z(2, 2), lin_vel_w=z(2, 3),
                    ang_vel_w=z(2, 3), com_total=z(2, 3), com_mechanism=z(2, 3),
                    joint_pos=z(2, 3), joint_vel=z(2, 3), half_size=torch.full((2, 2), .2),
                    params=z(2, 9), last_action=z(2, 3))


def test_observations_are_current_state_only_and_frame_consistent():
    s = snapshot()
    s.yaw[:] = math.pi/2
    s.com_total[:, 0] = .1
    actor, critic = observe(s, ObservationCfg())
    assert actor.shape == (2, 31) and critic.shape == (2, 34)
    terms, _ = observation_terms(s, ObservationCfg())
    assert torch.allclose(terms['com_total'][:, 1], torch.full((2,), -.1))
    assert not any('ref' in x or 'ee' in x for x in terms)
    hidden_actor, hidden_critic = observe(s, ObservationCfg(physical_params_in_actor=False, com_offset_in_actor=False))
    assert hidden_actor.shape == (2, 21) and hidden_critic.shape == critic.shape


def test_rewards_penalize_tilt_motion_and_failure_without_tracking():
    z = torch.zeros(2)
    cfg = RewardCfg()
    reward, terms = compute_rewards(cfg, z, torch.tensor([0., .2]), z, torch.zeros(2, 2), z, z,
                                    torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([False, True]))
    assert reward[0] > reward[1]
    assert terms['termination'][1] == -10
    assert not {'track', 'centre', 'progress'} & terms.keys()


def test_mass_randomization_preserves_footprint():
    dr = MovingMassRandomization()
    samples = dr.sample_many(10, 1)
    assert all(dr.terminal_mass[0] <= p.terminal_mass <= dr.terminal_mass[1] for p in samples)
    assert samples == dr.sample_many(10, 1)
    assert samples[0].with_dynamics(samples[1]).footprint == samples[0].footprint


def test_checkpoint_config_roundtrip_and_rejection(tmp_path):
    pytest.importorskip('rsl_rl')
    import yaml
    from hybrid_control.rl.policy_no_controller.config import load_env_cfg, load_configuration
    from hybrid_control.rl.policy_no_controller.env import StabilizationEnvCfg
    cfg = StabilizationEnvCfg()
    assert load_env_cfg(yaml.safe_load(yaml.safe_dump(asdict(cfg)))) == cfg
    with pytest.raises(ValueError, match='incompatible'):
        load_env_cfg({'controller': 'ik_qp'})
    with pytest.raises(ValueError, match='require'):
        load_configuration(checkpoint=str(tmp_path/'model_0.pt'))


@pytest.mark.parametrize('kwargs', [{'knot_interval': 0}, {'difficulty': 1.1}, {'max_velocity': (1, 0, 1)}])
def test_invalid_spline_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        TrajectoryCfg(**kwargs)


def test_checkpoint_load_restores_nested_settings(tmp_path):
    pytest.importorskip('rsl_rl')
    from dataclasses import replace
    import yaml
    from hybrid_control.rl.policy_no_controller.config import load_configuration
    from hybrid_control.rl.policy_no_controller.env import StabilizationEnvCfg
    cfg = replace(StabilizationEnvCfg(), physics_dt=1/100, episode_length_s=9,
                  mechanism=replace(MechanismCfg(), mast_length=.6),
                  trajectory=TrajectoryCfg(difficulty=.3), observation=ObservationCfg(com_offset_in_actor=False))
    (tmp_path/'params').mkdir()
    (tmp_path/'params'/'env.yaml').write_text(yaml.safe_dump(asdict(cfg)))
    (tmp_path/'params'/'agent.yaml').write_text(yaml.safe_dump({'num_steps_per_env': 16}))
    restored, agent = load_configuration(checkpoint=str(tmp_path/'model_0.pt'))
    assert restored == cfg and agent['num_steps_per_env'] == 16
    with pytest.raises(ValueError, match='override'):
        load_configuration(config_path='unused', checkpoint='unused')


def test_angular_limits_and_speed_floor():
    mechanism = MechanismCfg()
    assert mechanism.lower[:2] == (-math.pi/2, -math.pi/2)
    assert mechanism.upper[:2] == (math.pi/2, math.pi/2)
    s = spline(n=32)
    speeds = torch.stack([s.evaluate(t.expand(32))[1] for t in torch.linspace(0, 15, 601)])
    assert (speeds.abs().amax(0) >= torch.tensor(s.cfg.min_peak_velocity)-1e-3).all()
    # Reduced difficulty intentionally reduces the minimum peak speed too.
    easy = spline(difficulty=.2)
    v = torch.stack([easy.evaluate(t.expand(4))[1] for t in torch.linspace(0, 15, 601)])
    assert (v.abs().amax(0) >= .2*torch.tensor(easy.cfg.min_peak_velocity)-1e-3).all()


def test_planar_checkpoint_is_rejected():
    pytest.importorskip('rsl_rl')
    from hybrid_control.rl.policy_no_controller.config import load_env_cfg
    with pytest.raises(ValueError, match='older planar'):
        load_env_cfg({'environment_kind': 'moving_mass_v1'})


def test_upper_masses_randomized_and_preserved_on_dynamics_reset():
    from dataclasses import replace
    dr = MovingMassRandomization()
    samples = dr.sample_many(128, 42)
    for name in ('carriage_mass', 'gimbal_mass', 'mast_mass'):
        lo, hi = getattr(dr, name)
        values = [getattr(p, name) for p in samples]
        assert all(lo <= value <= hi for value in values)
        assert len(set(values)) == len(values)
        fixed = replace(dr, **{name: (lo, lo)})
        assert all(getattr(p, name) == lo for p in fixed.sample_many(4))
        for bounds in ((0, 1), (2, 1), (1, float('inf')), (float('nan'), 1)):
            with pytest.raises(ValueError):
                replace(dr, **{name: bounds})
        with pytest.raises(ValueError):
            replace(samples[0], **{name: 0})
    updated = samples[0].with_dynamics(samples[1])
    assert updated == replace(samples[1], footprint=samples[0].footprint)


def test_radial_v2_checkpoint_is_rejected():
    pytest.importorskip('rsl_rl')
    from hybrid_control.rl.policy_no_controller.config import load_env_cfg
    with pytest.raises(ValueError, match='incompatible'):
        load_env_cfg({'environment_kind': 'moving_mass_v2'})


def test_shared_ridgeback_defaults_and_platform_sampling():
    import random
    from dataclasses import fields
    from hybrid_control.sim_interfaces.isaac.ridgeback.params import BaseParams, DomainRandomization
    from hybrid_control.sim_interfaces.isaac.moving_mass.params import MovingMassParams, PlatformRandomization
    from hybrid_control.rl.policy_with_controller.rewards import RewardCfg as RidgebackRewardCfg

    for name in ('footprint', 'base_mass', 'base_com_z'):
        assert getattr(MovingMassParams(), name) == getattr(BaseParams(), name)
    platform, ridgeback = PlatformRandomization(), DomainRandomization()
    for f in fields(platform):
        assert getattr(platform, f.name) == getattr(ridgeback, f.name)
    for seed in range(10):
        footprint, mass, height = platform.sample(random.Random(seed))
        reference = ridgeback.sample(random.Random(seed))
        assert (footprint, mass, height) == (reference.footprint, reference.base_mass, reference.base_com_z)
    for f in fields(RewardCfg):
        assert getattr(RewardCfg(), f.name) == getattr(RidgebackRewardCfg(), f.name)



def test_terminal_mass_matches_ridgeback_payload_distribution():
    import random
    from dataclasses import replace
    from hybrid_control.sim_interfaces.isaac.ridgeback.params import BaseParams, DomainRandomization
    from hybrid_control.sim_interfaces.isaac.moving_mass.params import MovingMassParams
    assert MovingMassParams().terminal_mass == BaseParams().payload
    for bias in (1.0, 2.2, 3.0):
        ridgeback = DomainRandomization(hard_bias=bias)
        dr = MovingMassRandomization()
        dr = replace(dr, platform=replace(dr.platform, hard_bias=bias))
        assert dr.terminal_mass == ridgeback.payload
        for seed in range(20):
            actual = dr.sample(random.Random(seed))
            expected = ridgeback.sample(random.Random(seed))
            assert actual.terminal_mass == expected.payload
