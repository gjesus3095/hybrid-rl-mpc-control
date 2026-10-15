"""Independent checks of the demo's analytical CoM and scripted joint probes."""
import math
import pytest

torch = pytest.importorskip('torch')
from hybrid_control.sim_interfaces.isaac.moving_mass.diagnostics import analytical_com, protocol_targets
from hybrid_control.sim_interfaces.isaac.moving_mass.params import MechanismCfg, MovingMassParams


def test_com_upright_roll_pitch_and_rotated_base():
    cfg = MechanismCfg()
    params = [MovingMassParams(base_mass=30, terminal_mass=3)] * 4
    q = torch.tensor([[0, 0, .2], [0, math.pi/2, .2], [math.pi/2, 0, .2], [0, math.pi/2, .2]])
    root = torch.tensor([[0., 0, .01], [0, 0, .01], [0, 0, .01], [1, 2, .01]])
    quat = torch.tensor([[0., 0, 0, 1]]*3 + [[0, 0, math.sqrt(.5), math.sqrt(.5)]])
    actual = analytical_com(q, root, quat, params, cfg)
    neutral_z = (30*.1 + 4*.56)/34 + .01
    excursion = (.25*.25 + 3*.54)/34
    expected = torch.tensor([[0, 0, neutral_z+excursion], [excursion, 0, neutral_z],
                             [0, -excursion, neutral_z], [1, 2+excursion, neutral_z]])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_probe_endpoints_and_analytic_velocity():
    initial = torch.tensor([[0., 0., .2]])
    target = torch.tensor([[0., math.pi/2, .4]])
    for time, expected in [(0, initial), (1, initial), (5, target), (8, target), (12, initial), (15, initial)]:
        q, v = protocol_targets(time, initial, target, 1, 4, 3, True)
        assert torch.allclose(q, expected, atol=1e-6)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)
    for time in (2., 4., 9., 11.):
        eps = .001
        left, _ = protocol_targets(time-eps, initial, target, 1, 4, 3, True)
        right, _ = protocol_targets(time+eps, initial, target, 1, 4, 3, True)
        q, v = protocol_targets(time, initial, target, 1, 4, 3, True)
        assert torch.allclose((right-left)/(2*eps), v, atol=2e-4)
        assert torch.equal(q[:, 0], initial[:, 0])


def test_com_uses_each_episodes_upper_masses():
    cfg = MechanismCfg()
    params = [MovingMassParams(carriage_mass=1, gimbal_mass=.5, mast_mass=.5),
              MovingMassParams(carriage_mass=.25, gimbal_mass=.125, mast_mass=.125)]
    q = torch.tensor([[0, math.pi/2, .2]] * 2)
    actual = analytical_com(q, torch.zeros(2, 3), torch.tensor([[0., 0, 0, 1]] * 2), params, cfg)
    for i, p in enumerate(params):
        upper = p.carriage_mass + p.gimbal_mass + p.mast_mass + p.terminal_mass
        total = p.base_mass + upper
        expected = torch.tensor([(p.mast_mass*.25+p.terminal_mass*.54)/total,
                                 0, (p.base_mass*.1+upper*.56)/total])
        assert torch.allclose(actual[i], expected, atol=1e-6)
