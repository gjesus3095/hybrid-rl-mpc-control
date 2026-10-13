import pytest

from hybrid_control.sim_interfaces.isaac.ridgeback.params import BaseParams, DomainRandomization


def test_sampling_is_seeded_and_within_ranges():
    dr = DomainRandomization()
    a, b = dr.sample_many(16, seed=3), dr.sample_many(16, seed=3)
    assert a == b
    assert len(set(p.footprint for p in a)) == 16
    for p in a:
        assert dr.length[0] <= p.footprint[0] <= dr.length[1]
        assert dr.width[0] <= p.footprint[1] <= dr.width[1]
        assert dr.base_mass[0] <= p.base_mass <= dr.base_mass[1]
        assert dr.payload[0] <= p.payload <= dr.payload[1]


def test_with_dynamics_keeps_geometry():
    old, new = BaseParams(footprint=(0.3, 0.4)), BaseParams(footprint=(0.6, 0.6), base_mass=12.0)
    merged = old.with_dynamics(new)
    assert merged.footprint == (0.3, 0.4) and merged.base_mass == 12.0


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        BaseParams(base_mass=0.0)
