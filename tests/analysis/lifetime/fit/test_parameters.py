import warnings

import numpy as np
import pytest
from scipas.analysis.lifetime.fit.parameters import (
    FitParameter, ParameterMap, TAU_LOWER_BOUND)


def _map(lifetimes, t0=None, background=None):
    return ParameterMap(lifetimes,
                        t0 if t0 is not None else FitParameter(0.0),
                        background if background is not None else FitParameter(5.0))


def test_slot_layout_and_names():
    """Components are identified by index; the labels are positional."""
    pmap = _map([FitParameter(0.2), FitParameter(1.5)])
    assert pmap.parameter_names == ["tau_0", "tau_1", "t0", "background"]
    assert pmap.n_lifetime == 2
    assert pmap.n_free == 4
    assert list(pmap.free_mask) == [True] * 4


def test_free_mask_follows_fixed_flags():
    pmap = _map([FitParameter(0.2, fixed=True), FitParameter(1.5)],
                t0=FitParameter(0.0, fixed=True))
    assert list(pmap.free_mask) == [False, True, False, True]
    assert pmap.n_free == 2
    assert pmap.initial_vector() == pytest.approx([1.5, 5.0])


def test_pack_unpack_round_trip():
    pmap = _map([FitParameter(0.2), FitParameter(1.5, fixed=True)])
    taus, t0, background = pmap.unpack(pmap.pack([0.3, 1.5], 0.02, 7.0))
    assert taus == pytest.approx([0.3, 1.5])
    assert t0 == pytest.approx(0.02)
    assert background == pytest.approx(7.0)


def test_full_vector_restores_fixed_slots():
    pmap = _map([FitParameter(0.2, fixed=True), FitParameter(1.5)],
                t0=FitParameter(0.05, fixed=True))
    assert pmap.full_vector(np.array([1.7, 6.0])) == pytest.approx([0.2, 1.7, 0.05, 6.0])


def test_embed_covariance_places_free_block_and_zeros_fixed():
    pmap = _map([FitParameter(0.2, fixed=True), FitParameter(1.5)],
                t0=FitParameter(0.0, fixed=True))
    cov_free = np.array([[4.0, 1.0], [1.0, 9.0]])       # over [tau_1, background]
    full = pmap.embed_covariance(cov_free)

    assert full.shape == (4, 4)
    assert np.allclose(full, full.T)
    for slot in (0, 2):                                  # fixed tau_0 and t0
        assert np.all(full[slot] == 0.0)
        assert np.all(full[:, slot] == 0.0)
    assert full[1, 1] == 4.0
    assert full[3, 3] == 9.0
    assert full[1, 3] == 1.0 and full[3, 1] == 1.0


def test_embed_covariance_is_identity_when_all_free():
    pmap = _map([FitParameter(0.2)])
    cov_free = np.arange(9.0).reshape(3, 3)
    assert np.array_equal(pmap.embed_covariance(cov_free), cov_free)


# --- the tau floor, now applied by ParameterMap ------------------------------

def test_tau_floor_is_applied_to_every_lifetime():
    pmap = _map([FitParameter(0.2), FitParameter(1.5)])
    assert pmap.bounds_lower[0] == TAU_LOWER_BOUND
    assert pmap.bounds_lower[1] == TAU_LOWER_BOUND


def test_tau_floor_is_clamped_silently_by_default():
    """The default -inf is always below the floor; clamping it must not warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pmap = _map([FitParameter(0.2)])
    assert pmap.bounds_lower[0] == TAU_LOWER_BOUND
    assert not [w for w in caught if "physical" in str(w.message)]


def test_finite_lower_bound_below_floor_warns_and_is_clamped():
    with pytest.warns(UserWarning, match=r"tau_0.*physical range"):
        pmap = _map([FitParameter(0.2, lower=1e-9)])
    assert pmap.bounds_lower[0] == TAU_LOWER_BOUND


def test_warning_identifies_the_offending_component_by_index():
    with pytest.warns(UserWarning, match=r"tau_1"):
        _map([FitParameter(0.2), FitParameter(1.5, lower=1e-9)])


def test_lifetime_above_floor_keeps_its_own_bound():
    pmap = _map([FitParameter(0.2, lower=0.1, upper=0.5)])
    assert pmap.bounds_lower[0] == pytest.approx(0.1)
    assert pmap.bounds_upper[0] == pytest.approx(0.5)


def test_clamping_does_not_mutate_the_caller():
    original = FitParameter(0.2)
    _map([original])
    assert original.lower == -np.inf


def test_lifetime_below_the_floor_raises():
    """Clamping runs before the feasibility check, so a sub-picosecond tau fails."""
    with pytest.raises(ValueError, match="outside its"):
        _map([FitParameter(1e-6)])


# --- feasibility -------------------------------------------------------------

def test_initial_value_outside_bounds_raises():
    with pytest.raises(ValueError, match="outside its"):
        _map([FitParameter(5.0, lower=0.1, upper=1.0)])


def test_background_initial_value_outside_bounds_raises():
    with pytest.raises(ValueError, match="outside its"):
        _map([FitParameter(0.2)], background=FitParameter(-1.0, lower=0.0))


def test_bounds_are_reported_for_free_slots_only():
    pmap = _map([FitParameter(0.2, fixed=True),
                 FitParameter(1.5, lower=1.0, upper=2.0)],
                t0=FitParameter(0.0, lower=-0.5, upper=0.5),
                background=FitParameter(5.0, fixed=True))
    assert pmap.bounds_lower == pytest.approx([1.0, -0.5])
    assert pmap.bounds_upper == pytest.approx([2.0, 0.5])
