import numpy as np
import pytest
from scipas.core.time_resolution import MultiGaussianRF
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import generate_random_lt_spectrum
from scipas.analysis.lifetime.fit import LifetimeFitter, FitParameter
from scipas.core.lifetime import PASLifetime
from scispectrum import Spectrum

from conftest import components, reduced_chi_squared, build_spectrum, BACKGROUND


def test_fit_recovers_lifetimes(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=components(0.15, 1.2),
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(bg, lower=0.0),
    )
    assert result.optimizer.success
    taus = sorted(result.opt_parameters()[0])
    assert abs(taus[0] - 0.2) < 0.03
    assert abs(taus[1] - 1.5) < 0.15


def test_fit_recovers_background(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=components(0.15, 1.2),
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(40.0, lower=0.0),
    )
    assert abs(result.opt_parameters()[3] - bg) < 10


def test_fixed_params_stay_fixed(two_component_spectrum):
    """A fixed parameter is returned as the value it was fixed to.

    t0 is pinned away from zero on purpose. Fixing it at 0.0 would pass even if
    the fitter dropped the value entirely and reported its default, so the test
    could not tell "held fixed" from "silently zeroed". The spectrum's true t0
    is 0, which makes 0.07 a deliberately wrong value: the point is that the
    fitter honours it, not that it fits well.
    """
    pals, _, bg = two_component_spectrum
    fixed_tau, fixed_t0 = 0.2, 0.07
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=[FitParameter(fixed_tau, fixed=True),
                             FitParameter(1.2)],
        t0=FitParameter(fixed_t0, fixed=True),
        background=FitParameter(bg, lower=0.0),
    )
    taus, _, t0, _ = result.opt_parameters()
    assert taus[0] == pytest.approx(fixed_tau, abs=1e-10)
    assert t0 == pytest.approx(fixed_t0, abs=1e-10)


def test_single_component_fit():
    time = np.arange(-2, 10, 0.01)
    sigma = np.array([0.15 / (2 * np.sqrt(2 * np.log(2)))])
    irf = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("single", lifetimes=[0.5], intensities=[1.0])
    r = generate_random_lt_spectrum(time, model, irf, num_events=500_000, rng=1)
    bg = 20.0
    pals = PASLifetime(
        lifetime=Spectrum(counts=r.lifetime.counts + bg, axis_calib=r.lifetime.axis_calib),
        resolution=irf,
    )
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=components(0.4),
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(15.0, lower=0.0),
    )
    assert result.optimizer.success
    taus, intensities, _, _ = result.opt_parameters()
    assert abs(taus[0] - 0.5) < 0.05
    assert intensities.shape == (1,)


def test_no_components_raises(two_component_spectrum):
    pals, _, _ = two_component_spectrum
    with pytest.raises(ValueError, match="At least one component"):
        LifetimeFitter().fit(pals, lifetime_components=[])


def test_no_free_params_raises(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    with pytest.raises(ValueError, match="No free"):
        LifetimeFitter().fit(
            pals,
            lifetime_components=[FitParameter(0.2, fixed=True)],
            t0=FitParameter(0.0, fixed=True),
            background=FitParameter(bg, fixed=True),
        )


@pytest.mark.parametrize("t0_true", [-0.05, 0.08, 0.15])
def test_fit_recovers_t0(t0_true):
    pals = build_spectrum(t0_true=t0_true, rng=1)
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=components(0.15, 1.2),
        t0=FitParameter(0.0, lower=-0.5, upper=0.5),
        background=FitParameter(BACKGROUND, lower=0.0),
    )
    assert result.optimizer.success
    taus, _, t0, _ = result.opt_parameters()
    # t0 recovered to well under the bin spacing (0.01 ns)
    assert abs(t0 - t0_true) < 0.01
    # lifetimes must not be corrupted by the shift
    taus = sorted(taus)
    assert abs(taus[0] - 0.2) < 0.03
    assert abs(taus[1] - 1.5) < 0.15
    assert reduced_chi_squared(pals, result) < 2.0


def test_free_t0_has_finite_error():
    """Regression: the t0 Jacobian column must be non-degenerate.

    With the old support-truncation forward model the t0 finite-difference
    column was ~0, giving an absurd (1e5) error and a stalled fit.
    """
    pals = build_spectrum(t0_true=0.08, rng=1)
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=components(0.15, 1.2),
        t0=FitParameter(0.0, lower=-0.5, upper=0.5),
        background=FitParameter(BACKGROUND, lower=0.0),
    )
    t0_error = np.sqrt(float(result.pcov.sel(parameter="t0", parameter0="t0")))
    assert np.isfinite(t0_error)
    assert t0_error < 0.05


def test_covariance_is_slot_sized_and_zero_on_fixed(two_component_spectrum):
    """estimate_cov returns (n_free, n_free); the result must expose (n+2, n+2)."""
    pals, _, bg = two_component_spectrum
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=[FitParameter(0.2, fixed=True),
                             FitParameter(1.2)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(bg, lower=0.0),
    )
    assert result.pcov.shape == (4, 4)
    assert list(result.free.values) == [False, True, False, True]
    for slot in (0, 2):
        assert np.all(result.pcov.values[slot] == 0.0)
        assert np.all(result.pcov.values[:, slot] == 0.0)
    assert np.all(np.diag(result.pcov.values)[[1, 3]] > 0.0)
