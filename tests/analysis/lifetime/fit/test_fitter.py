import numpy as np
import pytest
from scipas.core.time_resolution import MultiGaussianRF
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import generate_random_lt_spectrum
from scipas.analysis.lifetime.fit import LifetimeFitter, FitParameter
from scipas.core.lifetime import PASLifetime
from scispectrum import Spectrum

from conftest import components, reduced_chi_squared


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
    pals, _, bg = two_component_spectrum
    fixed_tau = 0.2
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=[FitParameter(fixed_tau, fixed=True),
                             FitParameter(1.2)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(bg, lower=0.0),
    )
    taus, _, t0, _ = result.opt_parameters()
    assert taus[0] == pytest.approx(fixed_tau, abs=1e-10)
    assert t0 == pytest.approx(0.0, abs=1e-10)


def test_single_component_fit():
    time = np.arange(-2, 10, 0.01)
    sigma = np.array([0.15 / (2 * np.sqrt(2 * np.log(2)))])
    irf = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("single", lifetimes=[0.5], intensities=[1.0])
    r = generate_random_lt_spectrum(time, model, irf, num_events=500_000, rng=99)
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


def _shifted_spectrum(t0_true, seed=7, num_events=1_000_000):
    """Two-component spectrum with a genuine time-zero offset.

    The offset is injected by centering the generating IRF at ``t0_true``;
    the fit is then given a *centered* IRF so the free t0 parameter must
    absorb the shift. This is an independent check of t0 recovery (the
    generator never uses the fitter's forward model).
    """
    time = np.arange(-2, 15, 0.01)
    sigma = np.array([0.200 / (2 * np.sqrt(2 * np.log(2)))])
    irf_shifted = MultiGaussianRF(sigma, np.ones_like(sigma), np.array([t0_true]))
    irf_centered = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("true", lifetimes=[0.2, 1.5], intensities=[0.6, 0.4])
    bg = 50.0
    r = generate_random_lt_spectrum(time, model, irf_shifted, num_events=num_events, rng=seed)
    pals = PASLifetime(
        lifetime=Spectrum(counts=r.lifetime.counts + bg, axis_calib=r.lifetime.axis_calib),
        resolution=irf_centered,
    )
    return pals, bg


@pytest.mark.parametrize("t0_true", [-0.05, 0.08, 0.15])
def test_fit_recovers_t0(t0_true):
    pals, bg = _shifted_spectrum(t0_true)
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=components(0.15, 1.2),
        t0=FitParameter(0.0, lower=-0.5, upper=0.5),
        background=FitParameter(bg, lower=0.0),
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
    pals, bg = _shifted_spectrum(0.08)
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=components(0.15, 1.2),
        t0=FitParameter(0.0, lower=-0.5, upper=0.5),
        background=FitParameter(bg, lower=0.0),
    )
    t0_error = np.sqrt(float(result.pcov.sel(parameter="t0", parameter0="t0")))
    assert np.isfinite(t0_error)
    assert t0_error < 0.05


def test_fixed_nonzero_t0_shifts_model(two_component_spectrum):
    """A fixed non-zero t0 must move the model peak by that amount."""
    pals, _, bg = two_component_spectrum
    fitter = LifetimeFitter()
    args = dict(
        lifetime_components=[FitParameter(0.2, fixed=True),
                             FitParameter(1.5, fixed=True)],
        background=FitParameter(bg, lower=0.0),
    )
    base = fitter.fit(pals, t0=FitParameter(0.0, fixed=True), **args)
    shifted = fitter.fit(pals, t0=FitParameter(0.10, fixed=True), **args)

    time = pals.lifetime.axis
    peak_base = time[np.argmax(base.generate(*base.opt_parameters()).lifetime.counts)]
    peak_shifted = time[np.argmax(shifted.generate(*shifted.opt_parameters()).lifetime.counts)]
    assert peak_shifted - peak_base == pytest.approx(0.10, abs=0.02)


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
