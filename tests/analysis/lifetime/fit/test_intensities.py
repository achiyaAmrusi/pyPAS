import numpy as np
import pytest
from scipas.analysis.lifetime.fit.intensities import (
    solve_intensities, solve_intensities_with_covariance, _components_response_basis)
from scipas.analysis.lifetime.generator import _convolved_decay, _time_axis_calibration
from scipas.core.lifetime import PASLifetime
from scipas.core.time_resolution import MultiGaussianRF
from scispectrum import Spectrum


def test_components_response_basis_columns_have_unit_integral():
    """Each column is a density, so dt * sum(column) is 1 up to window truncation."""
    time = np.arange(-1.0, 12.0, 0.01)
    dt = time[1] - time[0]
    irf = MultiGaussianRF(np.array([0.06]), np.array([1.0]), np.array([0.0]))
    taus = np.array([0.2, 0.45])
    basis = _components_response_basis(time, taus, irf, 0.0)
    assert basis.shape == (len(time), 2)
    end = time[-1] + dt
    for j, tau in enumerate(taus):
        assert dt * basis[:, j].sum() == pytest.approx(-np.expm1(-end / tau), abs=1e-4)


def test_solve_recovers_intensities_without_noise(noiseless_spectrum):
    pals, taus, intensities, bg, _ = noiseless_spectrum
    solved = solve_intensities(pals, taus, 0.0, bg)
    assert solved == pytest.approx(intensities, abs=1e-3)
    assert solved.sum() == pytest.approx(1.0, abs=2e-3)


def test_solve_is_non_negative(noiseless_spectrum):
    """NNLS floors a component the data does not support, rather than going negative."""
    pals, taus, _, bg, _ = noiseless_spectrum
    # a third component far from anything present
    solved = solve_intensities(pals, np.append(taus, 5.0), 0.0, bg)
    assert np.all(solved >= 0.0)


def test_solve_is_insensitive_to_component_order(noiseless_spectrum):
    pals, taus, _, bg, _ = noiseless_spectrum
    forward = solve_intensities(pals, taus, 0.0, bg)
    reverse = solve_intensities(pals, taus[::-1], 0.0, bg)
    assert forward == pytest.approx(reverse[::-1], abs=1e-10)


def test_covariance_shape_and_symmetry(noiseless_spectrum):
    pals, taus, _, bg, _ = noiseless_spectrum
    _, cov = solve_intensities_with_covariance(pals, taus, 0.0, bg)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-18)   # positive semi-definite


def test_covariance_scales_as_inverse_counts():
    """sigma(I) must fall as 1/sqrt(N).

    Guards the "dt * T" column scaling: the solved coefficients are intensities,
    so their errors carry the Poisson scaling of the spectrum. Building the
    system from the bare basis instead would misscale this by (dt*T)^2.

    The background scales with the counts so the spectrum shape is identical in
    every run. Held at a fixed absolute level it would dominate the tail at low
    N, where the weights stop tracking the signal and sigma falls faster than
    1/sqrt(N).
    """
    time = np.arange(-1.0, 10.0, 0.01)
    dt = time[1] - time[0]
    irf = MultiGaussianRF(np.array([0.06]), np.array([1.0]), np.array([0.0]))
    taus, ints = np.array([0.2, 0.45]), np.array([0.7, 0.3])

    sigmas = []
    for total in (1e5, 1e6, 1e7):
        bg = 20.0 * total / 1e6
        counts = total * dt * _convolved_decay(time, taus, ints, irf, 0.0) + bg
        pals = PASLifetime(Spectrum(counts=counts, axis_calib=_time_axis_calibration(time)), irf)
        _, cov = solve_intensities_with_covariance(pals, taus, 0.0, bg)
        sigmas.append(np.sqrt(np.diag(cov)))

    sigmas = np.array(sigmas)
    # a decade more counts must shrink sigma by sqrt(10)
    for step in range(2):
        assert sigmas[step] / sigmas[step + 1] == pytest.approx(np.sqrt(10.0), rel=0.05)


def test_covariance_is_smaller_than_total_spread(noiseless_spectrum):
    """cov(I|tau) omits the lifetime-induced term, so it must understate the total."""
    from scipas.analysis.lifetime.fit import LifetimeFitter, FitParameter
    pals, taus, _, bg, _ = noiseless_spectrum
    _, cond_cov = solve_intensities_with_covariance(pals, taus, 0.0, bg)
    conditional = np.sqrt(np.diag(cond_cov))

    result = LifetimeFitter().fit(
        pals,
        lifetime_components=[FitParameter(t) for t in taus],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(bg, lower=0.0),
    )
    _, drawn, _, _ = result.sample(1000, rng=0)
    assert np.all(conditional < drawn.std(axis=0, ddof=1))
