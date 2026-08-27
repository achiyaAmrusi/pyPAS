import numpy as np
import pytest
from scipas.analysis.lifetime.fit.intensities import (
    solve_intensities, solve_intensities_with_covariance, _weighted_system)
from scipas.analysis.lifetime.generator import _convolved_decay


def test_solve_recovers_intensities_without_noise(noiseless_spectrum):
    pals, taus, intensities, bg, _ = noiseless_spectrum
    solved = solve_intensities(pals, taus, 0.0, bg)
    assert solved == pytest.approx(intensities, abs=1e-3)
    assert solved.sum() == pytest.approx(1.0, abs=2e-3)


def test_covariance_shape_and_symmetry(noiseless_spectrum):
    """cov(I|tau) is n x n over the components, not over the bins, and symmetric."""
    pals, taus, _, bg, _ = noiseless_spectrum
    _, cov = solve_intensities_with_covariance(pals, taus, 0.0, bg)
    assert cov.shape == (len(taus), len(taus))
    assert np.allclose(cov, cov.T)


def test_design(noiseless_spectrum):
    """Column j is the unit-intensity response for tau_j, scaled by "dt * T"
    and weighted by 1/sqrt(counts); the right-hand side is the net counts under
    the same weights."""
    pals, taus, _, bg, _ = noiseless_spectrum
    time = pals.lifetime.axis
    counts = pals.lifetime.counts
    dt = time[1] - time[0]
    total = counts.sum() - bg * len(counts)
    weight = 1.0 / np.sqrt(np.maximum(counts, 1.0))

    design, rhs = _weighted_system(pals, taus, 0.0, bg)

    assert design.shape == (len(time), len(taus))
    for j, tau in enumerate(taus):
        unit_response = _convolved_decay(time, np.array([tau]), np.array([1.0]),
                                         pals.resolution, 0.0)
        assert design[:, j] == pytest.approx(dt * total * unit_response * weight,
                                             rel=1e-12)
    assert rhs == pytest.approx((counts - bg) * weight, rel=1e-12)
