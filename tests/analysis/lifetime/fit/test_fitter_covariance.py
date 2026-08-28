"""Monte Carlo check that the reported covariance has the right scale.
Draw independent spectra from one truth, fit each, and compare the spread
of the fits against the "pcov" the fits themselves report.
Both checks center on the **ensemble mean**, not on the truth, because the
fitted background carries a known bias of about one count from the Neyman
weighting (see the weighting decision in CLAUDE.md).
The background is large - 100 counts/bin - for a separate reason: the same
weighting makes the reported sigma(bg) about 13% optimistic at ~20 counts/bin,
an O(1/counts) effect that is gone by 60.
"""
import numpy as np
import pytest

from scispectrum import Spectrum
from scipas.core.lifetime import PASLifetime
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import generate_random_lt_spectrum
from scipas.analysis.lifetime.fit import LifetimeFitter, FitParameter

from conftest import narrow_irf, NOISELESS_GRID

N_FITS = 400
TRUE_TAU = 0.25
TRUE_BACKGROUND = 100.0
NUM_EVENTS = 1_000_000


def _fit_one(seed):
    """Fit one Poisson realization of the single-component truth.
    The background is *drawn*, not added as a constant.
    """
    rng = np.random.default_rng(seed)
    model = LifetimeModel("one", lifetimes=[TRUE_TAU], intensities=[1.0])
    generated = generate_random_lt_spectrum(NOISELESS_GRID, model, narrow_irf(),
                                            num_events=NUM_EVENTS, rng=rng)
    counts = generated.lifetime.counts + rng.poisson(TRUE_BACKGROUND,
                                                     size=len(NOISELESS_GRID))
    pals = PASLifetime(
        lifetime=Spectrum(counts=counts.astype(float),
                          axis_calib=generated.lifetime.axis_calib),
        resolution=narrow_irf(),
    )
    return LifetimeFitter().fit(
        pals,
        lifetime_components=[FitParameter(TRUE_TAU * 1.1)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(TRUE_BACKGROUND * 0.8, lower=0.0),
    )


@pytest.fixture(scope="module")
def ensemble():
    """"(popt, pcov)" over the free slots for N_FITS independent realizations.

    Module scoped: the fits are the whole cost of this file.
    """
    results = [_fit_one(seed) for seed in range(N_FITS)]
    assert all(r.optimizer.success for r in results)

    free = results[0].free.values
    popt = np.array([r.popt.values[free] for r in results])
    pcov = np.array([r.pcov.values[np.ix_(free, free)] for r in results])
    return popt, pcov


def test_pcov_matches_the_ensemble_covariance(ensemble):
    """The reported pcov must match the covariance of the fits themselves.

    Two assertions rather than one comparison of the matrices, because the
    elements are known to very different relative precision: the diagonal to a
    few percent, the off-diagonal to about 50%, since the correlation is small
    and its sampling error is large beside it.

    The diagonal bound is the Monte Carlo error on a *standard deviation*,
    1/sqrt(2(n-1)): Var(s^2) = 2 sigma^4/(n-1), and the square root halves that
    relative error. The off-diagonal bound is the error on a correlation,
    (1 - rho^2)/sqrt(n), which for a small rho is 1/sqrt(n). Both are doubled
    and both are computed from N_FITS, so they follow if it changes.
    """
    popt, pcov = ensemble
    predicted = pcov.mean(axis=0)
    predicted_sigma = np.sqrt(np.diag(predicted))

    # the diagonal, as a ratio: abs rather than relative so the bound does not
    # scale with the prediction, which would let an inflated pcov widen it
    assert popt.std(axis=0, ddof=1) / predicted_sigma == pytest.approx(
        1.0, abs=2 / np.sqrt(2 * (N_FITS - 1)))

    # the off-diagonal, which the ratio above cannot see
    assert np.corrcoef(popt.T) == pytest.approx(
        predicted / np.outer(predicted_sigma, predicted_sigma),
        abs=2 / np.sqrt(N_FITS))
