"""Monte Carlo check that the reported covariance has the right scale.
draw independent spectra from one truth, fit each, and compare the spread
of the fits against the "pcov" the fits themselves report.
Both checks center on the **ensemble mean**, not on the truth. The fitted
background carries a known bias of about one count from the Neyman weighting
(see the weighting decision in CLAUDE.md), which is a property of the estimator
rather than of its covariance. Centring on the mean removes it, so what is
under test here is the covariance alone.
"""
import numpy as np
import pytest

from scispectrum import Spectrum
from scipas.core.lifetime import PASLifetime
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import generate_random_lt_spectrum
from scipas.analysis.lifetime.fit import LifetimeFitter, FitParameter

from conftest import narrow_irf, NOISELESS_GRID

N_FITS = 100
TRUE_TAU = 0.25
TRUE_BACKGROUND = 100.0
NUM_EVENTS = 1_000_000


def _fit_one(seed):
    """Fit one Poisson realization of the single-component truth.

    The background is *drawn*, not added as a constant. Adding it flat leaves
    the tail bins with no scatter at all, and the background is determined
    almost entirely by the tail - its fitted spread then collapses while pcov
    keeps predicting what Poisson bins would give, and the comparison fails
    against correct code.
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


def test_predicted_sigma_matches_the_scatter(ensemble):
    """Reported sigma must match the spread of the fits, parameter by parameter.

    Asserted on the ratio, so the bound is symmetric. "approx(predicted,
    rel=...)" is not: an inflated prediction widens its own tolerance, and a
    pcov twice too large slipped through that way while the Mahalanobis check
    below caught it.

    Tolerance from the Monte Carlo error on a standard deviation,
    1/sqrt(2(n-1)) = 7.1% at n=100, so 0.20 is a little under three of those.
    Asserted on the ratio, so the bound is symmetric. "approx(predicted,
    rel=...)" is not: an inflated prediction widens its own tolerance, and a
    pcov twice too large slips through that way.

    0.15 is about two Monte Carlo errors (7.1% at n=100) and twice the observed
    deviation - the ratios are 0.977 and 0.933.

    Sensitivity is asymmetric, because those ratios sit just below 1: an
    overstated sigma pushes them further down and is caught from about 1.1x,
    while an understated one pushes them back toward 1 and is masked until
    roughly 1.25x. The Mahalanobis check below is the guard on that side.
    """
    popt, pcov = ensemble
    observed = popt.std(axis=0, ddof=1)
    predicted = np.sqrt(np.diagonal(pcov, axis1=1, axis2=2)).mean(axis=0)

    assert observed / predicted == pytest.approx(1.0, abs=0.15)


def test_mahalanobis_matches_the_free_parameter_count(ensemble):
    """The whole matrix at once, correlations included.

    "d^2 = (theta - mean)^T pcov^-1 (theta - mean)" is chi-squared distributed
    with "n_free" degrees of freedom, so its mean over the ensemble must be
    "n_free" - here 2, for tau and background. A per-parameter comparison would
    miss a covariance with the right diagonal and wrong off-diagonal; this does
    not.

    The mean of n draws of chi2_k has standard error sqrt(2k/n) = 0.20 at k=2,
    n=100, and the bound is three of those. The observed mean is 1.79. Kept at
    three rather than two because chi-squared is skewed and this statistic is
    the backstop, not the sensitive one.
    """
    popt, pcov = ensemble
    n_free = popt.shape[1]
    deviation = popt - popt.mean(axis=0)

    d2 = np.array([d @ np.linalg.inv(c) @ d for d, c in zip(deviation, pcov)])

    assert d2.mean() == pytest.approx(n_free, abs=3 * np.sqrt(2 * n_free / N_FITS))
