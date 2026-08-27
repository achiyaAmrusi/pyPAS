import numpy as np
import pytest
from scipas.core.time_resolution import MultiGaussianRF
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import (generate_random_lt_spectrum,
                                                _convolved_decay,
                                                _time_axis_calibration)
from scipas.analysis.lifetime.fit import FitParameter
from scipas.core.lifetime import PASLifetime
from scispectrum import Spectrum


# Shared truth for the two-component spectra. Declared once so a fixture and a
# directly built spectrum cannot drift apart.
TIME_GRID = np.arange(-2, 15, 0.01)
IRF_SIGMA = np.array([0.200 / (2 * np.sqrt(2 * np.log(2)))])  # 200 ps FWHM
TRUE_MODEL = LifetimeModel("true", lifetimes=[0.2, 1.5], intensities=[0.6, 0.4])
BACKGROUND = 50.0
NUM_EVENTS = 1_000_000


def components(*values):
    """Free lifetime parameters from bare initial guesses."""
    return [FitParameter(v) for v in values]


def irf(centre=0.0):
    """Single-Gaussian instrument response of 200 ps FWHM, centred at "centre" ns."""
    return MultiGaussianRF(IRF_SIGMA, np.ones_like(IRF_SIGMA),
                           np.full_like(IRF_SIGMA, centre))


def build_spectrum(t0_true=0.0, rng=1, num_events=NUM_EVENTS):
    """Poisson two-component spectrum whose true time-zero is "t0_true".

    The offset is injected by centring the *generating* IRF at "t0_true", while
    the returned PASLifetime carries a *centred* IRF. A fit must therefore
    absorb the shift through its free t0 parameter, and it reaches it by a
    different route than the one that created it: generation moves the Gaussian
    centres, the fitter passes t0 to "convolve". Keeping the two paths distinct
    is what makes t0 recovery an independent check rather than a tautology.

    It also means the offset need not be a whole number of bins, which is the
    case that matters -- t0 lives in the resolution precisely so that sub-bin
    shifts are representable.

    Parameters
    ----------
    t0_true : float
        True time-zero in ns. At 0 the generating and fitting responses are
        identical and the spectrum is unshifted.
    rng : int or np.random.Generator
        Seed for the Poisson draw.
    num_events : int
        Expected total events before the window truncates the tail.

    Returns
    -------
    PASLifetime
        Counts per bin with "BACKGROUND" added, carrying the centred response.
    """
    generated = generate_random_lt_spectrum(TIME_GRID, TRUE_MODEL, irf(t0_true),
                                            num_events=num_events, rng=rng)
    return PASLifetime(
        lifetime=Spectrum(counts=generated.lifetime.counts + BACKGROUND,
                          axis_calib=generated.lifetime.axis_calib),
        resolution=irf(),
    )


# Shared truth for the noiseless spectra. The components sit closer together
# than TRUE_MODEL's and the response is narrower, which is what makes the linear
# solve worth testing on them.
NOISELESS_GRID = np.arange(-1.0, 10.0, 0.01)
NOISELESS_TAUS = np.array([0.2, 0.45])
NOISELESS_INTENSITIES = np.array([0.7, 0.3])
NOISELESS_TOTAL = 1_000_000
NOISELESS_BACKGROUND = 20.0


def narrow_irf(centre=0.0):
    """Single-Gaussian instrument response of 60 ps sigma, centred at "centre" ns."""
    sigma = np.array([0.06])
    return MultiGaussianRF(sigma, np.ones_like(sigma), np.full_like(sigma, centre))


def build_noiseless_spectrum(total=NOISELESS_TOTAL, background=NOISELESS_BACKGROUND):
    """Analytical two-component spectrum with a flat background, no Poisson noise.

    Lets the linear solve be checked against the intensities that built it,
    without counting statistics in the way. "total" and "background" are
    parameters because the covariance is expected to scale with them.

    Returns
    -------
    PASLifetime
        Counts per bin on "NOISELESS_GRID", carrying "narrow_irf()".
    """
    dt = NOISELESS_GRID[1] - NOISELESS_GRID[0]
    counts = total * dt * _convolved_decay(
        NOISELESS_GRID, NOISELESS_TAUS, NOISELESS_INTENSITIES, narrow_irf(), 0.0
    ) + background
    return PASLifetime(
        lifetime=Spectrum(counts=counts,
                          axis_calib=_time_axis_calibration(NOISELESS_GRID)),
        resolution=narrow_irf(),
    )


def reduced_chi_squared(pals, result):
    """Poisson reduced chi-squared of the best fit, computed by the caller.

    The fitter deliberately reports no goodness-of-fit; the tests derive it the
    way a user would, from "generate".
    """
    counts = pals.lifetime.counts
    model = result.generate(*result.opt_parameters()).lifetime.counts
    n_free = int(result.free.values.sum())
    return ((counts - model) ** 2 / np.maximum(counts, 1.0)).sum() / (len(counts) - n_free)


@pytest.fixture
def two_component_spectrum():
    """Unshifted two-component spectrum: "(pals, model, background)"."""
    return build_spectrum(), TRUE_MODEL, BACKGROUND


@pytest.fixture
def noiseless_spectrum():
    """"(pals, taus, intensities, background, total)" with no counting noise."""
    return (build_noiseless_spectrum(), NOISELESS_TAUS, NOISELESS_INTENSITIES,
            NOISELESS_BACKGROUND, NOISELESS_TOTAL)
