import numpy as np
import pytest
from scipas.core.time_resolution import MultiGaussianRF
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import generate_random_lt_spectrum
from scipas.analysis.lifetime.fit import FitParameter
from scipas.core.lifetime import PASLifetime
from scispectrum import Spectrum


def components(*values):
    """Free lifetime parameters from bare initial guesses."""
    return [FitParameter(v) for v in values]


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
    time = np.arange(-2, 15, 0.01)
    sigma = np.array([0.200 / (2 * np.sqrt(2 * np.log(2)))])
    irf = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("true", lifetimes=[0.2, 1.5], intensities=[0.6, 0.4])
    bg = 50.0
    r = generate_random_lt_spectrum(time, model, irf, num_events=1_000_000, rng=42)
    pals = PASLifetime(
        lifetime=Spectrum(counts=r.lifetime.counts + bg, axis_calib=r.lifetime.axis_calib),
        resolution=irf,
    )
    return pals, model, bg


@pytest.fixture
def noiseless_spectrum():
    """Analytical two-component spectrum with a flat background, no Poisson noise.

    Lets the linear solve be checked against the intensities that built it,
    without counting statistics in the way.
    """
    from scipas.analysis.lifetime.generator import _convolved_decay, _time_axis_calibration
    time = np.arange(-1.0, 10.0, 0.01)
    dt = time[1] - time[0]
    irf = MultiGaussianRF(np.array([0.06]), np.array([1.0]), np.array([0.0]))
    taus = np.array([0.2, 0.45])
    intensities = np.array([0.7, 0.3])
    total, bg = 1_000_000, 20.0
    counts = total * dt * _convolved_decay(time, taus, intensities, irf, 0.0) + bg
    pals = PASLifetime(
        lifetime=Spectrum(counts=counts, axis_calib=_time_axis_calibration(time)),
        resolution=irf,
    )
    return pals, taus, intensities, bg, total
