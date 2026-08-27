"""Tests for the time-resolution layer every fit routes through.

"convolve" is checked against a direct transcription of its own definition,

    out[m] = Sum_j signal[j] * IRF(time[m] - time[j] - t0) * dt

which shares only "evaluate" with the implementation. It therefore validates
the convolution bookkeeping - the zero-point index and the lag axis, where two
separate bugs have been fixed - rather than "evaluate" itself, which is covered
on its own below.
"""
import numpy as np
import pytest

from scispectrum import Spectrum, AxisCalibration
from scipas.core.time_resolution import (MultiGaussianRF, MeasuredRF,
                                         TIME_AXIS_NAME)
from scipas.core.lifetime import PASLifetime


def brute_convolve(resolution, signal, time, t0=0.0):
    """The definition, evaluated one output sample at a time."""
    dt = time[1] - time[0]
    return np.array([np.sum(signal * resolution.evaluate(t - time - t0)) * dt
                     for t in time])


def gaussian_irf(fwhm=0.200, centre=0.0):
    sigma = np.array([fwhm / (2 * np.sqrt(2 * np.log(2)))])
    return MultiGaussianRF(sigma, np.ones_like(sigma), np.full_like(sigma, centre))


def bump(time, centre=1.0, width=0.3):
    """Smooth signal supported well inside the window, so the fixed lag axis
    of "convolve" and the sliding one of "brute_convolve" cannot disagree."""
    return np.exp(-0.5 * ((time - centre) / width) ** 2)


# -----------------------------------------------------------------------------
# convolve
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("t0", [0.0, 0.037, -0.05, 0.15])
@pytest.mark.parametrize("start,stop,dt", [(-2.0, 10.0, 0.01),
                                           (-1.0, 8.0, 0.05),
                                           (-3.5, 6.0, 0.02)])
def test_convolve_matches_the_definition(t0, start, stop, dt):
    """Agreement is to machine precision, not to a tolerance: the two routes
    compute the same sum in a different order."""
    time = np.arange(start, stop, dt)
    irf = gaussian_irf()
    signal = bump(time)

    got = irf.convolve(signal=signal, time=time, t0=t0)
    want = brute_convolve(irf, signal, time, t0)

    assert np.abs(got - want).max() / np.abs(want).max() < 1e-12


def test_convolve_delta_reproduces_the_irf():
    """A unit impulse at t=0 must return the IRF itself.

    This is what pins "zero_point_index": the impulse sits at t=0, so the
    output has to be the IRF centred at t=0 too. An off-by-one in that index
    displaces the result by exactly one bin and nothing else would notice.
    """
    time = np.arange(-2.0, 10.0, 0.01)
    dt = time[1] - time[0]
    irf = gaussian_irf()

    signal = np.zeros_like(time)
    signal[np.argmin(np.abs(time))] = 1.0 / dt      # unit area impulse at t=0

    out = irf.convolve(signal=signal, time=time)

    assert np.abs(out - irf.evaluate(time)).max() < 1e-12
    assert time[np.argmax(out)] == pytest.approx(0.0, abs=dt)


def test_convolve_displaces_the_output_by_t0():
    """t0 shifts the result by exactly t0, with its shape untouched.

    Chosen so t0 is a whole number of bins, which lets the two curves be
    compared sample by sample with no interpolation. The comparison covers the
    whole curve rather than a peak position, so a shift of the wrong shape or
    magnitude fails as well as one of the wrong sign.
    """
    time = np.arange(-2.0, 10.0, 0.01)
    dt = time[1] - time[0]
    shift_bins = 10
    t0 = shift_bins * dt
    irf = gaussian_irf()
    signal = bump(time)

    base = irf.convolve(signal=signal, time=time, t0=0.0)
    shifted = irf.convolve(signal=signal, time=time, t0=t0)

    assert np.abs(shifted[shift_bins:] - base[:-shift_bins]).max() < 1e-12
    # and the sign is right: a positive t0 moves the curve later
    assert time[np.argmax(shifted)] > time[np.argmax(base)]


def test_convolve_preserves_area():
    """The IRF has unit integral, so convolving cannot change the signal's."""
    time = np.arange(-2.0, 10.0, 0.01)
    dt = time[1] - time[0]
    irf = gaussian_irf()
    signal = bump(time)

    out = irf.convolve(signal=signal, time=time)
    assert out.sum() * dt == pytest.approx(signal.sum() * dt, rel=1e-6)


# -----------------------------------------------------------------------------
# MultiGaussianRF
# -----------------------------------------------------------------------------

def test_multigaussian_has_unit_integral():
    time = np.arange(-3.0, 3.0, 0.001)
    dt = time[1] - time[0]
    assert gaussian_irf().evaluate(time).sum() * dt == pytest.approx(1.0, abs=1e-9)


def test_multigaussian_is_centred_on_t0():
    time = np.arange(-3.0, 3.0, 0.001)
    for centre in (-0.4, 0.0, 0.25):
        peak = time[np.argmax(gaussian_irf(centre=centre).evaluate(time))]
        assert peak == pytest.approx(centre, abs=1e-3)


def test_multigaussian_weights_are_normalized():
    irf = MultiGaussianRF(np.array([0.1, 0.3]), np.array([3.0, 1.0]),
                          np.array([0.0, 0.0]))
    assert irf.weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("sigmas,weights,t0", [
    (np.array([0.1]), np.array([1.0, 1.0]), np.array([0.0])),      # lengths differ
    (np.array([-0.1]), np.array([1.0]), np.array([0.0])),          # sigma <= 0
    (np.array([]), np.array([]), np.array([])),                    # empty
])
def test_multigaussian_rejects_bad_parameters(sigmas, weights, t0):
    with pytest.raises(ValueError):
        MultiGaussianRF(sigmas, weights, t0)


# -----------------------------------------------------------------------------
# MeasuredRF
# -----------------------------------------------------------------------------

def measured_irf(name=TIME_AXIS_NAME):
    time = np.arange(-1.0, 1.0, 0.01)
    counts = np.exp(-0.5 * (time / 0.08) ** 2)
    calib = AxisCalibration(lambda ch, _t=time: _t[0] + ch * 0.01, name=name)
    return MeasuredRF(Spectrum(counts=counts, axis_calib=calib))


def test_measured_rf_normalizes_counts():
    assert measured_irf().spectrum.counts.sum() == pytest.approx(1.0)


def test_measured_rf_is_zero_outside_its_support():
    irf = measured_irf()
    assert irf.evaluate(np.array([-50.0, 50.0])) == pytest.approx([0.0, 0.0])


def test_measured_rf_rejects_a_non_time_axis():
    with pytest.raises(ValueError, match="expected 'time'"):
        measured_irf(name="energy")


# -----------------------------------------------------------------------------
# PASLifetime
# -----------------------------------------------------------------------------

def time_spectrum(name=TIME_AXIS_NAME):
    calib = AxisCalibration(lambda ch: -2.0 + ch * 0.01, name=name)
    return Spectrum(counts=np.ones(100), axis_calib=calib)


def test_pas_lifetime_accepts_a_time_axis():
    pals = PASLifetime(time_spectrum(), gaussian_irf())
    assert pals.lifetime.axis_name == TIME_AXIS_NAME


@pytest.mark.parametrize("name", ["energy", "channel", "axis"])
def test_pas_lifetime_rejects_a_non_time_axis(name):
    with pytest.raises(ValueError, match="expected 'time'"):
        PASLifetime(time_spectrum(name=name), gaussian_irf())
