import numpy as np
from scipas.model.lifetime import LifetimeModel
from scipas.core.lifetime import TimeResolution, PASLifetime
from scispectrum import Spectrum
from scispectrum.calibration.axis import AxisCalibration


def _time_axis_calibration(time: np.ndarray) -> AxisCalibration:
    dt = time[1] - time[0]
    t0 = time[0]
    return AxisCalibration(lambda ch, _dt=dt, _t0=t0: ch * _dt + _t0, name="energy")


def _convolved_decay(
        time: np.ndarray,
        lifetimes: np.ndarray,
        intensities: np.ndarray,
        resolution: TimeResolution,
        t0: float = 0 ) -> np.ndarray:
    """
    Discrete multi-exponential decay "Sum_i (I_i/tau_i) exp(-t/tau_i)" for
    t > 0, convolved with the resolution function.
    The value of the distribution in the bin is taken as the analytical mean of the distribution.

    Parameters
    ----------
    time : np.ndarray
        Uniformly spaced time grid in ns, spanning time[0] < 0 < time[-1].
    lifetimes : np.ndarray
        Characteristic lifetime tau_i of each component, in ns.
    intensities : np.ndarray
        Relative intensity I_i of each component, in the order of "lifetimes".
    resolution : TimeResolution
        Instrument response function convolved with the decay.
    t0 : float
        Time-zero in ns; the resolution is evaluated on "time - t0".

    Returns
    -------
    np.ndarray
        the distribution density on "time". Inputs are not validated.
    """
    decay = np.zeros_like(time, dtype=float)
    lifetime = np.vstack(lifetimes)
    intensity = np.vstack(intensities)
    dt = time[1] - time[0]

    # bin k spans [time[k], time[k] + dt) and holds the mean of the decay over it
    index_time_0 = np.searchsorted(time, 0.0, side="right") - 1
    onset_edge = time[index_time_0] + dt
    # onset bin overlap t=0, so it is integrated from 0: Sum_i I_i (1 - exp(-edge/tau_i))
    decay[index_time_0] = (intensity * (-np.expm1(-onset_edge / lifetime))).sum() / dt
    # full bins: Sum_i I_i exp(-t/tau_i) (1 - exp(-dt/tau_i))
    decay[index_time_0 + 1:] = (intensity * (-np.expm1(-dt / lifetime)) * np.exp(-time[index_time_0 + 1:] / lifetime)).sum(axis=0) / dt
    decay = resolution.convolve(signal=decay, time=time, t0=t0)

    return decay

def generate_analytical_lt_spectrum(
    time: np.ndarray,
    model: LifetimeModel,
    resolution: TimeResolution,
    t0: float = 0) -> PASLifetime:
    """
    Generate a normalized positron lifetime spectrum on a given time grid
    using a discrete exponential model convolved with the resolution function.
    """

    if time.ndim != 1:
        raise ValueError("Time must be 1D array")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time axis must be strictly increasing")

    decay = _convolved_decay(time,
                             model.lifetimes,
                             model.intensities,
                             resolution,
                             t0)

    spectrum = Spectrum(
        counts=decay,
        axis_calib=_time_axis_calibration(time),
    )
    return PASLifetime(lifetime=spectrum, resolution=resolution)


def generate_random_lt_spectrum(
        time: np.ndarray,
        model: LifetimeModel,
        resolution: TimeResolution,
        t0 = 0,
        num_events: int = 1_000_000,
) -> PASLifetime:
    """
    Generate a Poisson-sampled positron lifetime spectrum on a given time grid
    using a discrete exponential model convolved with the resolution function.
    """
    dt = time[1] - time[0]
    analytical = generate_analytical_lt_spectrum(time=time,
                                                 model=model,
                                                 resolution=resolution,
                                                 t0=t0)
    measured = np.random.poisson(analytical.lifetime.counts * dt * num_events).astype(float)

    spectrum = Spectrum(
        counts=measured,
        axis_calib=_time_axis_calibration(time),
    )

    return PASLifetime(lifetime=spectrum, resolution=resolution)
