import numpy as np
from scipas.model.lifetime import LifetimeModel
from scipas.core.lifetime import TimeResolution, PASLifetime
from scipas.core.time_resolution import TIME_AXIS_NAME
from scispectrum import Spectrum
from scispectrum.calibration.axis import AxisCalibration


def _time_axis_calibration(time: np.ndarray) -> AxisCalibration:
    dt = time[1] - time[0]
    t0 = time[0]
    return AxisCalibration(lambda ch, _dt=dt, _t0=t0: ch * _dt + _t0, name=TIME_AXIS_NAME)


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

    The counts are a density: each bin holds the mean of the decay over that
    bin, so "dt * sum(counts)" is 1 up to the tail the window truncates,
    "Sum_i I_i exp(-T_end/tau_i)". Multiply by "dt * num_events" for counts
    per bin.

    Parameters
    ----------
    time : np.ndarray
        Uniformly spaced time grid in ns, strictly increasing and spanning
        time[0] < 0 < time[-1].
    model : LifetimeModel
        Lifetimes and intensities of the components. The model normalizes the
        intensities to sum 1 on construction.
    resolution : TimeResolution
        Instrument response function convolved with the decay.
    t0 : float
        Time-zero in ns, applied through the resolution.

    Returns
    -------
    PASLifetime
        Density on "time", carrying "resolution".

    Raises
    ------
    ValueError
        If "time" is not 1D, or is not strictly increasing.
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
        t0: float = 0,
        num_events: int = 1_000_000,
        rng=None,
) -> PASLifetime:
    """
    Generate a Poisson-sampled positron lifetime spectrum on a given time grid
    using a discrete exponential model convolved with the resolution function.

    Each bin is drawn from "Poisson(dt * num_events * density)", with the
    density from "generate_analytical_lt_spectrum".

    Parameters
    ----------
    time : np.ndarray
        Uniformly spaced time grid in ns, strictly increasing and spanning
        time[0] < 0 < time[-1].
    model : LifetimeModel
        Lifetimes and intensities of the components.
    resolution : TimeResolution
        Instrument response function convolved with the decay.
    t0 : float
        Time-zero in ns, applied through the resolution.
    num_events : int
        Expected total number of events before the window truncates the tail.
        Default 1_000_000.
    rng : int or np.random.Generator, optional
        Seed or generator, for reproducible draws. Default None, which draws
        from a fresh unseeded generator.

    Returns
    -------
    PASLifetime
        Counts per bin on "time", carrying "resolution".

    Raises
    ------
    ValueError
        If "time" is not 1D, or is not strictly increasing.
    """
    dt = time[1] - time[0]
    generator = np.random.default_rng(rng)
    analytical = generate_analytical_lt_spectrum(time=time,
                                                 model=model,
                                                 resolution=resolution,
                                                 t0=t0)
    measured = generator.poisson(analytical.lifetime.counts * dt * num_events).astype(float)

    spectrum = Spectrum(
        counts=measured,
        axis_calib=_time_axis_calibration(time),
    )

    return PASLifetime(lifetime=spectrum, resolution=resolution)
