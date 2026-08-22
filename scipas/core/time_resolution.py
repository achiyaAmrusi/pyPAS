from scispectrum import Spectrum
from abc import ABC, abstractmethod
import numpy as np

class TimeResolution(ABC):
    """
    Abstract base class for time-domain resolution (instrument response) modeling.

    Conceptual Meaning
    ------------------
    TimeResolution represents the detector or measurement system response function
    that distorts the ideal physical time spectrum before observation.

    In positron fit spectroscopy, the measured spectrum is typically modeled as:

        Measured_signal(t) = Convolution[ True_physics_signal(t), resolution(t') ]

    where resolution(t') is the instrument response function.

    This class defines the interface for evaluating the resolution function
    and performing numerical convolution with a physical signal.
    Based on this interface, various resolution types can be defined.
    SciPAS currently provides two types of resolution classes:
    - MeasuredRF, measured based resolution function
    - MultiGaussianRF, gausian sum fit based resolution

    Methods
    -------
    - evaluate(time: np.ndarray) -> np.ndarray
    - convolve(signal: np.ndarray, time: np.ndarray, t0: float = 0) -> np.ndarray

    Notes
    -----
    - The convolution is performed in the time domain.
    - The implementation assumes constant grid spacing.
    """
    @abstractmethod
    def evaluate(self, time: np.ndarray) -> np.ndarray:
        """
        Returns the instrument response function evaluated on the time axis.

        Parameters
        ----------
        time : np.ndarray
            Time grid on which the resolution function is evaluated.

        Returns
        -------
        np.ndarray
            Resolution function values on the given grid.
        """
        pass

    def convolve(self, signal: np.ndarray, time: np.ndarray, t0=0) -> np.ndarray:
        """
        Numerically convolve physical signal with instrument response function.

        Computes, on the grid "time",

            out[m] = Sum_j signal[j] * IRF(time[m] - time[j] - t0) * dt

        The IRF enters as a function of the time difference dt*(m - j),
        It is evaluated on that grid of differences.

            (arange(len(time)) - zero_point_index) * dt - t0

        zero_point_index = round(-time[0] / dt) locates t = 0 on the grid. It
        serves twice, and must be the same index in both: it fixes the zero of
        the IRF grid, and it is the offset at which the np.convolve output is
        sliced back onto "time",

            [zero_point_index : len(time) + zero_point_index]


        Parameters
        ----------
        signal : np.ndarray
            Ideal physical signal before detector response distortion.
        time : np.ndarray
            Uniformly spaced time grid corresponding to signal discretization.
        t0 : float
            Time-zero shift of the resolution; the IRF is displaced by t0.
            Default 0.

        Returns
        -------
        np.ndarray
            Convolved signal on the same time grid ``time``, scaled by dt.

        Notes
        -----
        - Assumes uniform time spacing dt = time[1] - time[0].
        - time must span zero: time[0] <= 0 <= time[-1].
        - The IRF must be resolved by the grid and decay inside it.
        """
        dt = time[1] - time[0]
        zero_point_index = int(round(-time[0] / dt))

        irf = self.evaluate((np.arange(len(time)) - zero_point_index) * dt - t0)

        return np.convolve(signal, irf, mode="full")[zero_point_index:len(time)+zero_point_index] * dt


class MeasuredRF(TimeResolution):
    """
    Resolution function constructed directly from measured spectrum data.
    This class wraps experimentally measured detector response spectra.

    Parameters
    ----------
    spectrum : Spectrum
        Spectroscopy spectrum container holding detector response counts.

    Notes
    -----
    - The spectrum counts are normalized to sum 1 on the measured grid.
    """
    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum
        self.spectrum.counts = self.spectrum.counts/self.spectrum.counts.sum()

    def evaluate(self, time: np.ndarray) -> np.ndarray:
        """
        Interpolate the measured resolution function onto the given time grid.
        Values outside the measured support are 0.

        Parameters
        ----------
        time : np.ndarray
            Time axis to evaluate on.

        Returns
        -------
        np.ndarray
            Resolution function values at ``time``.
        """
        return np.asarray(np.interp(time,
                                    self.spectrum.energy.values,
                                    self.spectrum.counts,
                                    left=0.0,
                                    right=0.0))


class MultiGaussianRF(TimeResolution):
    """
    Multi-component Gaussian Instrument Response Function.

    Mathematical Model
    ------------------
    The resolution function is modeled as a weighted mixture of Gaussian kernels:

        IRF(t) = Σ_i w_i exp(-(t - t0_i)^2 / (2 σ_i^2))

    Parameters
    ----------
    sigmas : np.ndarray
        Standard deviations of Gaussian components.
    weights : np.ndarray
        Mixing weights of Gaussian components.
        The weights are normalized in the initialization
    t0 : np.ndarray
        Center offsets of Gaussian components.

    """

    def __init__(self, sigmas: np.ndarray, weights: np.ndarray, t0: np.ndarray):
        self.sigmas = sigmas
        self.weights = weights
        self.t0 = t0
        if (self.sigmas.ndim != 1 or len(self.sigmas) == 0) or (self.weights.ndim != 1 or len(self.weights) == 0)  or (self.t0.ndim != 1 or len(self.t0) == 0):
            raise ValueError("sigmas and weights must be nonempty 1D")

        if not (len(self.sigmas) == len(self.weights) == len(self.t0)):
            raise ValueError("sigmas, t0 and weights must have same length")

        if np.any(self.sigmas <= 0) or np.any(self.weights < 0):
            raise ValueError("All sigmas must be positive")

        self.weights = self.weights / self.weights.sum()  # normalize


    def evaluate(self, time: np.ndarray) -> np.ndarray:
        """
        Evaluate multi-Gaussian resolution function.

        Parameters
        ----------
        time : np.ndarray
         Time axis.

        Returns
        -------
        np.ndarray
            Normalized resolution function.
        """
        sigma = np.vstack(self.sigmas)
        weight = np.vstack(self.weights)
        t_center = np.vstack(self.t0)
        components = weight * np.sqrt(1/(2*np.pi*sigma**2)) * np.exp(-(time - t_center)**2 / (2 * sigma**2))
        # Normalize numerically (important for discrete grid)
        return components.sum(axis=0)
