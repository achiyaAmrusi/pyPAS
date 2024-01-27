import numpy as np


def gaussian(x, amplitude, mean, fwhm):
    """
    Gaussian function.

    Parameters:
    - x: array-like
        Input values.
    - amplitude: float
        Amplitude of the Gaussian.
    - mean: float
        Mean (center) of the Gaussian.
    - fwhm: float
        Standard deviation/2.35482 (width) of the Gaussian
        this is half of the distance for which the Gaussian gives half of the maximum value.

    Returns:
    - y: array-like
        Gaussian values.
    """
    return amplitude * np.exp(-(1/2)*((x-mean) / (fwhm/2.35482))**2)


def gaussian_1_dev(x, amplitude, mean, fwhm):
    """
    First derivative of a Gaussian.

    Parameters:
    - x: array-like
        Input values.
    - amplitude: float
        Amplitude of the Gaussian.
    - mean: float
        Mean (center) of the Gaussian.
    - fwhm: float
        Standard deviation/2.35482 (width) of the Gaussian
        this is half of the distance for which the Gaussian gives half of the maximum value.
    Returns:
    numpy array
        first derivaive of a Gaussian.

    """
    return amplitude * (-(x-mean) / (fwhm/2.35482)**2) * gaussian(x, amplitude, mean, fwhm)


def gaussian_2_dev(x, amplitude, mean, fwhm):
    """
    First derivative of a Gaussian.

    Parameters:
    - x: array-like
        Input values.
    - amplitude: float
        Amplitude of the Gaussian.
    - mean: float
        Mean (center) of the Gaussian.
    - fwhm: float
        Standard deviation/2.35482 (width) of the Gaussian
        this is half of the distance for which the Gaussian gives half of the maximum value.
    Returns:
    numpy array
        first derivaive of a Gaussian.

    """
    return amplitude * (((fwhm/2.35482)**2-(x-mean)**2) / (fwhm/2.35482)**4) * gaussian(x, amplitude, mean, fwhm)

