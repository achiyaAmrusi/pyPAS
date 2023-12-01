from uncertainties import unumpy
import numpy as np
from scipy.optimize import curve_fit


def residual_std_weight(params, data_x, data_y):
    a = params['a']
    b = params['b']
    return (a*data_x + b - unumpy.nominal_values(data_y)) / (unumpy.std_devs(data_y)+1e-5)


def gaussian(x, amplitude, mean, stddev):
    """
    Gaussian function.

    Parameters:
    - x: array-like
        Input values.
    - amplitude: float
        Amplitude of the Gaussian.
    - mean: float
        Mean (center) of the Gaussian.
    - stddev: float
        Standard deviation (width) of the Gaussian.

    Returns:
    - y: array-like
        Gaussian values.
    """
    return amplitude * np.exp(-((x-mean) / (2 * stddev))**2)


def fit_gaussian(xarray_spectrum, initial_peak_center=0, initial_std=1):
    """
    Fit a Gaussian to an xarray spectrum.

    Parameters:
    - xarray_spectrum: xarray.DataArray
        The spectrum with energy as coordinates.
    - initial_peak_center: guess for initial peak center (default is 0)
    - initial_std: guess for initial std (default is 1 as approximated to HPGe detectors)
    Returns:
    - fit_params: tuple
        The tuple containing the fit parameters (amplitude, mean, stddev).
    """
    # Extract counts and energy values from xarray
    counts = xarray_spectrum.values
    energy_values = xarray_spectrum.coords['energy'].values

    # Initial guess for fit parameters
    initial_guess = [np.max(counts), initial_peak_center, initial_std]

    # Perform the fit
    fit_params, cov = curve_fit(gaussian, energy_values, counts, p0=initial_guess)

    return fit_params, cov

