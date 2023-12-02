import numpy as np
import xarray as xr
import math
from uncertainties import unumpy
import lmfit
from pyspectrum import fit_functions

ELECTRON_MASS = 511


def subtract_background_from_spectra_peak(spectrum, energy_in_the_peak, detector_energy_resolution,
                                          peak_limit_low_energy=None, peak_limit_high_energy=None):
    """ create a subtracted background pyspectrum from the xarray pyspectrum
    the method of the subtraction is as follows -
     - find the peak domain and the mean count in the edges
     - calculated erf according to the edgec
     - subtract edges from pyspectrum  """
    # calculating the peak domain
    if (peak_limit_low_energy is None) or (peak_limit_high_energy is None):
        peak_limit_low_energy, peak_limit_high_energy = domain_of_peak(spectrum,
                                                                       energy_in_the_peak, detector_energy_resolution)
    energy_center_of_the_peak = calculate_peak_center(spectrum, energy_in_the_peak, detector_energy_resolution)
    spectrum_slice_low_energy = spectrum.sel(energy=slice(peak_limit_low_energy - 5 * detector_energy_resolution,
                                                          peak_limit_low_energy))
    spectrum_slice_high_energy = spectrum.sel(energy=slice(peak_limit_high_energy,
                                                           peak_limit_high_energy + 5 * detector_energy_resolution))
    # the mean counts in the domain edges (which define the background function)
    mean_of_function_slice_low_energy = unumpy.nominal_values(spectrum_slice_low_energy.values).mean()
    mean_of_function_slice_high_energy = unumpy.nominal_values(spectrum_slice_high_energy.values).mean()
    # background subtraction
    erf_background = np.array([(math.erf(energy_value_from_peak_center) + 1) for energy_value_from_peak_center in
                               -(spectrum['energy'].values
                                 - unumpy.nominal_values(energy_center_of_the_peak).tolist())])
    # we want to subtract the background from the peak only
    compton_edge_energy = (energy_center_of_the_peak -
                           energy_center_of_the_peak * (1 / 1 + 2 * (energy_center_of_the_peak / ELECTRON_MASS)))
    theta_funtion = np.array([1 if (peak_limit_high_energy > energy > compton_edge_energy) else 0
                              for energy in spectrum['energy'].values])
    spectrum_no_bg = (spectrum - 0.5 * (
            mean_of_function_slice_low_energy - mean_of_function_slice_high_energy) * erf_background * theta_funtion)
    return spectrum_no_bg


def domain_of_peak(spectrum, energy_in_the_peak, detector_energy_resolution):
    """ define the total area of the peak
        The function takes pyspectrum slice in size of the resolution and check from which energy the counts are constant
        however because the counts are not constant,
        it checks when the counts N_sigma from the mean is larger than 1
        The auther notes that it is noticeable that the large energy side of the peak is much less noisy than lower side
        """
    fit_params = lmfit.Parameters()
    start_of_energy_slice = energy_in_the_peak
    energy_step_size = spectrum['energy'].values[1] - spectrum['energy'].values[0]

    flag = True
    while flag and start_of_energy_slice > spectrum['energy'].values[1]:
        spectrum_slice = spectrum.sel(energy=slice(start_of_energy_slice - 3 * detector_energy_resolution,
                                                   start_of_energy_slice))
        fit_params.add('a', value=1.0)
        fit_params.add('b', value=0.0)
        result = lmfit.minimize(fit_functions.residual_std_weight, fit_params,
                                args=(spectrum_slice['energy'].values, spectrum_slice.values))
        flag = not ((result.params['a'].value <= 0) or (result.params['a'].value - result.params['a'].stderr <= 0))
        start_of_energy_slice = start_of_energy_slice - energy_step_size
    left_energy_peak_domain = start_of_energy_slice

    fit_params = lmfit.Parameters()
    fit_params.add('a', value=1.0)
    fit_params.add('b', value=0.0)
    flag = True
    start_of_energy_slice = energy_in_the_peak
    while flag and start_of_energy_slice < spectrum['energy'].values[len(spectrum['energy'].values) - 2]:
        spectrum_slice = spectrum.sel(energy=slice(start_of_energy_slice,
                                                   start_of_energy_slice + 3 * detector_energy_resolution))
        fit_params.add('a', value=1.0)
        fit_params.add('b', value=0.0)
        result = lmfit.minimize(fit_functions.residual_std_weight, fit_params,
                                args=(spectrum_slice['energy'].values, spectrum_slice.values))
        flag = not ((result.params['a'].value >= 0) or (result.params['a'].value - result.params['a'].stderr >= 0))
        start_of_energy_slice = start_of_energy_slice + energy_step_size
    right_energy_peak_domain = start_of_energy_slice
    return left_energy_peak_domain, right_energy_peak_domain


def calculate_peak_center(spectrum, energy_of_the_peak, detector_energy_resolution):
    """ calculate the center of the peak
    the function operate by the following order -
     calculate the peak domain,
      find maximal value
      define domain within fwhm edges
      calculate the mean energy which is the center of the peak (like center of mass)
      """
    # Calculate the peak domain and slice the peak
    peak_limit_low_energy, peak_limit_high_energy = domain_of_peak(spectrum,
                                                                   energy_of_the_peak, detector_energy_resolution, )
    peak_slice = spectrum.sel(energy=slice(peak_limit_low_energy, peak_limit_high_energy))
    maximal_count = peak_slice.max()
    # Calculate the half-maximum count
    half_max_count = maximal_count / 2

    # Find the energy values at which the counts are closest to half-maximum on each side
    left_energy = peak_slice.where(peak_slice >= half_max_count, drop=True)['energy'].to_numpy()[0]
    right_energy = peak_slice.where(peak_slice >= half_max_count, drop=True)['energy'].to_numpy()[-1]
    # define the full width half maximum area (meaning the area which is bounded by the fwhm edges)
    fwhm_slice = spectrum.sel(energy=slice(left_energy, right_energy))
    # return the mean energy in the fwhm which is the energy center
    return (fwhm_slice * fwhm_slice.coords['energy']).sum() / fwhm_slice.sum()


def peak_center_rough_estimation(spectrum, energy_of_the_peak, detector_energy_resolution=1,
                                 peak_limit_low_energy=None, peak_limit_high_energy=None):
    """ calculate the center of the peak
    the function operate by the following order -
     calculate the peak domain,
      find maximal value
      define domain within fwhm edges
      calculate the mean energy which is the center of the peak (like center of mass)
      """
    # Calculate the peak domain and slice the peak
    if (peak_limit_low_energy is None) or (peak_limit_high_energy is None):
        peak_limit_low_energy, peak_limit_high_energy = domain_of_peak(spectrum,
                                                                       energy_of_the_peak, detector_energy_resolution)
    peak_slice = spectrum.sel(energy=slice(peak_limit_low_energy, peak_limit_high_energy))
    maximal_count = peak_slice.max()
    # Calculate the half-maximum count
    half_max_count = maximal_count / 2

    # Find the energy values at which the counts are closest to half-maximum on each side
    left_energy = peak_slice.where(peak_slice >= half_max_count, drop=True)['energy'].to_numpy()[0]
    right_energy = peak_slice.where(peak_slice >= half_max_count, drop=True)['energy'].to_numpy()[-1]
    # define the full width half maximum area (meaning the area which is bounded by the fwhm edges)
    fwhm_slice = spectrum.sel(energy=slice(left_energy, right_energy))
    # return the mean energy in the fwhm which is the energy center
    return (fwhm_slice * fwhm_slice.coords['energy']).sum() / fwhm_slice.sum()
