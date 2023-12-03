import xarray as xr
import numpy as np
from uncertainties import ufloat
import lmfit
from pyspectrum import fit_functions, spectrum_tools


class Spectrum:

    # Constructor method
    def __init__(self, counts, channels, energy_calibration_poly):
        """ Constructor of Spectrum
        input:
         counts - 1d np.ndarray of spectrum counts
         channels - 1d np.ndarray of spectrum channels
         energy_calibration_poly - np.poly1d  which is the calibration
         """
        # Instance variables
        if not (isinstance(counts, np.ndarray) and counts.ndim == 1):
            raise TypeError("Variable counts must be of type 1 dimension np.array.")
        self.counts = counts
        if not (isinstance(channels, np.ndarray) and channels.ndim == 1):
            raise TypeError("Variable channels must be of type 1 dimension np.array.")
        self.channels = channels
        if not isinstance(energy_calibration_poly, np.poly1d):
            raise TypeError("Variable energy_calibration_poly must be of type numpy.poly1d.")
        self.energy_calibration = energy_calibration_poly

    # Instance method
    def xr_spectrum(self, errors=False):
        """return pyspectrum in xarray
        if errors is True the spectrum will be in uarray format which include the counts error"""
        if not errors:
            spectrum = xr.DataArray(self.counts, coords={'energy': self.energy_calibration(self.channels)},
                                    dims=['energy'])
        else:
            counts_with_error = [ufloat(count, count ** 0.5) for count in self.counts]
            spectrum = xr.DataArray(counts_with_error, coords={'energy': self.energy_calibration(self.channels)},
                                    dims=['energy'])
        return spectrum

    def update_energy_calibration(self, calibration_poly):
        """update the energy calibration polynom of Spectrum """
        if not isinstance(calibration_poly, np.poly1d):
            raise TypeError("Variable x must be of type numpy.poly1d.")
        self.energy_calibration = calibration_poly

    def domain_of_peak(self, energy_in_the_peak, detector_energy_resolution=1):
        """ define the total area of the peak
            The function takes spectrum slice in size of the resolution and check from which energy the counts are constant
            however because the counts are not constant,
            it checks when the counts N_sigma from the mean is larger than 1
            The auther notes that it is noticeable that the large energy side of the peak is much less noisy
            than lower energy side
            warning : for generated data from gaussian the domain is all the space and thus the function keep
             searching the peak domain to the end of the spectrum! this is time intensive
            """
        spectrum = self.xr_spectrum(True)
        fit_params = lmfit.Parameters()
        start_of_energy_slice = energy_in_the_peak
        energy_step_size = spectrum['energy'].values[1] - spectrum['energy'].values[0]

        flag = True
        # the while keeps on until the peak is over or the spectrum is over
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

    def peak_gaussian_fit_parameters(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False):
        """ return the full width half maximum of a peak
           if background_subtraction=True subtract background
            """
        spectrum = self.xr_spectrum()
        left_energy_peak_domain, right_energy_peak_domain = self.domain_of_peak(energy_in_the_peak,
                                                                                resolution_estimation)
        estimated_peak_center = spectrum_tools.peak_center_rough_estimation(spectrum, energy_in_the_peak,
                                                                            resolution_estimation,
                                                                            left_energy_peak_domain,
                                                                            right_energy_peak_domain)
        if background_subtraction:
            spectrum = spectrum_tools.subtract_background_from_spectra_peak(spectrum, estimated_peak_center,
                                                                            resolution_estimation,
                                                                            left_energy_peak_domain,
                                                                            right_energy_peak_domain)
        peak_spectrum = spectrum.sel(energy=slice(left_energy_peak_domain, right_energy_peak_domain))
        fit_params, cov = fit_functions.fit_gaussian(peak_spectrum, estimated_peak_center, resolution_estimation)
        return fit_params, cov

    def peak_fwhm_fit_method(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False):
        """ return the full width half maximum of a peak
        if background_subtraction=True subtract background"""
        fit_params, cov = self.peak_gaussian_fit_parameters(energy_in_the_peak,
                                                            resolution_estimation, background_subtraction)
        fwhm = fit_params[2]
        fwhm_error = (cov[2, 2] ** 0.5 +
                      (cov[1, 1] ** 0.5 / fit_params[1]) * np.abs(cov[2, 1]) ** 0.5 +
                      (cov[0, 0] ** 0.5 / fit_params[0]) * np.abs(cov[2, 0]) ** 0.5)
        return fwhm, fwhm_error

    def peak_amplitude_fit_method(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False):
        """ return the full width half maximum of a peak
        if background_subtraction=True subtract background"""
        fit_params, cov = self.peak_gaussian_fit_parameters(energy_in_the_peak,
                                                            resolution_estimation, background_subtraction)
        amplitude = fit_params[0]
        amplitude_error = (cov[0, 0] ** 0.5 +
                           (cov[1, 1] ** 0.5 / fit_params[1]) * np.abs(cov[0, 1]) ** 0.5 +
                           (cov[2, 2] ** 0.5 / fit_params[2]) * np.abs(cov[0, 2]) ** 0.5)
        return amplitude, amplitude_error

    def peak_center_fit_method(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False):
        """ return the full width half maximum of a peak
        if background_subtraction=True subtract background"""
        fit_params, cov = self.peak_gaussian_fit_parameters(energy_in_the_peak,
                                                            resolution_estimation, background_subtraction)
        amplitude = fit_params[1]
        amplitude_error = (cov[1, 1] ** 0.5 +
                           (cov[0, 0] ** 0.5 / fit_params[0]) * np.abs(cov[1, 0]) ** 0.5 +
                           (cov[2, 2] ** 0.5 / fit_params[2]) * np.abs(cov[1, 2]) ** 0.5)
        return amplitude, amplitude_error

    def peak_energy_center_first_moment_method(self, energy_in_the_peak, detector_energy_resolution=1,
                                               background_subtraction=False):
        """ calculate the center of the peak
        the function operate by the following order -
         calculate the peak domain,
          find maximal value
          define domain within fwhm edges
          calculate the mean energy which is the center of the peak (like center of mass)
          """
        # Calculate the peak domain and slice the peak
        fit_params, cov = self.peak_gaussian_fit_parameters(energy_in_the_peak,
                                                            detector_energy_resolution, background_subtraction)
        fwhm = fit_params[2]
        fwhm_error = (cov[2, 2] ** 0.5 +
                      (cov[1, 1] ** 0.5 / fit_params[1]) * np.abs(cov[2, 1]) ** 0.5 +
                      (cov[0, 0] ** 0.5 / fit_params[0]) * np.abs(cov[2, 0]) ** 0.5)
        gaussian_center = fit_params[1]
        gaussian_center_error = (cov[1, 1] ** 0.5 +
                                 (cov[2, 2] ** 0.5 / fit_params[1]) * np.abs(cov[1, 2]) ** 0.5 +
                                 (cov[0, 0] ** 0.5 / fit_params[0]) * np.abs(cov[1, 0]) ** 0.5)
        minimal_energy = gaussian_center - gaussian_center_error - fwhm - fwhm_error
        maximal_energy = gaussian_center + gaussian_center_error + fwhm + fwhm_error
        fwhm_slice = (self.xr_spectrum()).sel(energy=slice(minimal_energy, maximal_energy))
        # return the mean energy in the fwhm which is the energy center
        return (fwhm_slice * fwhm_slice.coords['energy']).sum() / fwhm_slice.sum()

    def counts_in_fwhm_sum_method(self, energy_in_the_peak, detector_energy_resolution=1):
        """ calculate the center of the peak
        the function operate by the following order -
         calculate the peak domain,
          find maximal value
          define domain within fwhm edges
          calculate the mean energy which is the center of the peak (like center of mass)
          """
        # Calculate the peak domain and slice the peak
        fwhm, _ = self.peak_fwhm_fit_method(energy_in_the_peak, detector_energy_resolution, background_subtraction=True)
        peak_center = self.peak_energy_center_first_moment_method(energy_in_the_peak, detector_energy_resolution,
                                                                  background_subtraction=True)
        minimal_energy = peak_center - fwhm / 2
        maximal_energy = peak_center + fwhm / 2
        fwhm_slice = (self.xr_spectrum()).sel(energy=slice(minimal_energy, maximal_energy))
        # return counts under fwhm
        return float(fwhm_slice.sum()), float(np.abs(fwhm_slice.sum()) ** 0.5)

    def counts_in_fwhm_fit_method(self, energy_in_the_peak, detector_energy_resolution=1):
        """ calculate the counts in under a peak without background using gaussian fit to the peak
        given the amplitude A of a gaussian A*exp(-x**2/sigma), the area under the fwhm is approximately-
        0.761438079*A*np.sqrt(2 *pi)*(fwhm/(2*np.sqrt(2*np.log(2))))
        however, this has units of counts*Energy because the spectrum is a histogram.
        To compensate, another factor is needed which is 1/bin_energy_size to get counts

          """
        # Calculate the peak amplitude and fwhm
        amplitude, amplitude_error = self.peak_amplitude_fit_method(energy_in_the_peak,
                                                                    detector_energy_resolution,
                                                                    background_subtraction=True)
        fwhm, _ = self.peak_fwhm_fit_method(energy_in_the_peak, detector_energy_resolution,
                                            background_subtraction=True)
        # energy size of each bin
        bin_energy_size = self.energy_calibration(1) - self.energy_calibration(0)
        # factor to get area under the fwhm
        factor_of_area = 0.761438079 * np.sqrt(2 * np.pi) * (fwhm / (2 * np.sqrt(2 * np.log(2))))
        return factor_of_area * amplitude * (1 / bin_energy_size), factor_of_area * amplitude_error * (
                    1 / bin_energy_size)
