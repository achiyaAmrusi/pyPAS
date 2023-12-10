"""
Module for handling spectral data.

This module defines the Spectrum class for representing and processing spectral data.
"""


import xarray as xr
import numpy as np
import pandas as pd
from uncertainties import ufloat
import lmfit
from pyspectrum import fit_functions, spectrum_tools


class Spectrum:
    """
        Represents a spectrum with methods for data manipulation and analysis.

        Attributes:
        - counts (numpy.ndarray): Array of counts.
        - channels (numpy.ndarray): Array of channels.
        - energy_calibration (numpy.poly1d): Polynomial for energy calibration.
           Methods:

    - `__init__(self, counts, channels, energy_calibration_poly=np.poly1d([1, 0]))`:
      Constructor method to initialize a Spectrum instance.

    - `xr_spectrum(self, errors=False)`:
      Returns the spectrum in xarray format. If errors is True, the xarray values will be in ufloat format.

    - `change_energy_calibration(self, energy_calibration)`:
      Change the energy calibration polynomial of the Spectrum.

    - `domain_of_peak(self, energy_in_the_peak, detector_energy_resolution=1)`:
      Find the energy domain of a peak in a spectrum.

    - `peak_gaussian_fit_parameters(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False)`:
      Fit a Gaussian function to a peak in the spectrum and return fit parameters.

    - `peak_fwhm_fit_method(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False)`:
      Calculate the Full Width at Half Maximum (FWHM) of a peak in the spectrum using peak_gaussian_fit_parameters.

    - `peak_amplitude_fit_method(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False)`:
      Calculate the amplitude of a peak in the spectrum using peak_gaussian_fit_parameters.

    - `peak_center_fit_method(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False)`:
      Calculate the center (mean) of a peak in the spectrum using peak_gaussian_fit_parameters.

    - `peak_energy_center_first_moment_method(self, energy_in_the_peak, detector_energy_resolution=1,
                                              background_subtraction=False)`:
      Calculate the center (mean) of a peak using the first moment method.

    - `counts_in_fwhm_sum_method(self, energy_in_the_peak, detector_energy_resolution=1)`:
      Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak.

    - `counts_in_fwhm_fit_method(self, energy_in_the_peak, detector_energy_resolution=1)`:
      Calculate the sum of counts within the FWHM using a fit-based method.
        """

    # Constructor method
    def __init__(self, counts, channels, energy_calibration_poly=np.poly1d([1, 0])):
        """ Constructor of Spectrum.

        Parameters:
        - counts (np.ndarray): 1D array of spectrum counts.
        - channels (np.ndarray): 1D array of spectrum channels.
        - energy_calibration_poly (np.poly1d): Calibration polynomial for energy calibration.
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
        """Return pyspectrum in xarray format
        Parameters:
        - errors (bool): If True, the xarray values will be in ufloat format, including counts error(no option to
        time normalize yet).

        Returns:
        - xr.DataArray: Xarray representation of the spectrum.
        """
        if not errors:
            spectrum = xr.DataArray(self.counts, coords={'energy': self.energy_calibration(self.channels)},
                                    dims=['energy'])
        else:
            counts_with_error = [ufloat(count, abs(count) ** 0.5) for count in self.counts]
            spectrum = xr.DataArray(counts_with_error, coords={'energy': self.energy_calibration(self.channels)},
                                    dims=['energy'])
        return spectrum

    def change_energy_calibration(self, energy_calibration):
        """change the energy calibration polynom of Spectrum
         Parameters:
         - energy_calibration (np.poly1d): The calibration function energy_calibration(channel) -> detector energy.

        Returns:
            Nothing.
                """
        if not isinstance(energy_calibration, np.poly1d):
            raise TypeError("Variable x must be of type numpy.poly1d.")
        self.energy_calibration = energy_calibration

    def domain_of_peak(self, energy_in_the_peak, detector_energy_resolution=1):
        """Find the energy domain of a peak in a spectrum.

          The function get a point on the peak, and then from the point E take a spectrum slice to the higher(lower)
          energy in the size of the detector resolution, i.e the spectrum in (E, E+resolution).
          if the slice is not relatively constant, we move to the next slice - (E+energy_bin, E+energy_bin+resolution).
          if the slice is relatively constant, or has the opposite sign from the slope of than the slice have reached
          the end of the spectrum.

         Note: It is noticeable that the higher energy side of the peak is much less noisy
            than lower energy side, so the user should expect that.
         Warning: The function keeps searching for the peak domain until either the peak ends or the spectrum ends.
         For spectra generated from a Gaussian, the function may search the entire spectrum, making the function
         even more time-intensive.

         TODO-
          there is a problem in this function which is that it checks
          if the slope is negative when it goes left, however if the
          starting poit is on the far right of the peak, it might just be
          positive and the function will stop.
          i need to be more specific an find a solution
          the solution is to define the maximum in the energy_in_the_peak close domain (up to resolution)
          and demand that in the slice that the background is only when the maximum in the background slice is
          smaller than the maximum in the energy_in_the_peak close domain (up to resolution)

         Parameters:
         - energy_in_the_peak (float): The energy around which to find the peak domain.
         - detector_energy_resolution (float, optional): The resolution of the detector. Default is 1.

         Returns:
         - tuple: A tuple containing the left and right boundaries of the identified peak domain.
         """
        spectrum = self.xr_spectrum()
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

    def peak_gaussian_fit_parameters(self, energy_in_the_peak, energy_resolution_estimation=1,
                                     background_subtraction=False):
        """Fit a Gaussian function to a peak in the spectrum and return fit parameters.

        The function fits a Gaussian function to the specified peak in the spectrum. The peak's location and
        resolution are estimated, and if background_subtraction is enabled, background subtraction is performed
        before the fitting process.

        Parameters:
        - energy_in_the_peak (float): The energy in the peak which to fit the Gaussian peak.
        - resolution_estimation (float, optional): The estimated resolution of the peak (not of the detector,
        for example in doppler broadening.) Default is 1.
        - background_subtraction (bool, optional): If True, subtract background before fitting. Default is False.

        Returns:
        - tuple: A tuple containing the fit parameters and covariance matrix of the Gaussian fit.
        The fit parameters include:
        - Amplitude: Amplitude of the Gaussian peak.
        - Center: Center (mean) of the Gaussian peak.
        - FWHM: Full Width at Half Maximum of the Gaussian peak.

        The covariance matrix provides the uncertainties in the fit parameters.
        """

        spectrum = self.xr_spectrum()
        left_energy_peak_domain, right_energy_peak_domain = self.domain_of_peak(energy_in_the_peak,
                                                                                energy_resolution_estimation)
        estimated_peak_center = spectrum_tools.peak_center_rough_estimation(spectrum, energy_in_the_peak,
                                                                            energy_resolution_estimation,
                                                                            left_energy_peak_domain,
                                                                            right_energy_peak_domain)
        if background_subtraction:
            spectrum = spectrum_tools.subtract_background_from_spectra_peak(spectrum, estimated_peak_center,
                                                                            energy_resolution_estimation,
                                                                            left_energy_peak_domain,
                                                                            right_energy_peak_domain)
        peak_spectrum = spectrum.sel(energy=slice(left_energy_peak_domain, right_energy_peak_domain))
        fit_params, cov = fit_functions.fit_gaussian(peak_spectrum, estimated_peak_center, energy_resolution_estimation)
        return fit_params, cov

    def peak_fwhm_fit_method(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False):
        """ Calculate the Full Width at Half Maximum (FWHM) of a peak in the spectrum.

       The function estimates the FWHM of the specified peak by fitting a Gaussian function to it.
       If background_subtraction is enabled, background subtraction is performed before the fitting process.

       Parameters:
       - energy_in_the_peak (float): The energy around which to calculate the FWHM.
       - resolution_estimation (float, optional): The estimated resolution of the peak. Default is 1.
       - background_subtraction (bool, optional): If True, subtract background before fitting. Default is False.

       Returns:
       - tuple: A tuple containing the FWHM and its associated uncertainty.
       The uncertainty is calculated considering the covariance matrix obtained from the Gaussian fit.
       the uncertainty given regard the covariance of the other fit parameters up to one std in them.

       Note: The function assumes the output of `peak_gaussian_fit_parameters` is used for fitting.
       """
        fit_params, cov = self.peak_gaussian_fit_parameters(energy_in_the_peak,
                                                            resolution_estimation, background_subtraction)
        fwhm = fit_params[2]
        fwhm_error = (cov[2, 2] ** 0.5 +
                      (cov[1, 1] ** 0.5 / fit_params[1]) * np.abs(cov[2, 1]) ** 0.5 +
                      (cov[0, 0] ** 0.5 / fit_params[0]) * np.abs(cov[2, 0]) ** 0.5)
        return fwhm, fwhm_error

    def peak_amplitude_fit_method(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False):
        """Calculate the amplitude of a peak in the spectrum.

        The function estimates the amplitude of the specified peak by fitting a Gaussian function to it.
        If background_subtraction is enabled, background subtraction is performed before the fitting process.

        Parameters:
        - energy_in_the_peak (float): The energy around which to calculate the peak amplitude.
        - resolution_estimation (float, optional): The estimated resolution of the peak. Default is 1.
        - background_subtraction (bool, optional): If True, subtract background before fitting. Default is False.

        Returns:
        - tuple: A tuple containing the peak amplitude and its associated uncertainty.
          The uncertainty is calculated considering the covariance matrix obtained from the Gaussian fit,
          accounting for the covariance of the other fit parameters up to one standard deviation in them.

        Note: The function assumes the output of `peak_gaussian_fit_parameters` is used for fitting.
        """
        fit_params, cov = self.peak_gaussian_fit_parameters(energy_in_the_peak,
                                                            resolution_estimation, background_subtraction)
        amplitude = fit_params[0]
        amplitude_error = (cov[0, 0] ** 0.5 +
                           (cov[1, 1] ** 0.5 / fit_params[1]) * np.abs(cov[0, 1]) ** 0.5 +
                           (cov[2, 2] ** 0.5 / fit_params[2]) * np.abs(cov[0, 2]) ** 0.5)
        return amplitude, amplitude_error

    def peak_center_fit_method(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=False):
        """Calculate the center (mean) of a peak in the spectrum.

        The function estimates the center of the specified peak by fitting a Gaussian function to it.
        If background_subtraction is enabled, background subtraction is performed before the fitting process.

        Parameters:
        - energy_in_the_peak (float): The energy around which to calculate the peak center.
        - resolution_estimation (float, optional): The estimated resolution of the peak. Default is 1.
        - background_subtraction (bool, optional): If True, subtract background before fitting. Default is False.

        Returns:
        - tuple: A tuple containing the peak center and its associated uncertainty.
          The uncertainty is calculated considering the covariance matrix obtained from the Gaussian fit,
          accounting for the covariance of the other fit parameters up to one standard deviation in them.

        Note: The function assumes the output of `peak_gaussian_fit_parameters` is used for fitting.
        """
        fit_params, cov = self.peak_gaussian_fit_parameters(energy_in_the_peak,
                                                            resolution_estimation, background_subtraction)
        amplitude = fit_params[1]
        amplitude_error = (cov[1, 1] ** 0.5 +
                           (cov[0, 0] ** 0.5 / fit_params[0]) * np.abs(cov[1, 0]) ** 0.5 +
                           (cov[2, 2] ** 0.5 / fit_params[2]) * np.abs(cov[1, 2]) ** 0.5)
        return amplitude, amplitude_error

    def peak_energy_center_first_moment_method(self, energy_in_the_peak, detector_energy_resolution=1,
                                               background_subtraction=False):
        """Calculate the center (mean) of a peak in the spectrum.

        The function estimates the center of the specified peak by finding the full width half maximum domain and
        that it use the mean on the spectrum slice to calculate the peak center .
        If background_subtraction is enabled, background subtraction is performed before the fitting process.

        Parameters:
        - energy_in_the_peak (float): The energy around which to calculate the peak center.
        - detector_energy_resolution (float, optional): The energy resolution of the peak. Default is 1.
        - background_subtraction (bool, optional): If True, subtract background before fitting. Default is False.

        Returns:
        - ufloat: A ufloat containing the peak center and its associated uncertainty.
          The uncertainty is calculated using uncertainties package

        Note: The function assumes the output of `peak_gaussian_fit_parameters` is used for fitting.
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
        fwhm_slice = (self.xr_spectrum(errors=True)).sel(energy=slice(minimal_energy, maximal_energy))
        # return the mean energy in the fwhm which is the energy center
        return (fwhm_slice * fwhm_slice.coords['energy']).sum() / fwhm_slice.sum()

    def counts_in_fwhm_sum_method(self, energy_in_the_peak, detector_energy_resolution=1):
        """Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak.

        The function operates by the following steps:
        1. Calculate the FWHM of the specified peak using the `peak_fwhm_fit_method`.
        2. Estimate the center of the peak using the `peak_energy_center_first_moment_method`.
        3. Define the energy domain within the FWHM edges.
        4. Slice the spectrum to obtain the counts within the FWHM domain.
        5. Return the sum of counts within the FWHM and its associated uncertainty.

        Parameters:
        - energy_in_the_peak (float): The energy around which to calculate the FWHM and sum counts.
        - detector_energy_resolution (float, optional): The resolution of the detector. Default is 1.

        Returns:
        - ufloat: A ufloat containing the sum of counts within the FWHM and its associated uncertainty.
          The uncertainty is calculated using uncertainties package

        Note: The function assumes background subtraction is performed during FWHM calculation.
        """
        # Calculate the peak domain and slice the peak
        fwhm, _ = self.peak_fwhm_fit_method(energy_in_the_peak, detector_energy_resolution, background_subtraction=True)
        peak_center = self.peak_energy_center_first_moment_method(energy_in_the_peak, detector_energy_resolution,
                                                                  background_subtraction=True)
        minimal_energy = peak_center - fwhm / 2
        maximal_energy = peak_center + fwhm / 2
        fwhm_slice = (self.xr_spectrum(errors=True)).sel(energy=slice(minimal_energy, maximal_energy))
        # return counts under fwhm
        return fwhm_slice.sum()

    def counts_in_fwhm_fit_method(self, energy_in_the_peak, detector_energy_resolution=1):
        """Calculate the sum of counts within the Full Width at Half Maximum (FWHM) of a peak.

        The function operates by the following steps:
        1. Estimate the amplitude of the specified peak using the `peak_amplitude_fit_method`.
        2. Estimate the FWHM of the specified peak using the `peak_fwhm_fit_method`.
        3. using the formula for the counts to return the counts number
        the formula is  0.761438079*A*np.sqrt(2 *pi)*(fwhm/(2*np.sqrt(2*np.log(2))))* 1/bin_energy_size
        where
        - 0.761438079 the area under the fwhm of a standard gaussian A*exp(-x**2/sigma)
        - A gaussian amplitude
        - np.sqrt(2 *pi)*(fwhm/(2*np.sqrt(2*np.log(2)))) amplitude correction (change of variable)
        - 1/bin_energy_size change of variable (The amplitude depends on the bin width)

        Parameters:
        - energy_in_the_peak (float): The energy around which to calculate the FWHM and sum counts.
        - detector_energy_resolution (float, optional): The resolution of the detector. Default is 1.

        Returns:
        - tuple: A tuple containing the sum of counts within the FWHM and its associated uncertainty.
          The uncertainty is calculated considering the covariance matrix obtained from the Gaussian fit,
          accounting for the covariance of the other fit parameters up to one standard deviation in them.

        Note: The function assumes background subtraction is performed during FWHM calculation.
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

    @classmethod
    def load_spectrum_file_to_spectrum_class(file_path, energy_calibration_poly=np.poly1d([1, 0])):
        """
        load spectrum from a file which has 2 columns which tab between them
        first column is the channels/energy and the second is counts
        function return Spectrum
        input :
        spectrum file - two columns with tab(\t) between them.
         first line is column names - channel, counts
         energy_calibration - numpy.poly1d([a, b])
        """
        # Load the pyspectrum file in form of DataFrame
        try:
            data = pd.read_csv(file_path, sep='\t')
        except ValueError:
            raise FileNotFoundError(f"The given data file path '{file_path}' do not exist.")
        return Spectrum(data[data.columns[1]].to_numpy(), data[data.columns[0]].to_numpy(), energy_calibration_poly)
