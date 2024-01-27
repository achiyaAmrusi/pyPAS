"""
Module for handling spectral data.

This module defines the Spectrum class for representing and processing spectral data.
"""


import xarray as xr
import numpy as np
import pandas as pd
from uncertainties import ufloat
import lmfit
from .fit_functions import residual_std_weight


class Spectrum:
    """
        Represents a spectrum with methods for data manipulation and analysis.

        Attributes:
        - counts (numpy.ndarray): Array of counts.
        - channels (numpy.ndarray): Array of channels.
        - energy_calibration (numpy.poly1d): Polynomial for energy calibration.
        - fwhm_calibration (function): function for fwhm calibration channel->fwhm.

           Methods:

    - `__init__(self, counts, channels, energy_calibration_poly=np.poly1d([1, 0]))`:
      Constructor method to initialize a Spectrum instance.

    - `xr_spectrum(self, errors=False)`:
      Returns the spectrum in xarray format. If errors is True, the xarray values will be in ufloat format.

    - `change_energy_calibration(self, energy_calibration)`:
      Change the energy calibration polynomial of the Spectrum.

    - `change_fwhm_calibration(self, fwhm_calibration)`:
      Change the fwhm calibration function of the Spectrum.

# this function must be in gamma_find_peak
 #   - `domain_of_peak(self, energy_in_the_peak, detector_energy_resolution=1)`:
 #     Find the energy domain of a peak in a spectrum.
        """

    # Constructor method
    def __init__(self, counts, channels, energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None):
        """ Constructor of Spectrum.

        Parameters:
        - counts (np.ndarray): 1D array of spectrum counts.
        - channels (np.ndarray): 1D array of spectrum channels.
        - energy_calibration_poly (np.poly1d): Calibration polynomial for energy calibration.
        - fwhm_calibration (function): a given method
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
        self.fwhm_calibration = fwhm_calibration

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

    def calibrate_energy(self, energy_calibration):
        """change the energy calibration polynom of Spectrum
         Parameters:
         - energy_calibration (np.poly1d): The calibration function energy_calibration(channel) -> detector energy.

        Returns:
            Nothing.
                """
        if not isinstance(energy_calibration, np.poly1d):
            raise TypeError("Variable x must be of type numpy.poly1d.")
        self.energy_calibration = energy_calibration

    def calibrate_fwhm(self, fwhm_calibration):
        """change the energy calibration polynom of Spectrum
         Parameters:
         - fwhm_calibration (function): The calibration function fwhm_calibration(channel) -> fwhm in the channel.

        Returns:
            Nothing.
                """
        if not callable(fwhm_calibration):
            raise TypeError("fwhm_calibration needs to be callable.")
        self.fwhm_calibration = fwhm_calibration

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
            result = lmfit.minimize(residual_std_weight, fit_params,
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
            result = lmfit.minimize(residual_std_weight, fit_params,
                                    args=(spectrum_slice['energy'].values, spectrum_slice.values))
            flag = not ((result.params['a'].value >= 0) or (result.params['a'].value - result.params['a'].stderr >= 0))
            start_of_energy_slice = start_of_energy_slice + energy_step_size
        right_energy_peak_domain = start_of_energy_slice
        return left_energy_peak_domain, right_energy_peak_domain


    @classmethod
    def load_spectrum_file_to_spectrum_class(cls, file_path,
                                             energy_calibration_poly=np.poly1d([1, 0]),
                                             fwhm_calibration=None, sep='\t'):

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
            data = pd.read_csv(file_path, sep=sep)
        except ValueError:
            raise FileNotFoundError(f"The given data file path '{file_path}' do not exist.")
        return Spectrum(data[data.columns[1]].to_numpy(), data[data.columns[0]].to_numpy(),
                        energy_calibration_poly,
                        fwhm_calibration)
