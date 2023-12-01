import xarray as xr
import numpy as np
import lmfit
import s_auxiliary_function
import s_tools


class Spectrum:

    # Constructor method
    def __init__(self, counts, channels, energy_calibration_poly):
        # Instance variables
        self.counts = counts
        self.channels = channels
        self.energy_calibration = energy_calibration_poly

    # Instance method
    def xr_spectrum(self):
        """return spectrum in xarray"""
        return xr.DataArray(self.counts, coords={'energy': self.energy_calibration(self.channels)}, dims=['energy'])

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
            The auther notes that it is noticeable that the large energy side of the peak is much less noisy than lower side
            """
        spectrum = self.xr_spectrum()
        fit_params = lmfit.Parameters()
        start_of_energy_slice = energy_in_the_peak
        energy_step_size = spectrum['energy'].values[1] - spectrum['energy'].values[0]

        flag = True
        while flag:
            spectrum_slice = spectrum.sel(energy=slice(start_of_energy_slice - 3 * detector_energy_resolution,
                                                       start_of_energy_slice))
            fit_params.add('a', value=1.0)
            fit_params.add('b', value=0.0)
            result = lmfit.minimize(s_auxiliary_function.residual_std_weight, fit_params,
                                    args=(spectrum_slice['energy'].values, spectrum_slice.values))
            flag = not ((result.params['a'].value <= 0) or (result.params['a'].value - result.params['a'].stderr <= 0))
            start_of_energy_slice = start_of_energy_slice - energy_step_size
        left_energy_peak_domain = start_of_energy_slice

        fit_params = lmfit.Parameters()
        fit_params.add('a', value=1.0)
        fit_params.add('b', value=0.0)
        flag = True
        start_of_energy_slice = energy_in_the_peak
        while flag:
            spectrum_slice = spectrum.sel(energy=slice(start_of_energy_slice,
                                                       start_of_energy_slice + 3 * detector_energy_resolution))
            fit_params.add('a', value=1.0)
            fit_params.add('b', value=0.0)
            result = lmfit.minimize(s_auxiliary_function.residual_std_weight, fit_params,
                                    args=(spectrum_slice['energy'].values, spectrum_slice.values))
            flag = not ((result.params['a'].value >= 0) or (result.params['a'].value - result.params['a'].stderr >= 0))
            start_of_energy_slice = start_of_energy_slice + energy_step_size
        right_energy_peak_domain = start_of_energy_slice
        return left_energy_peak_domain, right_energy_peak_domain

    def peak_fwhm(self, energy_in_the_peak, resolution_estimation=1, background_subtraction=True):
        """ return the full width half maximum of a peak """
        left_energy_peak_domain, right_energy_peak_domain = self.domain_of_peak(energy_in_the_peak,
                                                                                resolution_estimation)
        spectrum = self.xr_spectrum()
        if background_subtraction:
            spectrum = s_tools.subtract_background_from_spectra_peak(spectrum, energy_in_the_peak,
                                                                     resolution_estimation)
        peak_spectrum = spectrum.sel(energy=slice(left_energy_peak_domain, right_energy_peak_domain))
        fit_params, cov = s_auxiliary_function.fit_gaussian(peak_spectrum, energy_in_the_peak, resolution_estimation)
        fwhm = fit_params[2]
        fwhm_error = (cov[2, 2] ** 0.5 +
                      (cov[1, 1] ** 0.5 / fit_params[1]) * np.abs(cov[2, 1]) ** 0.5 +
                      (cov[0, 0] ** 0.5 / fit_params[0]) * np.abs(cov[2, 0]) ** 0.5)/fit_params[2]
        return fwhm, fwhm_error
