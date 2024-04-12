import numpy as np
import pandas as pd
import xarray as xr
from uncertainties import ufloat
from pyspectrum import Peak, Spectrum, FindPeaks, Convolution, gaussian_2_dev

ELECTRON_REST_MASS = 511


class PASdb(Peak):
    """
    Class represents doppler broadening Spectrum and include extra properties of calculations for s and w parameters
    in the context of Positron Annihilation Spectroscopy (PAS).

    Parameters
    ----------
    peak_xarray: xr.DataArray
     The peak counts and energies in form of an xarray
    ubackground_l, ubackground_r: ufloat (default ufloat(0, 1))
     Mean counts from the left and right to the peak
     This is needed for background subtraction

    Attributes
    ----------
    peak_xarray: xr.DataArray
     The peak counts and energies in form of an xarray
    height_left, height_right: ufloat (default ufloat(0, 1))
     Mean counts from the left and right to the peak
     This is needed for background subtraction
    estimated_center, estimated_resolution: float
     the estimated mean and resolution

    Methods
    -------
    spectrum_s_calculation
      Calculates the s parameter for the 511 kev peak of Spectrum .
      Note that the error is calculated using the uncertainties module in pyspectrum values.

    spectrum_w_calculation
      Calculates the w parameter for the 511 kev peak of Spectrum .
      Note that the error is calculated using the uncertainties module in pyspectrum values.

    from_file(cls, spectrum_file_path,
                  energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None,
                  sep='\t', **kwargs)
    create PASdb from file
    """

    def __init__(self, peak_xarray: xr.DataArray, ubackground_l=ufloat(0, 1), ubackground_r=ufloat(0, 1)):
        super().__init__(peak_xarray, ubackground_l, ubackground_r)

    def s_parameter_calculation(self, energy_domain_total, energy_domain_s):
        """Calculate the S parameter for the 511 kev peak of Spectrum according to domain definitions.
        Note that the error is calculated from the use of uncertainties module in pyspectrum values.

        Parameters
        ----------
        energy_domain_total: iterable  (tuple/list)
         Tuple containing the total energy domain of interest of the defect parameter
         calculation (e.g., (E1, E2)).
        energy_domain_s: iterable  (tuple/list)
         Tuple containing the specific energy domain for S parameter calculation.

        Returns
        -------
        ufloat
         The calculated s parameter with associated uncertainty.
        """

        peak = self.subtract_background()
        energy_bin_size = (self.peak['channel'][1] - self.peak['channel'][0]).values

        e_1 = energy_domain_s[0]
        e_2 = energy_domain_s[1]

        e_1_peak = energy_domain_total[0]
        e_2_peak = energy_domain_total[1]

        s_parm = peak.sel(channel=slice(e_1, e_2)).sum().values / (
            peak.sel(channel=slice(e_1_peak, e_2_peak)).sum().values)

        # add the edges of the peaks into the s parameters
        s_parm_1_edges_frac = ((peak.sel(channel=slice(e_1, e_2))['channel'].min() - e_1) / energy_bin_size)
        s_parm_2_edges_frac = ((e_2 - peak.sel(channel=slice(e_1, e_2))['channel'].max()) / energy_bin_size)
        s_parm_edges = ((s_parm_1_edges_frac * peak.sel(channel=slice(e_1 - energy_bin_size, e_1)).sum().values +
                         s_parm_2_edges_frac * peak.sel(channel=slice(e_2, e_2 + energy_bin_size)).sum().values) /
                        peak.sel(channel=slice(e_1_peak, e_2_peak)).sum().values)
        return s_parm + s_parm_edges

    def w_parameter_calculation(self, energy_domain_total, energy_domain_w_left, energy_domain_w_right):
        """Calculate the W parameter for the 511 kev peak of Spectrum according to domain definitions.
        Note that the error is calculated from the use of uncertainties module in pyspectrum values.

        Parameters
        ----------
        energy_domain_total: iterable (tuple/list)
         Tuple containing the total energy domain of interest of the defect parameter
         calculation (e.g., (E1, E2)).
        energy_domain_w_left, energy_domain_w_right: iterable (tuple/list)
         Tuple (2 index) containing the specific energy domain
         for W parameter calculation in the right and left wing.

        Returns
        -------
        ufloat
         The calculated s parameter with associated uncertainty.
        """

        peak = self.subtract_background()
        energy_bin_size = (self.peak['channel'][1] - self.peak['channel'][0]).values

        e_1_l = energy_domain_w_left[0]
        e_2_l = energy_domain_w_left[1]

        e_1_r = energy_domain_w_right[0]
        e_2_r = energy_domain_w_right[1]

        e_1_peak = energy_domain_total[0]
        e_2_peak = energy_domain_total[1]

        w_parm = (peak.sel(channel=slice(e_1_l, e_2_l)).sum().values
                  + peak.sel(channel=slice(e_1_r, e_2_r)).sum().values) / (
                     peak.sel(channel=slice(e_1_peak, e_2_peak)).sum().values)

        w_l_1_edges_frac = ((peak.sel(channel=slice(e_1_l, e_2_l))['channel'].min() - e_1_l) / energy_bin_size)
        w_l_2_edges_frac = ((e_2_l - peak.sel(channel=slice(e_1_l, e_2_l))['channel'].max()) / energy_bin_size)

        w_l_edges = ((w_l_1_edges_frac * peak.sel(channel=slice(e_1_l - energy_bin_size, e_1_l)).sum().values +
                      w_l_2_edges_frac * peak.sel(channel=slice(e_2_l, e_2_l + energy_bin_size)).sum().values) /
                     peak.sel(channel=slice(e_1_peak, e_2_peak)).sum().values)

        w_r_1_edges_frac = ((peak.sel(channel=slice(e_1_r, e_2_r))['channel'].min() - e_1_r) / energy_bin_size)
        w_r_2_edges_frac = ((e_2_r - peak.sel(channel=slice(e_1_r, e_2_r))['channel'].max()) / energy_bin_size)

        w_r_edges = ((w_r_1_edges_frac * peak.sel(channel=slice(e_1_r - energy_bin_size, e_1_r)).sum().values +
                      w_r_2_edges_frac * peak.sel(channel=slice(e_2_r, e_2_r + energy_bin_size)).sum().values) /
                     peak.sel(channel=slice(e_1_peak, e_2_peak)).sum().values)
        return w_parm + w_l_edges + w_r_edges

    @classmethod
    def from_file(cls, spectrum_file_path,
                  energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None,
                  sep='\t', **kwargs):
        """
        load spectrum from a file which has 2 columns which tab between them
        first column is the channels/energy and the second is counts
        function return Spectrum

        Parameters
        ----------
        spectrum_file_path: str
        two columns with tab(\t) between them. first line is column names - channel, counts
        energy_calibration_poly: numpy.poly1d([a, b])
        the energy calibration of the detector
        fwhm_calibration: Callable
        a function that given energy/channel(first raw in file) returns the fwhm
        sep: str
        the separation letter
        kwargs: more parameter for pd.read_csv
        Returns
        -------
        PASdb
        db spectrum from the file in PASdb class .
        """
        # Load the pyspectrum file in form of DataFrame
        spectrum = Spectrum.from_file(file_path=spectrum_file_path,
                                      energy_calibration_poly=energy_calibration_poly,
                                      fwhm_calibration=fwhm_calibration,
                                      sep=sep, **kwargs)
        estimated_FWHM_ch = lambda ch: fwhm_calibration(energy_calibration_poly(ch)) / energy_calibration_poly[1]
        convolution = Convolution(estimated_FWHM_ch, gaussian_2_dev, 4)
        find_peaks = FindPeaks(spectrum, convolution, fitting_type='HPGe_spectroscopy')
        peak = find_peaks.to_peak(ELECTRON_REST_MASS)
        return PASdb(peak.peak, peak.height_left, peak.height_right)

    @classmethod
    def from_dataframe(cls, spectrum_data_frame: pd.DataFrame,
                       energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None):
        """
        load spectrum from a dataframe which has 2 columns
        first column is the channel and the second is counts
        function return Spectrum

        Parameters
        ----------
        spectrum_data_frame: pd.DataFrame
         spectrum in form of a dataframe such that the column are -  'channel', 'counts'
        energy_calibration_poly: numpy.poly1d([a, b])
        the energy calibration of the detector
        fwhm_calibration: Callable
        a function that given energy/channel(first raw in file) returns the fwhm

        Returns
        -------
        PASdb
        db spectrum from the file in PASdb class .
        """
        # Load the pyspectrum file in form of DataFrame
        spectrum = Spectrum.from_dataframe(spectrum_data_frame, energy_calibration_poly=energy_calibration_poly,
                                           fwhm_calibration=fwhm_calibration)
        estimated_fwhm_ch = lambda ch: fwhm_calibration(energy_calibration_poly(ch)) / energy_calibration_poly[1]
        convolution = Convolution(estimated_fwhm_ch, gaussian_2_dev, 4)
        find_peaks = FindPeaks(spectrum, convolution, fitting_type='HPGe_spectroscopy')
        peak = find_peaks.to_peak(ELECTRON_REST_MASS)
        return PASdb(peak.peak, peak.height_left, peak.height_right)

    @classmethod
    def from_spectrum(cls, spectrum: Spectrum):
        """
        load spectrum, look for the 511 peak and return it

        Parameters
        ----------
        spectrum: pd.DataFrame
         spectrum object with annhilation peak

        Returns
        -------
        PASdb
        db spectrum from the file in PASdb class .
        """
        # Load the pyspectrum file in form of DataFrame

        estimated_fwhm_ch = lambda ch: spectrum.fwhm_calibration(
            spectrum.energy_calibration(ch)) / spectrum.energy_calibration[1]
        convolution = Convolution(spectrum.fwhm_calibration, gaussian_2_dev, 5)
        find_peaks = FindPeaks(spectrum, convolution, fitting_type='HPGe_spectroscopy')
        peak = find_peaks.to_peak(ELECTRON_REST_MASS)
        return PASdb(peak.peak, peak.height_left, peak.height_right)
