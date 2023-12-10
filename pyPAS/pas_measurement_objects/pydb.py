import pandas as pd
import numpy as np
from pyspectrum.spectrum import Spectrum
from pyspectrum import spectrum_tools
from uncertainties import unumpy
ELECTRON_REST_MASS = 511


class PASdb(Spectrum):
    """
    Class represents doppler broadening Spectrum and include extra properties of calculations for s and w parameters
    in the context of Positron Annihilation Spectroscopy (PAS).

    Attributes:
    - Inherits attributes from the Spectrum class.

    Methods:
    - spectrum_s_calculation
      Calculates the s parameter for the 511 kev peak of Spectrum .
      Note that the error is calculated using the uncertainties module in pyspectrum values.

    - spectrum_w_calculation
      Calculates the w parameter for the 511 kev peak of Spectrum .
      Note that the error is calculated using the uncertainties module in pyspectrum values.
    """

    def s_parameter_calculation(self, peak_energy_resolution, energy_domain_total, energy_domain_s,
                                background_subtraction=True):
        """Calculate the S parameter for the 511 kev peak of Spectrum according to domain definitions.
        Note that the error is calculated from the use of uncertainties module in pyspectrum values.

        Parameters:
        - peak_energy_resolution (float): The energy resolution of the detector in the 511 kev energy peak.
        - energy_domain_total (tuple/list): Tuple containing the total energy domain of interest of the defect parameter
         calculation (e.g., (E1, E2)).
        - energy_domain_s (tuple/list): Tuple containing the specific energy domain for S parameter calculation.
        - background_subtraction (bool, optional): If True, subtract background before calculation. Default is True.

        Returns:
        - ufloat: The calculated s parameter with associated uncertainty.
        """

        peak_energy_center = self.peak_energy_center_first_moment_method(ELECTRON_REST_MASS, peak_energy_resolution,
                                                                         background_subtraction)

        if background_subtraction:
            spectrum = spectrum_tools.subtract_background_from_spectra_peak(self.xr_spectrum(errors=True),
                                                                            unumpy.nominal_values(peak_energy_center),
                                                                            peak_energy_resolution)
        else:
            spectrum = self.xr_spectrum(errors=True)

        energy_bin_size = (spectrum['energy'][1] - spectrum['energy'][0]).values

        e_1 = energy_domain_s[0]
        e_2 = energy_domain_s[1]

        e_1_peak = energy_domain_total[0]
        e_2_peak = energy_domain_total[1]

        s_parm = spectrum.sel(energy=slice(e_1, e_2)).sum().values / (
            spectrum.sel(energy=slice(e_1_peak, e_2_peak)).sum().values)

        # add the edges of the peaks into the s parameters
        s_parm_1_edges_frac = ((spectrum.sel(energy=slice(e_1, e_2))['energy'].min() - e_1) / energy_bin_size)
        s_parm_2_edges_frac = ((e_2 - spectrum.sel(energy=slice(e_1, e_2))['energy'].max()) / energy_bin_size)
        s_parm_edges = ((s_parm_1_edges_frac * spectrum.sel(energy=slice(e_1 - energy_bin_size, e_1)).sum().values +
                         s_parm_2_edges_frac * spectrum.sel(energy=slice(e_2, e_2 + energy_bin_size)).sum().values) /
                        spectrum.sel(energy=slice(e_1_peak, e_2_peak)).sum().values)
        return s_parm + s_parm_edges

    def w_parameter_calculation(self, peak_energy_resolution, energy_domain_total,
                                energy_domain_w_left, energy_domain_w_right,
                                background_subtraction=True):
        """Calculate the W parameter for the 511 kev peak of Spectrum according to domain definitions.
        Note that the error is calculated from the use of uncertainties module in pyspectrum values.

        Parameters:
        - peak_energy_resolution (float): The energy resolution of the detector in the 511 kev energy peak.
        - energy_domain_total (tuple/list): Tuple containing the total energy domain of interest of the defect parameter
         calculation (e.g., (E1, E2)).
        - energy_domain_w_left, energy_domain_w_right (tuple/list): Tuple containing the specific energy domain
         for W parameter calculation in the right and left wing.
        - background_subtraction (bool, optional): If True, subtract background before calculation. Default is True.

        Returns:
        - ufloat: The calculated s parameter with associated uncertainty.
        """

        peak_energy_center = self.peak_energy_center_first_moment_method(ELECTRON_REST_MASS,
                                                                         peak_energy_resolution,
                                                                         background_subtraction)

        if background_subtraction:
            spectrum = spectrum_tools.subtract_background_from_spectra_peak(self.xr_spectrum(errors=True),
                                                                            unumpy.nominal_values(peak_energy_center),
                                                                            peak_energy_resolution)
        else:
            spectrum = self.xr_spectrum(errors=True)

        energy_bin_size = (spectrum['energy'][1] - spectrum['energy'][0]).values

        e_1_l = energy_domain_w_left[0]
        e_2_l = energy_domain_w_left[1]

        e_1_r = energy_domain_w_right[0]
        e_2_r = energy_domain_w_right[1]

        e_1_peak = energy_domain_total[0]
        e_2_peak = energy_domain_total[1]

        w_parm = (spectrum.sel(energy=slice(e_1_l, e_2_l)).sum().values
                  + spectrum.sel(energy=slice(e_1_r, e_2_r)).sum().values) / (
                     spectrum.sel(energy=slice(e_1_peak, e_2_peak)).sum().values)

        w_l_1_edges_frac = ((spectrum.sel(energy=slice(e_1_l, e_2_l))['energy'].min() - e_1_l) / energy_bin_size)
        w_l_2_edges_frac = ((e_2_l - spectrum.sel(energy=slice(e_1_l, e_2_l))['energy'].max()) / energy_bin_size)

        w_l_edges = ((w_l_1_edges_frac * spectrum.sel(energy=slice(e_1_l - energy_bin_size, e_1_l)).sum().values +
                      w_l_2_edges_frac * spectrum.sel(energy=slice(e_2_l, e_2_l + energy_bin_size)).sum().values) /
                     spectrum.sel(energy=slice(e_1_peak, e_2_peak)).sum().values)

        w_r_1_edges_frac = ((spectrum.sel(energy=slice(e_1_r, e_2_r))['energy'].min() - e_1_r) / energy_bin_size)
        w_r_2_edges_frac = ((e_2_r - spectrum.sel(energy=slice(e_1_r, e_2_r))['energy'].max()) / energy_bin_size)

        w_r_edges = ((w_r_1_edges_frac * spectrum.sel(energy=slice(e_1_r - energy_bin_size, e_1_r)).sum().values +
                      w_r_2_edges_frac * spectrum.sel(energy=slice(e_2_r, e_2_r + energy_bin_size)).sum().values) /
                     spectrum.sel(energy=slice(e_1_peak, e_2_peak)).sum().values)
        return w_parm + w_l_edges + w_r_edges

    @classmethod
    def from_file(cls, spectrum_file_path, energy_calibration_poly=np.poly1d([1, 0])):
        """
        load spectrum from a file which has 2 columns which tab between them
        first column is the channels/energy and the second is counts
        function return Spectrum

        Parameters:
        - spectrum_file_path () - two columns with tab(\t) between them.
         the first line is column names - channel, counts.
        - energy_calibration - numpy.poly1d([a, b]).

        Returns:
        - PASdb: db spectrum from the file in PASdb class .
        """
        # Load the pyspectrum file in form of DataFrame
        try:
            data = pd.read_csv(spectrum_file_path, sep='\t')
        except ValueError:
            raise FileNotFoundError(f"The given data file path '{spectrum_file_path}' do not exist.")
        return PASdb(data[data.columns[1]].to_numpy(), data[data.columns[0]].to_numpy(), energy_calibration_poly)
