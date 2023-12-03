import xarray as xr
import numpy as np
from uncertainties import ufloat
import lmfit
from pyspectrum.spectrum import Spectrum
from pyspectrum import spectrum_tools

ELECTRON_REST_MASS = 511


class DBpas(Spectrum):
    """ class that holds the properties of Spectrum but also include s and w calculation  """

    def spectrum_s_calculation(self, peak_energy_resolution, energy_domain_total, energy_domain_s,
                               background_subtraction=True):
        """calculate the s parameter for a xarray type pyspectrum according to domain definitions
        note that the error is calculated from the use of uncertainties module in pyspectrum values """

        peak_energy_center = self.peak_energy_center_first_moment_method(ELECTRON_REST_MASS, peak_energy_resolution,
                                                                         background_subtraction)

        if background_subtraction:
            spectrum = spectrum_tools.subtract_background_from_spectra_peak(self.xr_spectrum(errors=True),
                                                                            peak_energy_center, peak_energy_resolution)
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

    def spectrum_w_calculation(self, peak_energy_resolution, energy_domain_total,
                               energy_domain_w_left, energy_domain_w_right,
                               background_subtraction=True):
        """calculate the w parameter for a xarray type pyspectrum according to domain definitions
            note that the error is calculated from the use of uncertainties module in pyspectrum values """

        peak_center = self.peak_energy_center_first_moment_method(ELECTRON_REST_MASS, peak_energy_resolution,
                                                                  background_subtraction)

        if background_subtraction:
            spectrum = spectrum_tools.subtract_background_from_spectra_peak(self.xr_spectrum(errors=True),
                                                                            peak_center, peak_energy_resolution)
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
