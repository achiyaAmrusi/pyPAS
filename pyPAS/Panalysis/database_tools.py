import xarray as xr


def spectrum_s_w_calculation(spectrum,
                             energy_domain_peak, energy_domain_s,
                             energy_domain_w_l, energy_domain_w_r):
    """ calculate the s and w parameters of an xarray spectrum given the domain for the parameter integration"""
    # calculating s from spectrum
    s_parameter = spectrum_s_calculation(spectrum, energy_domain_peak, energy_domain_s)
    w_parameter = spectrum_w_calculation(spectrum, energy_domain_peak, energy_domain_w_l, energy_domain_w_r)
    return s_parameter, w_parameter


def spectrum_s_calculation(spectrum, energy_domain_peak, energy_domain_s):
    """calculate the s parameter for a xarray type spectrum according to domain definitions
    note that the error is calculated from the use of uncertainties module in spectrum values """
    e_1 = energy_domain_s[0]
    e_2 = energy_domain_s[1]

    e_1_peak = energy_domain_peak[0]
    e_2_peak = energy_domain_peak[1]

    energy_bin_size = (spectrum['energy'][1] - spectrum['energy'][0]).values

    if not isinstance(spectrum, xr.DataArray):
        print("""spectrum is not in x array form""")
        return None
    s_parm = spectrum.sel(energy=slice(e_1, e_2)).sum().values / (
        spectrum.sel(energy=slice(e_1_peak, e_2_peak)).sum().values)
    # add the edges of the peaks into the s parameters
    s_edges = (spectrum.sel(energy=slice(e_1 - 2 * energy_bin_size, e_1 - energy_bin_size)).sum().values +
               spectrum.sel(energy=slice(e_2 + energy_bin_size, e_2 + 2 * energy_bin_size)).sum().values) / (
                  spectrum.sel(energy=slice(e_1_peak, e_2_peak)).sum().values)
    return s_parm + s_edges


def spectrum_w_calculation(spectrum, energy_domain_peak, energy_domain_w_left, energy_domain_w_right):
    """calculate the w parameter for a xarray type spectrum according to domain definitions
        note that the error is calculated from the use of uncertainties module in spectrum values """
    e_1_l = energy_domain_w_left[0]
    e_2_l = energy_domain_w_left[1]

    e_1_r = energy_domain_w_right[0]
    e_2_r = energy_domain_w_right[1]

    e_1_peak = energy_domain_peak[0]
    e_2_peak = energy_domain_peak[1]

    s_parm = (spectrum.sel(energy=slice(e_1_l, e_2_l)).sum().values
              + spectrum.sel(energy=slice(e_1_r, e_2_r)).sum().values) / (
                 spectrum.sel(energy=slice(e_1_peak, e_2_peak)).sum().values)
    return s_parm


def read_hist(hist_path_path):
    return 0