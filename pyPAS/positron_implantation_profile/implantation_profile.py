import numpy as np
import xarray as xr
from warnings import warn


def ghosh_profile(depth_vector, positron_energy, density, gosh_parms):
    """
    positron positron_implantation_profile according to [1].
    For some materials the parameters for the fit can be taken from [1,2].
    The parameters for 2 are included in this package library,
     and can be extracted using the function PyPAS.positron_implantation_profile.gosh_material_parmeters
    To get more exact value it is recommended to run MC simulation.
    Parameters
    ----------
    - depth_vector: np.ndarray
    the vector on which the positron_implantation_profile is calculated [micro-meters] (example: np.arange(1,1e5,1))
    - positron_energy: float
    the positron energy in keV
    - density: float
    - gosh_parms: dictionary
    the parameters for the fit which include the index - l, m, clm, Nlm, n, and B
    for example, aluminum parameters can be extracted using gosh_material_parmeters().iloc[4]
    Returns
    -------
    The implemented thermalized positron distribution  [positrons/micrometer/s]

    Reference
    ---------
    [1] V.J. Ghosh et al. https://doi.org/10.1016/0169-4332(94)00331-9.
    [2] Jerzy Dryzek et al. https://doi.org/10.1016/j.nimb.2008.06.033.
    """
    if ('l' not in gosh_parms) or ('m' not in gosh_parms) or ('N_lm' not in gosh_parms) or (
            'c_lm' not in gosh_parms) or ('B' not in gosh_parms) or ('n' not in gosh_parms) or (
            'density' not in gosh_parms):
        raise KeyError("The function requires gosh_parms to be a pd.Series from DataFrame gosh_material_parmeters()")
    l = gosh_parms['l']
    m = gosh_parms['m']
    N_lm = gosh_parms['N_lm']
    c_lm = gosh_parms['c_lm']
    z_bar = (gosh_parms['B'] * (gosh_parms['density'] / density)) * positron_energy ** gosh_parms['n']
    return xr.DataArray((N_lm / z_bar) * ((depth_vector / (c_lm * z_bar)) ** l) * np.exp(-(depth_vector / (c_lm * z_bar)) ** m),
                        coords={'x': depth_vector})


def makhov_profile(depth_vector, positron_energy, density, makhov_parms):
    """
    positron positron_implantation_profile according to makovian positron_implantation_profile[1].
    The parameters for 2 are included in this package library,
     and can be extracted using the function PyPAS.positron_implantation_profile.makhov_material_parmeters
    To get more exact value it is recommended to run MC simulation.
        Parameters
        ----------
        - depth_vector: np.ndarray
        the vector on which the positron_implantation_profile is calculated [micro-meters]
        - positron_energy: float
        the positron energy in keV
        - density: float
        density of the material in gr/cc
        - makhov_parms: dictionary
        the parameters for the fit which include the index - n, m, A_half
        for example, aluminum parameters are makhov_material_parmeters().iloc[4]
        Returns
        -------
    The implemented thermalized positron distribution  [positrons/micrometer/s]

    [1] Jerzy Dryzek et al. https://doi.org/10.1016/j.nimb.2008.06.033.
    """
    if ('m' not in makhov_parms) or ('n' not in makhov_parms) or ('A_half' not in makhov_parms):
        raise KeyError(
            "The function requires makhov_parms to be a pd.Series from DataFrame makhov_material_parameters()")

    m = makhov_parms['m']
    n = makhov_parms['n']
    a_half = makhov_parms['A_half']

    z_half = a_half * positron_energy ** n / density
    z_0 = z_half / (np.log(2)) ** (1 / m)
    return xr.DataArray(m * (depth_vector ** (m - 1) / z_0 ** m) * np.exp(-(depth_vector / z_0) ** m),
                        coords={'x': depth_vector})


def multilayer_implementation_profile(positron_energy: float, depth_vector: np.ndarray,
                                      widths: list, materials_parameters: list, densities: list,
                                      implementation_profile_function=ghosh_profile):
    """
    Calculate the positrons implementation profile in a multilayer sample, meaning,
    the sample is composed from a number of layers and in each layer is composed of a different material.
    The profile of the positrons in each material is obtained by the profile function (Makhov or Ghosh) and
    Its respected parameters which are obtained by gosh_material_parmeters and makhov_material_parmeters

    Parameters
    ----------
    - positron_energy: float
        the positron energy in keV
    - depth_vector: np.ndarray
        the vector on which the positron_implantation_profile is calculated [micro-meters]
    - widths: list
        floats list of each layer width
    - materials_parameters: list
        a list in which each element is pd.Series which represent the material parameters of the layer.
        for example, aluminum parameters are gosh_material_parmeters().iloc[4]
    - densities: list
         floats list of each layer density
    - implementation_profile_function: Callable (Default ghosh_profile)
        The implementation profile function type (ghosh_profile or makhov_profile)
        The type needs to aligen with the parameters given inmaterials_parameters
    Returns
    -------
    The implemented thermalized positron distributiom [positrons/micrometer/s]
        """

    implementation_profile = np.zeros_like(depth_vector)

    # if the depth vector is too short or too long rais warning
    if depth_vector[-1] > sum(widths):
        warn('The implementation depth is larger than the size of all the layer\n' \
             'the extra depth is caculated according to the last layer')
    if depth_vector[-1] < sum(widths[:-1]):
        warn('the implentation depth dose not reach last layer')

    # calculate each layer start and end indices #
    layers_indices = []
    implementation_grid_indices = xr.DataArray(range(depth_vector.size), coords={'x': depth_vector})
    layer_first_index = 0
    layer_last_index = 0
    total_width = 0

    for width in widths:
        total_width = total_width + width
        layer_first_index = layer_last_index
        layer_last_index = implementation_grid_indices.interp(x=total_width).item()
        # if the layers are contained in the implememtation profile
        if depth_vector[-1] > total_width:
            layer_last_index = int(np.ceil(layer_last_index))
            layers_indices.append([layer_first_index, layer_last_index])
        else:
            layers_indices.append([layer_first_index, depth_vector.size - 1])
            # The next layer is not included in the depth vector so break
            break
    # if the layers don't cover all the implementation profile pretend the last layer is infinite
    if depth_vector[-1] > total_width:
        layers_indices.append([layer_first_index, depth_vector.size - 1])
        materials_parameters.append(materials_parameters[-1])
        densities.append(densities[-1])

    # calculate the implementation depth asa function of the depth for each layer #
    for index, layer_indices in enumerate(layers_indices):
        implementation_profile[layer_indices[0]:layer_indices[1]] = implementation_profile_function(
            depth_vector[layer_indices[0]:layer_indices[1]],
            positron_energy,
            densities[index],
            materials_parameters[-1])
    return xr.DataArray(implementation_profile, coords={'x': depth_vector})
