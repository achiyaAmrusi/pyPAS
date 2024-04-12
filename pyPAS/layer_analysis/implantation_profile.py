import numpy as np


def ghosh_profile(depth_vector, positron_energy, gosh_parms):
    """
    positron profile according to [1].
    For some materials the parameters for the fit can be taken from [1,2].
    The parameters for 2 are included in this package library, and can be extracted using the function **
    Note that the there is no density in this formula and the density for which the materials parameters were given arn't include in the papers.
    To get more exact value it is recommended to run MC simulation.
        Parameters
        ----------
        - depth_vector: np.ndarray
        the vector on which the profile is calculated [nano-meters] (example: np.arange(1,1e5,1))
        - positron_energy: float
        the positron energy in keV
        - density: float
        - gosh_parms: dictionary
        the parameters for the fit which include the index - l, m, clm, Nlm, n, and B
        Returns
        -------
    [1] V.J. Ghosh et al. https://doi.org/10.1016/0169-4332(94)00331-9.
    [2] Jerzy Dryzek et al. https://doi.org/10.1016/j.nimb.2008.06.033.
    """
    l = gosh_parms['l']
    m = gosh_parms['m']
    N_lm = gosh_parms['N_lm']
    c_lm = gosh_parms['c_lm']
    z_bar = gosh_parms['B']*positron_energy**gosh_parms['n']
    return (N_lm/z_bar)*((depth_vector/(c_lm*z_bar))**l)*np.exp(-(depth_vector/(c_lm*z_bar))**m)


def makhov_profile(depth_vector, positron_energy, density, makhov_parms):
    """
    positron profile according to makovian profile[1].
    The parameters for 2 are included in this package library, and can be extracted using the function **
    To get more exact value it is recommended to run MC simulation.
        Parameters
        ----------
        - depth_vector: np.ndarray
        the vector on which the profile is calculated [nano-meters]
        - positron_energy: float
        the positron energy in keV
        - density: float
        density of the material in gr/cc
        - makhov_parms: dictionary
        the parameters for the fit which include the index - n, n, A_half
        Returns
        -------
    [1] Jerzy Dryzek et al. https://doi.org/10.1016/j.nimb.2008.06.033.
    """
    m = makhov_parms['m']
    n = makhov_parms['n']
    a_half = makhov_parms['A_half']
    z_half = a_half*positron_energy**n/density
    z_0 = z_half/(np.log(2))**(1/m)
    return m*(depth_vector**(m-1)/z_0**m)*np.exp(-(depth_vector/z_0)**m)
