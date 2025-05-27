import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, std_devs
from scipy.optimize import curve_fit
from pyPAS.sample import Sample, Material, Layer
from pyPAS.positron_annihilation_profile.positron_profile_solver import positrons_annihilation_profile_solver
from pyPAS.positron_annihilation_profile.annhilation_channels import profile_annihilation_fraction
import warnings


class OneBulkDiffusionLengthOptimization:
    """
    Optimization to find the bulk diffusion length of a sample.
    This optimization is conducted for one layer sample (surface and bulk).
    Note, given initial Sample with number of annihilation rates, they are summed into one bulk annihilation rate.
    The equation solved for the optimization is the effective equation -
    d**2c(z)/dz**2 -d/dz((v_d(z)/D(z))*c(z))+I(z)-(1/L_eff**2)*c(x) = 0
    where L_eff - > The diffusion length /effective annihilation rate

    TODO: this optimization does not include drift velocity (when there are electric fields)
    Parameters
    ----------
    - positron_implantation_profiles: list
    The profile of the positrons in the sample, this is a list of xarray
    - s_measurement: pd.Series
    The measurement ufloat values of S parameters with the energies as index
    - initial_guess: Sample
    A Sample which is the initial guess of the sample (meaning with diffusion parameter and annihilation rate
    Note, In order to have initial guess of specific diffusion length, L = (D/λ)**0.5
    - num_of_mesh_cells: int Default 10000
    number of cells for the discrimination of the space for the solution of the positron transport equation
    """

    def __init__(self, positron_implantation_profiles: list,
                 s_measurement: pd.Series,
                 initial_guess: Sample,
                 num_of_mesh_cells=10000):
        """
        Parameters
        ----------
         - positron_implantation_profiles: list
          a list of the implantation positron_implantation_profile for each energy
        - s_measurement: pd.Series
         a series of the s measurements, note it as to be the same length as positron_implantation_profiles
         - initial_guess: Sample
         2 parameter for initial guess
        """
        self.positron_implantation_profiles = positron_implantation_profiles
        self.initial_sample = initial_guess
        self.energies = s_measurement.index
        self.s_measurement = nominal_values(s_measurement)
        self.s_measurement_dev = std_devs(s_measurement)
        self.num_of_mesh_cells = num_of_mesh_cells

    def make_sample(self, eff_surface_capture_rate, eff_rate):
        """
        Makes sample with the parameter given alpha_surface and diffusion coefficient
        The annihilation rate in normlized to 1.

        Parameters
        ----------
        - surface_capture_rate: float
        The surface capture rate
        - eff_rate: float
        The effective rate is the inverse of the diffusion length squared
        Returns
        -------
        sample: Sample
        """
        material = Material(diffusion=1, mobility=0, annihilation_rate_bulk=eff_rate)
        layer = Layer(starting_point=0, width=self.initial_sample.size, material=material)
        return Sample([layer], eff_surface_capture_rate)

    def rate_matrix(self, sample):
        """
        Calculates the rate matrix of annihilation fractions for a given sample and multiple positron implantation profiles.

        For each implantation profile, the function:
        1. Solves the positron annihilation profile using the provided sample and profile.
        2. Calculates the annihilation fraction in three distinct regions: surface, bulk, and defects.

        Parameters
        ----------
        sample : Sample
            The sample in which positron annihilation takes place.

        Returns
        -------
        annihilation_channel_rate_matrix : np.ndarray
            A transposed matrix where each row corresponds to a profile and contains two values:
            the fraction of positrons annihilating in the surface, bulk, respectively.
        """
        annihilation_channel_rate_matrix = np.zeros((2, len(self.energies)))
        for i, energy in enumerate(self.energies):
            positron_implantation_profile = self.positron_implantation_profiles[i]
            positron_profile_sol = positrons_annihilation_profile_solver(positron_implantation_profile,
                                                                         sample,
                                                                         mesh_size=self.num_of_mesh_cells)
            df = profile_annihilation_fraction(positron_profile_sol, sample)
            annihilation_channel_rate_matrix[0, i] = df.loc[(0, 'surface')].item()
            annihilation_channel_rate_matrix[1, i] = df.loc[(0, 'bulk')].item()
        return np.transpose(annihilation_channel_rate_matrix)

    def s_parameter_calculation(self, energies, eff_surface_capture_rate, eff_rate):
        """
        For given sample, given the sample parameters, function calculate the expected S parameter per energy

        Parameters
        ----------
        - surface_capture_rate: float
        The surface capture rate
        - eff_rate: float
        The effective rate is the inverse of the diffusion length squared
        Returns
        -------
        s_sample: np.ndarray
        the expected s parameters
        """
        sample = self.make_sample(eff_surface_capture_rate, eff_rate)
        # find s_parm using linear regression
        annihilation_channel_rate_matrix = self.rate_matrix(sample)
        s_vec = np.linalg.lstsq(annihilation_channel_rate_matrix, self.s_measurement, rcond=None)[0]
        s_sample = annihilation_channel_rate_matrix @ s_vec

        if np.any(s_vec >= 1):
            # if the s value is above 1 make the result high
            return s_sample*np.inf
        return s_sample

    def optimize_diffusion_length(self, bounds=None):
        """
        Function find optimization for the diffusion length in the sample layers

        Parameters
        ----------
        - bounds: list, Default [(0, 0), (np.inf, np.inf)]
        The bounds for the diffusion length parameters
        Returns
        -------
        s_sample: np.ndarray
        the expected s parameters
        """
        layer_material = (self.initial_sample.layers[0]).material
        # The effective rate is the inverse of the diffusion length squared
        effective_rate = np.array(list(layer_material.rates.values())).sum()/layer_material.diffusion
        effective_surface_capture_rate = self.initial_sample.surface_capture_rate/layer_material.diffusion
        initial_guess = [effective_surface_capture_rate, effective_rate]
        # set bound of the effective diffusion and the surface_capture_rate
        if bounds is None:
            bounds = [(0, 0), (np.inf, np.inf)]
        else:
            if bounds[1][1] != 0 and bounds[1][1] != 0:
                lower_bounds = (1/bounds[1][0], 1/bounds[1][1]**0.5)
            else:
                warnings.warn("upper bound of L_a and L_b cannot be 0 use small number instead ", ValueError)
                lower_bounds = (0, 0)
            if bounds[0][0] != 0 and bounds[0][1] != 0:
                upper_bounds = (1/bounds[0][0], 1/bounds[0][1]**0.5)
            else:
                upper_bounds = (np.inf, np.inf)
            bounds = [lower_bounds, upper_bounds]
        parm, cov = curve_fit(f=self.s_parameter_calculation, xdata=self.energies, ydata=self.s_measurement,
                              sigma=self.s_measurement_dev, absolute_sigma=True,
                              p0=initial_guess, bounds=bounds)
        effective_rate_fit = ufloat(parm[1], cov[1, 1]**0.5)
        effective_surface_capture_rate_fit = ufloat(parm[0], cov[0, 0]**0.5)
        effective_length = {'layer_0': 1/effective_rate_fit**(1/2), 'surface': 1/effective_surface_capture_rate_fit}

        return effective_length
