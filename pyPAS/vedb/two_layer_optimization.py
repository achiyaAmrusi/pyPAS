import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, std_devs
from scipy.optimize import curve_fit
from pyPAS.sample import Sample, Material, Layer
from pyPAS.positron_annihilation_profile.positron_profile_solver import positrons_annihilation_profile_solver
from pyPAS.positron_annihilation_profile.annhilation_channels import profile_annihilation_fraction
import warnings


class TwoBulkDiffusionLengthOptimization:
    """
    Optimization to find the bulk diffusion length of a sample.
    This optimization is conducted for one layer sample (surface and bulk).
    Note, given initial Sample with number of annihilation rates, they are summed into one bulk annihilation rate.
    Also, this optimization does not include drift velocity (when there are electric fields)
    Parameters
    ----------
    - positron_implementation_profiles: list
    The profile of the positrons in the sample, this is a list of xarray
    - s_measurement: pd.Series
    The measurement ufloat values of S parameters with the energies as index
    - initial_guess: Sample
    A Sample which is the initial guess of the sample (meaning with diffusion parameter and annihilation rate
    Note, In order to have initial guess of specific diffusion length, L = (D/λ)**0.5
    - num_of_mesh_cells: int Default 10000
    number of cells for the discrimination of the space for the solution of the positron transport equation
    """

    def __init__(self, positron_implementation_profiles: list,
                 s_measurement: pd.Series,
                 initial_guess: Sample,
                 num_of_mesh_cells=10000):
        """
        Parameters
        ----------
         - positron_implementation_profiles: list
          a list of the implementation positron_implantation_profile for each energy
        - s_measurement: pd.Series
         a series of the s measurements, note it as to be the same length as positron_implementation_profiles
         - initial_guess: Sample
         2 parameter for initial guess
        """
        self.positron_implementation_profiles = positron_implementation_profiles
        self.initial_sample = initial_guess
        self.energies = s_measurement.index
        self.s_measurement = nominal_values(s_measurement)
        self.s_measurement_dev = std_devs(s_measurement)
        self.num_of_mesh_cells = num_of_mesh_cells

    def make_sample(self, eff_surface_capture_rate, eff_rate_0, eff_rate_1):
        """
        Makes sample with the parameter given alpha_surface and diffusion coefficient
        The annihilation rate in normlized to 1.
        Parameters
        ----------
        - surface_capture_rate: float
        The surface capture rate
        - eff_rate_0, eff_rate_1: float
        The effective rate is the inverse of the diffusion length squared
        Returns
        -------
        sample: Sample
        """
        material_0 = Material(diffusion=1, mobility=0, annihilation_rate_bulk=eff_rate_0)
        material_1 = Material(diffusion=1, mobility=0, annihilation_rate_bulk=eff_rate_1)
        layer_0 = Layer(starting_point=0, width=self.initial_sample.layers[0].width, material=material_0)
        layer_1 = Layer(starting_point=0, width=self.initial_sample.layers[1].width, material=material_1)
        return Sample([layer_0, layer_1], eff_surface_capture_rate)

    def rate_matrix(self, sample):
        """
        Calculates the rate matrix of annihilation fractions for a given sample and multiple positron implementation profiles.

        For each implementation profile, the function:
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
        annihilation_channel_rate_matrix = np.zeros((3, len(self.energies)))
        for i, energy in enumerate(self.energies):
            positron_implementation_profile = self.positron_implementation_profiles[i]
            positron_profile_sol = positrons_annihilation_profile_solver(positron_implementation_profile,
                                                                         sample,
                                                                         mesh_size=self.num_of_mesh_cells)
            df = profile_annihilation_fraction(positron_profile_sol, sample)
            annihilation_channel_rate_matrix[0, i] = df.loc[(0, 'surface')].item()
            annihilation_channel_rate_matrix[1, i] = df.loc[(0, 'bulk')].item()
            annihilation_channel_rate_matrix[2, i] = df.loc[(1, 'bulk')].item()
        return np.transpose(annihilation_channel_rate_matrix)

    def s_parameter_calculation(self, energies, eff_surface_capture_rate, eff_rate_0, eff_rate_1):
        """
        For given sample, given the sample parameters, function calculate the expected S parameter per energy

        Parameters
        ----------
        - surface_capture_rate: float
        The surface capture rate
        - eff_rate_0, eff_rate_1: float
        The effective rate is the inverse of the diffusion length squared
        Returns
        -------
        s_sample: np.ndarray
        the expected s parameters
        """
        sample = self.make_sample(eff_surface_capture_rate, eff_rate_0, eff_rate_1)
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
        # initial guess
        layer_material_0 = (self.initial_sample.layers[0]).material
        layer_material_1 = (self.initial_sample.layers[1]).material
        # The effective rate is the inverse of the diffusion length squared
        effective_rate_0 = np.array(list(layer_material_0.rates.values())).sum()/layer_material_0.diffusion
        effective_rate_1 = np.array(list(layer_material_1.rates.values())).sum() / layer_material_1.diffusion
        effective_surface_capture_rate = self.initial_sample.surface_capture_rate/layer_material_0.diffusion

        initial_guess = [effective_surface_capture_rate, effective_rate_0, effective_rate_1]

        # set bound of the effective diffusion and the surface_capture_rate
        if bounds is None:
            bounds = [(0, 0, 0), (np.inf, np.inf, np.inf)]
        else:
            if bounds[1][1] != 0 and bounds[1][1] != 0 and bounds[1][2] != 0:
                lower_bounds = (1/bounds[1][0], 1/bounds[1][1]**0.5, 1/bounds[1][2]**0.5)
            else:
                warnings.warn("upper bound of L_a and L_b cannot be 0 use small number instead ", ValueError)
                lower_bounds = (0, 0, 0)
            if bounds[0][0] != 0 and bounds[0][1] != 0 and bounds[0][2] != 0:
                upper_bounds = (1/bounds[0][0], 1/bounds[0][1]**0.5, 1/bounds[0][2]**0.5)
            else:
                upper_bounds = (np.inf, np.inf, np.inf)
            bounds = [lower_bounds, upper_bounds]
        # fit the effective rates
        parm, cov = curve_fit(f=self.s_parameter_calculation, xdata=self.energies, ydata=self.s_measurement,
                              sigma=self.s_measurement_dev, p0=initial_guess, bounds=bounds)
        # turn the effective rates into diffusion length
        effective_rate_fit_0 = ufloat(parm[1], cov[1, 1]**0.5)
        effective_rate_fit_1 = ufloat(parm[2], cov[2, 2] ** 0.5)
        effective_surface_capture_rate_fit = ufloat(parm[0], cov[0, 0]**0.5)
        effective_length = {'layer_1': 1/effective_rate_fit_1**(1/2), 'layer_0': 1/effective_rate_fit_0**(1/2),
                            'surface': 1/effective_surface_capture_rate_fit}

        return effective_length

