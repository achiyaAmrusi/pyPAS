import numpy as np
import pandas as pd

from uncertainties import nominal_value, std_dev
from scipy.optimize import curve_fit
from pyPAS.pyvedb.sample import Sample, Material, Layer
from pyPAS.pyvedb.positron_profile import fast_positrons_annihilation_profile
from pyPAS.pyvedb.annhilation_channels import profile_annihilation_fraction


class SurfaceBulkDefectsOptimization:
    """
    """

    def __init__(self, positron_implementation_profiles: list,
                 s_measurement: pd.Series,
                 initial_guess: Sample,
                 num_of_mesh_cells=10000):
        """
        Parameters
        ----------
         - positron_implementation_profiles: list
          a list of the implementation profile for each energy
        - s_measurement: pd.Series
         a series of the s measurements, note it as to be the same length as positron_implementation_profiles
         - initial_guess: Sample
         2 parameter for initial guess
        """
        self.positron_implementation_profiles = positron_implementation_profiles
        self.initial_sample = initial_guess
        self.energies = s_measurement.index
        self.s_measurement = np.array([nominal_value(s) for s in s_measurement])
        self.s_measurement_dev = np.array([std_dev(s) for s in s_measurement])
        self.num_of_mesh_cells = num_of_mesh_cells

    def make_sample(self, alpha_surface, lambda_bulk, lambda_defects):
        m1 = Material(1, 0, lambda_bulk, lambda_defects)
        l1 = Layer(self.initial_sample.size, m1)
        return Sample([l1], alpha_surface)

    def rate_matrix(self, sample):
        annihilation_channel_rate_matrix = np.zeros((3, len(self.energies)))
        for i, energy in enumerate(self.energies):
            positron_implementation_profile = self.positron_implementation_profiles[i]
            positron_profile_sol = fast_positrons_annihilation_profile(positron_implementation_profile,
                                                                       sample,
                                                                       num_of_mesh_cells=self.num_of_mesh_cells)
            df = profile_annihilation_fraction(positron_profile_sol, sample)
            annihilation_channel_rate_matrix[0, i] = df.loc[(0, 'surface')].item()
            annihilation_channel_rate_matrix[1, i] = df.loc[(0, 'bulk')].item()
            annihilation_channel_rate_matrix[2, i] = df.loc[(0, 'defects')].item()
        return np.transpose(annihilation_channel_rate_matrix)

    def rates(self, energies, alpha_surface, lambda_bulk, lambda_defects):
        sample = self.make_sample(alpha_surface, lambda_bulk, lambda_defects)
        # find s_parm using linear regression
        annihilation_channel_rate_matrix = self.rate_matrix(sample)
        s_vec = np.linalg.lstsq(annihilation_channel_rate_matrix, self.s_measurement, rcond=None)[0]
        s_sample = annihilation_channel_rate_matrix @ s_vec
        if np.any(s_vec > 1):
            return np.inf
        return s_sample

    def optimize_parameters(self, bounds=None):
        if bounds is None:
            bounds = (0, np.inf)
        layer_material = (self.initial_sample.layers[0]).material
        initial_guess = [self.initial_sample.surface_capture_rate,
                         layer_material.annihilation_rates['bulk'], layer_material.annihilation_rates['defects']]

        # optimize only the rates
        parm, cov = curve_fit(f=self.rates, xdata=self.energies, ydata=self.s_measurement,
                              sigma=self.s_measurement_dev, p0=initial_guess, bounds=bounds)
        return parm, cov


class SurfaceBulkOptimization:
    """
    """

    def __init__(self, positron_implementation_profiles: list,
                 s_measurement: pd.Series,
                 initial_guess: Sample,
                 num_of_mesh_cells=10000):
        """
        Parameters
        ----------
         - positron_implementation_profiles: list
          a list of the implementation profile for each energy
        - s_measurement: pd.Series
         a series of the s measurements, note it as to be the same length as positron_implementation_profiles
         - initial_guess: Sample
         2 parameter for initial guess
        """
        self.positron_implementation_profiles = positron_implementation_profiles
        self.initial_sample = initial_guess
        self.energies = s_measurement.index
        self.s_measurement = np.array([nominal_value(s) for s in s_measurement])
        self.s_measurement_dev = np.array([std_dev(s) for s in s_measurement])
        self.num_of_mesh_cells = num_of_mesh_cells

    def make_sample(self, alpha_surface, lambda_bulk):
        material = Material(1, 0, lambda_bulk, 0)
        layer = Layer(self.initial_sample.size, material)
        return Sample([layer], alpha_surface)

    def rate_matrix(self, sample):
        annihilation_channel_rate_matrix = np.zeros((2, len(self.energies)))
        for i, energy in enumerate(self.energies):
            positron_implementation_profile = self.positron_implementation_profiles[i]
            positron_profile_sol = fast_positrons_annihilation_profile(positron_implementation_profile,
                                                                       sample,
                                                                       num_of_mesh_cells=self.num_of_mesh_cells)
            df = profile_annihilation_fraction(positron_profile_sol, sample)
            annihilation_channel_rate_matrix[0, i] = df.loc[(0, 'surface')].item()
            annihilation_channel_rate_matrix[1, i] = df.loc[(0, 'bulk')].item()
        return np.transpose(annihilation_channel_rate_matrix)

    def rates(self, energies, alpha_surface, lambda_bulk):
        sample = self.make_sample(alpha_surface, lambda_bulk)
        # find s_parm using linear regression
        annihilation_channel_rate_matrix = self.rate_matrix(sample)
        s_vec = np.linalg.lstsq(annihilation_channel_rate_matrix, self.s_measurement, rcond=None)[0]
        s_sample = annihilation_channel_rate_matrix @ s_vec
        if np.any(s_vec > 1):
            return np.inf * s_sample
        return s_sample

    def optimize_parameters(self, bounds=None):
        if bounds is None:
            bounds = (0, np.inf)
        layer_material = (self.initial_sample.layers[0]).material
        initial_guess = [self.initial_sample.surface_capture_rate,
                         layer_material.annihilation_rates['bulk']]
        # optimize only the rates
        parm, cov = curve_fit(f=self.rates, xdata=self.energies, ydata=self.s_measurement,
                              sigma=self.s_measurement_dev, p0=initial_guess, bounds=bounds)
        return parm, cov


class SurfaceDefectsOptimization:
    """
    """

    def __init__(self, positron_implementation_profiles: list,
                 s_measurement: pd.Series,
                 initial_guess: Sample,
                 num_of_mesh_cells=10000):
        """
        Parameters
        ----------
         - positron_implementation_profiles: list
          a list of the implementation profile for each energy
        - s_measurement: pd.Series
         a series of the s measurements, note it as to be the same length as positron_implementation_profiles
         - initial_guess: Sample
         2 parameter for initial guess
        """
        self.positron_implementation_profiles = positron_implementation_profiles
        self.initial_sample = initial_guess
        self.energies = s_measurement.index
        self.s_measurement = np.array([nominal_value(s) for s in s_measurement])
        self.s_measurement_dev = np.array([std_dev(s) for s in s_measurement])
        self.num_of_mesh_cells = num_of_mesh_cells

    def make_sample(self, alpha_surface, lambda_defects):
        lambda_bulk = self.initial_sample.layers[0].material.annihilation_rates['bulk']
        m1 = Material(1, 0, lambda_bulk, lambda_defects)
        l1 = Layer(self.initial_sample.size, m1)
        return Sample([l1], alpha_surface)

    def rate_matrix(self, sample):
        annihilation_channel_rate_matrix = np.zeros((3, len(self.energies)))
        for i, energy in enumerate(self.energies):
            positron_implementation_profile = self.positron_implementation_profiles[i]
            positron_profile_sol = fast_positrons_annihilation_profile(positron_implementation_profile,
                                                                       sample,
                                                                       num_of_mesh_cells=self.num_of_mesh_cells)
            df = profile_annihilation_fraction(positron_profile_sol, sample)
            annihilation_channel_rate_matrix[0, i] = df.loc[(0, 'surface')].item()
            annihilation_channel_rate_matrix[1, i] = df.loc[(0, 'bulk')].item()
            annihilation_channel_rate_matrix[2, i] = df.loc[(0, 'defects')].item()
        return np.transpose(annihilation_channel_rate_matrix)

    def rates(self, energies, alpha_surface, lambda_defects):
        sample = self.make_sample(alpha_surface, lambda_defects)
        # find s_parm using linear regression
        annihilation_channel_rate_matrix = self.rate_matrix(sample)
        s_vec = np.linalg.lstsq(annihilation_channel_rate_matrix, self.s_measurement, rcond=None)[0]
        s_sample = annihilation_channel_rate_matrix @ s_vec
        if np.any(s_vec > 1):
            return np.inf
        return s_sample

    def optimize_parameters(self, bounds=None):
        if bounds is None:
            bounds = (0, np.inf)
        layer_material = (self.initial_sample.layers[0]).material
        initial_guess = [self.initial_sample.surface_capture_rate, layer_material.annihilation_rates['defects']]

        # optimize only the rates
        parm, cov = curve_fit(f=self.rates, xdata=self.energies, ydata=self.s_measurement,
                              sigma=self.s_measurement_dev, p0=initial_guess, bounds=bounds)
        return parm, cov


class DefectsOptimization:
    """
        """

    def __init__(self, positron_implementation_profiles: list,
                 s_measurement: pd.Series,
                 initial_guess: Sample,
                 num_of_mesh_cells=10000):
        """
            Parameters
            ----------
             - positron_implementation_profiles: list
              a list of the implementation profile for each energy
            - s_measurement: pd.Series
             a series of the s measurements, note it as to be the same length as positron_implementation_profiles
             - initial_guess: Sample
             2 parameter for initial guess
            """
        self.positron_implementation_profiles = positron_implementation_profiles
        self.initial_sample = initial_guess
        self.energies = s_measurement.index
        self.s_measurement = np.array([nominal_value(s) for s in s_measurement])
        self.s_measurement_dev = np.array([std_dev(s) for s in s_measurement])
        self.num_of_mesh_cells = num_of_mesh_cells

    def make_sample(self, lambda_defects):
        lambda_bulk = self.initial_sample.layers[0].material.annihilation_rates['bulk']
        alpha_surface = self.initial_sample.surface_capture_rate
        material = Material(1, 0, lambda_bulk, lambda_defects)
        single_layer = Layer(self.initial_sample.size, material)
        return Sample([single_layer], alpha_surface)

    def rate_matrix(self, sample):
        annihilation_channel_rate_matrix = np.zeros((3, len(self.energies)))
        for i, energy in enumerate(self.energies):
            positron_implementation_profile = self.positron_implementation_profiles[i]
            positron_profile_sol = fast_positrons_annihilation_profile(positron_implementation_profile,
                                                                       sample,
                                                                       num_of_mesh_cells=self.num_of_mesh_cells)
            df = profile_annihilation_fraction(positron_profile_sol, sample)
            annihilation_channel_rate_matrix[0, i] = df.loc[(0, 'surface')].item()
            annihilation_channel_rate_matrix[1, i] = df.loc[(0, 'bulk')].item()
            annihilation_channel_rate_matrix[2, i] = df.loc[(0, 'defects')].item()
        return np.transpose(annihilation_channel_rate_matrix)

    def rates(self, energies, lambda_defects):
        sample = self.make_sample(lambda_defects)
        # find s_parm using linear regression
        annihilation_channel_rate_matrix = self.rate_matrix(sample)
        s_vec = np.linalg.lstsq(annihilation_channel_rate_matrix, self.s_measurement, rcond=None)[0]
        s_sample = annihilation_channel_rate_matrix @ s_vec
        if np.any(s_vec > 1):
            return np.inf
        return s_sample

    def optimize_parameters(self, bounds=None):
        if bounds is None:
            bounds = (0, np.inf)
        layer_material = (self.initial_sample.layers[0]).material
        initial_guess = [layer_material.annihilation_rates['defects']]

        # optimize only the rates
        parm, cov = curve_fit(f=self.rates, xdata=self.energies, ydata=self.s_measurement,
                              sigma=self.s_measurement_dev, p0=initial_guess, bounds=bounds)
        return parm, cov
