import numpy as np
import pandas as pd
from pyPAS.pyspectrum import Spectrum
from pyPAS.pyPdb import PASdb

ELECTRON_REST_MASS = 511


class PAScdb:
    """CDB class represents CDB measurements.
    An instance hold the coincidence measurement list pair (after filtering).
    From the list, 2d cdb histogram, doppler broadening spectrum, and resolution spectrum can be made.

    Attributes:
    - pair_list (numpy.ndarray): 2D array of coincidence measurement pairs.

    Methods:
    - `__init__(self, measurements_pair_list)`:
      Constructor method to initialize a PAScdb instance.
      input: measurements_pair_list - 2D numpy.ndarray of spectrum counts.

    - `histogram_2d(self, energy_dynamic_range, mesh_interval)`:
      Returns the 2D CDB histogram.
      input:
        energy_dynamic_range (list): The energy range from ELECTRON_REST_MASS to the energy limit.
        mesh_interval (float): The energy interval between each 2 bins.
      return:
        2D histogram, edges of the x-axis, edges of y-axis.

    - `doppler_broadening_spectrum(self, energy_dynamic_range, mesh_interval)`:
      Returns the doppler broadening spectrum.
      Calculate the horizontal cut of the 2D CDB histogram.
      return: PASdb instance representing the doppler broadening spectrum.

    - `resolution_spectrum(self, energy_dynamic_range, mesh_interval)`:
      Returns the resolution spectrum.
      Calculate the vertical cut of the 2D CDB histogram.
      return: Spectrum instance representing the resolution spectrum.
    """

    # Constructor method
    def __init__(self, measurements_pair_list):
        """ Constructor of CDB measurement
        input:
         counts_pair_list - 2d np.ndarray of spectrum counts
         """
        self.pair_list = measurements_pair_list

    # Instance method
    """change the energy calibration polynom of Spectrum
     Parameters:
     - energy_calibration (np.poly1d): The calibration function energy_calibration(channel) -> detector energy.

    Returns:
        Nothing.
            """

    def histogram_2d(self, energy_dynamic_range, mesh_interval):
        """ return the energy coincidence two-dimensional histogram from all the cdb pairs in dataframe format.
            Parameters:
               energy_dynamic_range (list) - The energy range of the histogram away from ELECTRON_REST_MASS.
               for example, if the minimal energy measured in the coincidence was 500 kev you might consider to have a
                dynamic range of 11 kev to capture it
               mesh_interval (float) : the energy interval between each 2 bins
            Returns:
                - tuple: a tuple containing hist_2d, edges of the x-axis, edges of y-axis
                 where hist_2d is the two-dimensional histogram of the measurements, and edges_x and edges_y are
                 the histogram edges"""
        cdb_pairs = self.pair_list
        # create the mesh
        # Adjust the range and number of bins as needed
        bin_edges_x = np.arange(energy_dynamic_range[0], energy_dynamic_range[1], mesh_interval)
        bin_edges_y = np.arange(energy_dynamic_range[0], energy_dynamic_range[1], mesh_interval)

        # Create 2D histogram
        db = (cdb_pairs['energy_1'] - cdb_pairs['energy_2']) / 2
        res = (cdb_pairs['energy_1'] + cdb_pairs['energy_2']) - 2 * ELECTRON_REST_MASS
        hist_2d, x_edges, y_edges = np.histogram2d(res, db, bins=[bin_edges_x, bin_edges_y])
        return hist_2d, x_edges, y_edges

    def doppler_broadening_spectrum(self, energy_dynamic_range, mesh_interval):
        """ return the energy coincidence doppler broadening from all the cdb pairs in dataframe format.
            Parameters:
               energy_dynamic_range (list) - The energy range of the histogram away from ELECTRON_REST_MASS.
               for example, if the minimal energy measured in the coincidence was 500 kev you might consider to have a
                dynamic range of 11 kev to capture it
               mesh_interval (float) : the energy interval between each 2 bins
            Returns:
                - PASdb: doppler broadening PASdb which has the domain according to energy_dynamic_range and the sum of
                counts of the two-dimensional histogram vertical cut"""
        hist_2d, x_edges, y_edges = self.histogram_2d(energy_dynamic_range, mesh_interval)
        doppler_broadening = hist_2d[0, :] * 0
        for i in range(len(hist_2d[:, 0])):
            doppler_broadening = doppler_broadening + hist_2d[i, :]
        domain = np.array([(x_edges[i + 1] + x_edges[i]) / 2 for i in range(len(x_edges) - 1)])
        return PASdb(doppler_broadening, domain)

    def resolution_spectrum(self, energy_dynamic_range, mesh_interval):
        """ return the energy coincidence resolution spectrum from all the cdb pairs in dataframe format.
            Parameters:
               energy_dynamic_range (list) - The energy range of the histogram away from ELECTRON_REST_MASS.
               for example, if the minimal energy measured in the coincidence was 500 kev you might consider to have a
                dynamic range of 11 kev to capture it
               mesh_interval (float) : the energy interval between each 2 bins
            Returns:
                - Spectrum: Spectrum which has the domain according to energy_dynamic_range and the sum of counts of the
                two-dimensional histogram vertical cut"""
        hist_2d, x_edges, y_edges = self.histogram_2d(energy_dynamic_range, mesh_interval)
        resolution = hist_2d[:, 0] * 0
        for i in range(len(hist_2d[0, :])):
            resolution = resolution + hist_2d[:, i]
        domain = np.array([(y_edges[i + 1] + y_edges[i]) / 2 for i in range(len(y_edges) - 1)])
        return Spectrum(resolution, domain)

    @classmethod
    def from_file(cls, file_path):
        """
        load spectrum from a file which has 2 columns with \t between them.
        each line in the file represents coincidence measurement.
        The first column is the energy measured in the first detector and the energy in the second column is the
        energy measured in the second detector.
        Parameters:
            file path - the path to the coincidence measurements list
        Returns:
            CDBpas: cdb spectrum from the file in PASdb class
        """
        try:
            data = pd.read_csv(file_path, sep='\t')
        except ValueError:
            raise FileNotFoundError(f"The given data file path '{file_path}' do not exist.")
        return PAScdb(data)
