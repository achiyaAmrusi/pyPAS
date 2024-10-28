import numpy as np
import pandas as pd
from pyspectrum.detector_parser import TimeChannelParser

ELECTRON_REST_MASS = 511


class PasCoincidenceFilter:
    """
    Coincidence filter for Positron Annihilation Spectroscopy(PAS). The class filters two files/ dataframes
     which two columns, one of energy bin (or channel with calibration), and the other is time stamp from the analyzer.
     The parser then checks which of the counts in the file happen in the same time and if they are coincidence.
    The definition of Coincidence PAS event is such the time difference of the two events os less than max_time_interval
    and that their energy sum difference from 2*ELECTRON_REST_MASS is in the range of #number_of_cdb_sigma of sigma.

    Note, the energy filter use SIGMA for the filter not FWHM.
      SIGMA = FWHM / (2 * np.log(2)) ** 0.5.

    Methods
    -------
    from_files(cls, det_1_time_channel_path: str, det_2_time_channel_path: str,
               sep=' ', skiprows=5,
               det_1_energy_calibration_poly=np.poly1d([1, 0]), det_2_energy_calibration_poly=np.poly1d([1, 0]),
               det_1_energy_resolution=1, det_2_energy_resolution=1,
               max_time_interval=10)
     return a dataframe of the coincidence cases

    from_dataframe(cls, det_1_time_channel: pd.DataFrame, det_2_time_channel: pd.DataFrame,
                   det_1_energy_calibration_poly=np.poly1d([1, 0]), det_2_energy_calibration_poly=np.poly1d([1, 0]),
                   det_1_energy_resolution=1, det_2_energy_resolution=1,
                   max_time_interval=10)
     return a dataframe of the coincidence cases

    df_time_channel_to_time_energy(cls, measurement_dataframe: pd.DataFrame,
                                   energy_calibration_poly=np.poly1d([1, 0]))
     change a spectrum dataframe with channels into energy units according to the calibration
    time_coincidence_check(cls, time_coincidence_pair, max_time_interval)
     check if the two times in time_coincidence_pair are less than the  max_time_interval of the detector
     (check if the measurement pair is coincidence )
    energy_coincidence_check(cls, energy_coincidence_pair, det_1_fwhm, det_2_fwhm)
     check if the two energies in energy_coincidence_pair can be coincidence (is energy1+energy2-1022 <3*joint_fwhm)
    """
    def __init__(self):
        pass

    @classmethod
    def from_files(cls, det_1_time_channel_path: str, det_2_time_channel_path: str,
                   sep=' ', skiprows=5,
                   det_1_energy_calibration_poly=np.poly1d([1, 0]), det_2_energy_calibration_poly=np.poly1d([1, 0]),
                   det_1_energy_resolution=1, det_2_energy_resolution=1,
                   max_time_interval=10, number_of_cdb_sigma=3):
        """
        Filter the files to get only the pas coincidence cases using from dataframe

        Parameters
        ----------
        det_1_time_channel_path, det_2_time_channel_path: str
         path to the file with the time and channel columns
        sep: str (default ' ')
         the seperation charecter between the time and channel in the file
        skiprows: int (default 5)
         number of lines to skip in the file
        det_1_energy_calibration_poly, det_2_energy_calibration_poly: numpy.poly1d([a, b]) (default np.poly1d([1, 0]))
        the energy calibration of the detector
        det_1_energy_resolution: float
         energy resolution of detector 1 in the annihilation peak (FWHM)
        det_2_energy_resolution: float
        energy resolution of detector 1 in the annihilation peak (FWHM)
         max_time_interval: float
            the maximal time interval of the detector to check if counts are coincidence

        Returns
        -------
        pd.DataFrame
        contain the coincidence pairs where the dataframe columns are ['energy_1, energy_2]

        """
        det_1_time_channel_df = TimeChannelParser.to_dataframe(det_1_time_channel_path, sep, skiprows)
        det_2_time_channel_df = TimeChannelParser.to_dataframe(det_2_time_channel_path, sep, skiprows)
        return cls.from_dataframe(det_1_time_channel_df, det_2_time_channel_df,
                                  det_1_energy_calibration_poly, det_2_energy_calibration_poly,
                                  det_1_energy_resolution, det_2_energy_resolution,
                                  max_time_interval, number_of_cdb_sigma)

    @classmethod
    def from_dataframe(cls, det_1_time_channel: pd.DataFrame, det_2_time_channel: pd.DataFrame,
                       det_1_energy_calibration_poly=np.poly1d([1, 0]), det_2_energy_calibration_poly=np.poly1d([1, 0]),
                       det_1_energy_resolution=1, det_2_energy_resolution=1,
                       max_time_interval=10, number_of_cdb_sigma=3):
        """
        Going through the time and energy stamps of 2 detectors data looking for coincidence pair.
        if the counts pair are valid coincidence measurement, the function saves it.

        Parameters
        ----------
        det_1_time_channel, det_2_time_channel: pd.dataframe
         The dataframe of the detector time stemps and channel reading.
        The columns names are 'time', 'channel'
        det_1_energy_calibration_poly, det_2_energy_calibration_poly: numpy.poly1d([a, b]) (default np.poly1d([1, 0]))
        the energy calibration of the detector
        det_1_energy_resolution: float
         energy resolution of detector 1 in the annihilation peak (FWHM)
        det_2_energy_resolution: float
        energy resolution of detector 1 in the annihilation peak (FWHM)
         max_time_interval: float
            the maximal time interval of the detector to check if counts are coincidence

        Returns
        -------
        pd.DataFrame
        contain the coincidence pairs where the dataframe columns are ['energy_1, energy_2]

        """
        coincidence_list = []
        det_1_time_energy = cls.df_time_channel_to_time_energy(det_1_time_channel, det_1_energy_calibration_poly)
        det_2_time_energy = cls.df_time_channel_to_time_energy(det_2_time_channel, det_2_energy_calibration_poly)
        # the index of 1 is defined in the for loop
        det_2_index = 0
        det_2_index_lim = len(det_2_time_energy) - 1
        for det_1_index, det_1_time in enumerate(det_1_time_energy['time']):
            while det_2_index < det_2_index_lim and (det_1_time >= det_2_time_energy['time'][det_2_index] or cls.time_coincidence_check([det_1_time, det_2_time_energy['time'][det_2_index]], max_time_interval)):
                if cls.time_coincidence_check([det_1_time, det_2_time_energy['time'][det_2_index]], max_time_interval):
                    coin_pair = [det_1_time_energy['energy'][det_1_index],
                                 det_2_time_energy['energy'][det_2_index]]
                    if cls.energy_coincidence_check(coin_pair, det_1_energy_resolution, det_2_energy_resolution, number_of_cdb_sigma):
                        coincidence_list.append(coin_pair)
                det_2_index = det_2_index + 1
        data = np.array(coincidence_list)
        return pd.DataFrame({'energy_1': data[:, 0], 'energy_2': data[:, 1]})

    @classmethod
    def df_time_channel_to_time_energy(cls, measurement_dataframe: pd.DataFrame,
                                       energy_calibration_poly=np.poly1d([1, 0])):
        """
        Takes time and channel and transform channel values into energy

        Parameters
        ----------
        measurement_dataframe: pd.DataFrame
         dataframe of the detector time stemps and channel reading.
        The columns names are 'time', 'channel'
        energy_calibration_poly : np.poly1d default(np.poly1d([1, 0]))
         Calibration polynomial for energy calibration.
        Returns
        -------
         pd.DataFrame
         new data frame with energy column
        """
        # Create an uncertainty dataset with 'channel' as a dimension
        measurement_dataframe['channel'] = energy_calibration_poly(measurement_dataframe['channel'])
        measurement_dataframe = measurement_dataframe.rename(columns={'channel': 'energy'})
        return measurement_dataframe

    @classmethod
    def time_coincidence_check(cls, time_coincidence_pair, max_time_interval):
        """
        Checks if the measurement pair is close enough in time to be a coincidence instance,
         meaning diff(t_1,t_2)<max_time_interval.
         Note, the photons pair detection time are theoretically equal up to the time resolution of the detectors.

        Parameters
        ----------
        time_coincidence_pair: iterable (list/tuple and such)
         time pair [t_1, t_2]
        max_time_interval : float
         The maximal time difference between two measurements to be a coincidence
        Returns
        -------
        boolean
         True if the coincidence pair is a coincidence instance and vice-versa.
            """
        return abs(time_coincidence_pair[1] - time_coincidence_pair[0]) < max_time_interval

    @classmethod
    def energy_coincidence_check(cls, energy_coincidence_pair, det_1_fwhm, det_2_fwhm, number_of_cdb_sigma=3):
        """
        Checks if the measurement pair energies sum is the same as 2*ELECTRON_REST_MASS up to the detection resolution.
        Meaning diff(sum(E_1,E_2), 2*ELECTRON_REST_MASS)<number_of_cdb_sigma*(sig_2 ** 2 + sig_1 ** 2) ** 0.5
        Note:
         1. the pair energy sum to 2*ELECTRON_REST_MASS which are theoretically equal.
         2. sigma = fwhm / (2 * np.log(2)) ** 0.5.
         3. For the filter to be comparable with the doppler broadening note that the filter here use SIGMA and not FWHM

        Parameters
        ----------
        energy_coincidence_pair: iterable(tuple/list)
         Energy pair [E_1, E_2]
        det_1_fwhm, det_2_fwhm: float
         The energy resolution of the i'th detector (FWHM)
        number_of_cdb_sigma: int default 3
         the condition is abs(energy_1 + energy_2 - 2 * ELECTRON_REST_MASS) < number_of_cdb_sigma*(sig_2 ** 2 + sig_1 ** 2) ** 0.5
        Returns
        -------
         boolean
           True if the coincidence pair is a coincidence instance and vice-versa.
            """
        energy_1 = energy_coincidence_pair[0]
        energy_2 = energy_coincidence_pair[1]
        # this calculation is expensive but is constant through all the measurement, i need to move it
        sig_1 = det_1_fwhm / (2 * np.log(2)) ** 0.5
        sig_2 = det_2_fwhm / (2 * np.log(2)) ** 0.5
        # The difference between the sum of the 2 energies from 1022 is larger than three time the resolution
        flag = abs(energy_1 + energy_2 - 2 * ELECTRON_REST_MASS) < number_of_cdb_sigma*(sig_2 ** 2 + sig_1 ** 2) ** 0.5
        return flag
