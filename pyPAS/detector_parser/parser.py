import pandas as pd
import numpy as np
from pyspectrum import Spectrum


class TimeChannelParser:
    """
    class include parsing functions for measurements data which have the form time_stamp, channel
    This is relevant for coincidence measurements
    Attributes
    ----------

    Methods
    -------
    to_dataframe(time_channel_path: str, sep=' ', skiprows=5)
     Filter the time_channel file from pileup and negative counts
    to_spectrum(time_channel_path: str,
                    energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None,
                    sep=' ',  skiprows=5,
                    num_of_channels=2 ** 14)
     Filter the time_channel file from pileup and negative counts,
    then transform data to spectrum using the calibrations
    counts_in_time_into_spectrum(cls, time_channel_df: pd.DataFrame, num_of_channels=2 ** 14)
     Takes a data frame with time stemp, count and pileup and turn into spectrum in dataframe form
    """
    def __init__(self):
        pass

    @staticmethod
    def to_dataframe(time_channel_path: str, sep=' ', skiprows=5):
        """
        Filter the time_channel file from pileup and negative counts
        Parameters
        ----------
        time_channel_path: str
         path to the file with the time and channel columns
        sep: str (default ' ')
        the seperation charecter between the time and channel in the file
        skiprows: int (default 5)
        number of lines to skip in the file
        Returns
        -------
        pd.DataFrame
         filtered dataframe of the time channel file
        """
        # 3 data bins for each row
        # 3 data bins for each row
        column_names = ['time', 'channel', 'pileUp']
        # Read the data from the input file into a DataFrame without the first lines
        try:
            data = pd.read_csv(time_channel_path,
                               skiprows=skiprows, sep=sep, names=column_names, usecols=range(3))
        except FileNotFoundError:
            raise FileNotFoundError(f"The given data file path '{time_channel_path}' does not exist.")

        filtered_data = data[(data.iloc[:, 2] == 0) & (data.iloc[:, 1] >= 0)]
        filtered_data.reset_index(inplace=True)
        return filtered_data[['time', 'channel']]

    @staticmethod
    def to_spectrum(time_channel_path: str,
                    energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None,
                    sep=' ',  skiprows=5,
                    num_of_channels=2 ** 14):
        """
        Filter the time_channel file from pileup and negative counts,
        then transform data to spectrum using the calibrations
        Parameters
        ----------
        time_channel_path: str
         path to the file with the time and channel columns
        energy_calibration_poly: numpy.poly1d([a, b])
         the energy calibration of the detector
        fwhm_calibration: Callable
        a function that given energy/channel(first raw in file) returns the fwhm
        sep: str (default ' ')
        the separation character between the time and channel in the file
        skiprows: int
        number of lines to skip in the file
        num_of_channels: int
        the number of channels in the measurement
        Returns
        -------
        pd.DataFrame
         filtered dataframe of the time channel file
        """
        time_channel_df = TimeChannelParser.to_dataframe(time_channel_path, sep=sep,  skiprows=skiprows)
        spectrum_df = TimeChannelParser.counts_in_time_into_spectrum(time_channel_df, num_of_channels)

        return Spectrum.from_dataframe(spectrum_df, energy_calibration_poly, fwhm_calibration)

    @classmethod
    def counts_in_time_into_spectrum(cls, time_channel_df: pd.DataFrame, num_of_channels=2 ** 14):
        """
        Takes a data frame with time stemp, count and pileup and turn into spectrum in dataframe form
        Parameters
        ----------
        time_channel_df: str
         dataframe of the time stemps and reading channels
        num_of_channels: int default = 2e14
        The number of chnnels in the detector
        Returns
        -------
        pd.DataFrame
         filtered dataframe of the time channel file
        """
        # construct spectrum
        spectrum_counts = time_channel_df['channel'].value_counts()
        partial_spectrum = pd.DataFrame({'channel': spectrum_counts.index, 'counts': spectrum_counts.values})
        partial_spectrum = partial_spectrum.set_index('channel')
        # take the last bin to be 0 (bug)
        partial_spectrum.loc[num_of_channels - 1] = 0
        # fills the spectrum with all the channels that didn't got counts
        full_spectrum = pd.DataFrame({'counts': np.zeros(num_of_channels - 1)}, index=list(range(1, 2 ** 14)))
        full_spectrum.index.name = 'channel'
        full_spectrum.update(partial_spectrum)
        return full_spectrum.reset_index()


