import numpy as np
import pandas as pd


ELECTRON_REST_MASS = 511


def cdb_pairs_filter_from_dataframe(det_1_data_time_energy, det_2_data_time_energy,
                                    det_1_energy_resolution, det_2_energy_resolution, max_time_interval):
    """ going through the time and energy stamps of 2 detectors data looking for coincidence pair.
    if the counts pair are valid coincidence measurement, the function saves it.
    input :
    det_1_data_time_energy, det_2_data_time_energy -
    dataFrame of all the measurements of a detector which have 2 column ->
    ('time':time of measurement, 'energy': measured energy)
    det_1_energy_resolution, det_2_energy_resolution - detector resolution
    time_interval_limit - the minimal time interval of the detector to check if counts are coincidence
    return:
    list of lists that contain the coincidence pairs
    """
    coincidence_list = []
    # the index of 1 is defined in the for loop
    det_2_index = 0
    det_2_index_lim = len(det_2_data_time_energy) - 1

    for det_1_index, det_1_time in enumerate(det_1_data_time_energy['time']):
        while det_1_time > det_2_data_time_energy['time'][det_2_index] and det_2_index < det_2_index_lim:
            if time_coincidence_check([det_1_time, det_2_data_time_energy['time'][det_2_index]], max_time_interval):
                coin_pair = [det_1_data_time_energy['energy'][det_1_index],
                             det_2_data_time_energy['energy'][det_2_index]]
                if energy_coincidence_check(coin_pair, det_1_energy_resolution, det_2_energy_resolution):
                    coincidence_list.append(coin_pair)
            det_2_index = det_2_index + 1
    data = np.array(coincidence_list)
    return pd.DataFrame({'energy_1': data[:, 0], 'energy_2': data[:, 1]})


def df_time_channel_to_time_energy(df, energy_calibration_poly):
    """
    takes time and channel stamps DataFrame and change the channel column into energy column
    return the new DataFrame
    """
    # Create an uncertainty dataset with 'channel' as a dimension
    df['channel'] = energy_calibration_poly(df['channel'])
    df = df.rename(columns={'channel': 'energy'})
    return df


def time_coincidence_check(time_coincidence_pair, max_time_interval):
    """ Return True if the measurement pair is close enough in time to be a coincidence instance and vice-versa.
        The photons pair detection time are theoretically equal up to the time resolution of the detectors.
        If time difference of the measurements is less tham max_time_interval the pair might be coincidence and
         vice-versa.
        input:
        - time_coincidence_pair(tuple/list): time pair [t_1, t_2]
        - max_time_interval (float): The maximal time difference between two measurements to be a coincidence
        return:
        flag (boolean):  True if the coincidence pair is a coincidence instance and vice-versa.
        """
    return abs(time_coincidence_pair[1] - time_coincidence_pair[0]) < max_time_interval


def energy_coincidence_check(energy_coincidence_pair, det_1_fwhm, det_2_fwhm):
    """ Return True if the coincidence pair is a coincidence instance and vice-versa.
        The pair energy sum to 2*ELECTRON_REST_MASS which are theoretically equal.
        If up to the energy sum is different from 2*ELECTRON_REST_MASS by 3 sigma, the energy pair is not coincidence.
        input:
        - energy_coincidence_pair(tuple/list): Energy pair [E_1, E_2]
        - det_i_fwhm (float): The energy resolution of the i'th detector
        return:
        flag (boolean):  True if the coincidence pair is a coincidence instance and vice-versa.
        """
    energy_1 = energy_coincidence_pair[0]
    energy_2 = energy_coincidence_pair[1]
    # this calculation is expensive but is constant through all the measurement, i need to move it
    sig_1 = det_1_fwhm / (2 * np.log(2)) ** 0.5
    sig_2 = det_2_fwhm / (2 * np.log(2)) ** 0.5
    # are the difference between the sum of the 2 energies from 1022 is larger than three time the resolution
    # from each detector center? (six sigmas in total
    flag = abs(energy_1 + energy_2 - 2 * ELECTRON_REST_MASS) < 3 * (sig_2 ** 2 + sig_1 ** 2) ** 0.5
    return flag


def cdb_pairs_filter_from_files(det_1_data_time_energy_path, det_2_data_time_energy_path,
                                det_1_energy_resolution, det_2_energy_resolution, max_time_interval):
    """ load dataframes of time and energy measurements, and then use cdb_pairs_filter_from_dataframe
    Parameters :
    - det_1_data_time_energy_path, det_2_data_time_energy_path (str) -
     files path of all the measurements of a detector which have 2 column ->
     ('time':time of measurement, 'energy': measured energy in aligned time)
    - det_1_energy_resolution, det_2_energy_resolution (float)- The detectors energy resolution
     time_interval_limit (float)- the minimal time interval of the detector to check if counts are coincidence
    Returns:
    - list of lists that contain the coincidence pairs
    """
    det_1_data_time_energy_dataframe = pd.read_csv(det_1_data_time_energy_path, sep='\t')
    det_2_data_time_energy_dataframe = pd.read_csv(det_2_data_time_energy_path, sep='\t')
    return cdb_pairs_filter_from_dataframe(det_1_data_time_energy_dataframe, det_2_data_time_energy_dataframe,
                                           det_1_energy_resolution, det_2_energy_resolution, max_time_interval)