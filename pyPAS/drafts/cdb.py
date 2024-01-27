import os

import numpy as np
import pandas as pd
import xarray as xr
import id_file
import db

ELECTRON_REST_MASS = 511


def cdb_create_coincidence_spectra(format_data_path, cdb_datatype_directory_path, name, energy_calib_poly_det_1,
                                   energy_calib_poly_det_2,
                                   energy_resolution_det_1, energy_resolution_det_2,
                                   grid_dim_size=10, two_dim_mesh_parm=0.1, time_interval_limit=10):
    """create 2d spectra of cdb, resolution pyspectrum and db file in CDB database for each measurement in pyspectrum file.
    Also, update id file to include 2d_spectra and 2D CDB
    input :
    - format_data_path : directory of cdb data in program format
    - cdb_datatype_directory_path : the new path for the directory to save
    - name : description of the measurement (suppose to be str)
    - energy calibration : np.poly1d:channel -> energy
    - time_interval_limit : the time such that if two counts counted in this time differential
                            they practically where counted in the same time.
                            this means if |time_of_count_det1-time_of_count_det2|<time_interval_limit
                            the counts where in the same time.
                            for huji the cdb time interval is 10 nanosecond
    - grid_dim_size : the limits of the cdb histogram. how much domain to consider in cdb(-E,E)
    - two_dim_mesh_parm : the mesh of the 2d cdb histogram
    """
    # is the directory exist and 'CDB_FORMAT' type
    try:
        database_type = id_file.is_in_id_file(format_data_path, 'data_base_type')
    except ValueError:
        raise FileNotFoundError(f"The given data directory path '{format_data_path}' not in format .")
    if not (database_type == "CDB_FORMAT"):
        print("wrong database type")
        return 1
    # open data directory - filled with energy directories, in each two measurements (one for each detector)
    data_files = format_data_path + '/cdb_data_files'
    try:
        # energy directories
        energies_list = os.listdir(data_files)
    except ValueError:
        raise ValueError(f"The given data directory path '{data_files}' not in format .")

    cdb_type_directory = cdb_datatype_directory_path
    # If the directory name already exist
    if not os.path.exists(cdb_type_directory):
        os.mkdir(cdb_type_directory)
    else:
        print("directory name is already taken, try somthing else")
        return 2
    # make spectra directory
    cdb_spectra_dir = cdb_type_directory + '/cdb_spectra'
    os.mkdir(cdb_spectra_dir)
    # create cdb id file
    cdb_write_id_file(cdb_type_directory, name, energy_calib_poly_det_1, energy_calib_poly_det_2,
                      energy_resolution_det_1, energy_resolution_det_2,
                      two_dim_mesh_parm, time_interval_limit)
    # create spectra files
    for energy in energies_list:
        dir_energy_cdb_data = data_files + '/' + energy
        # load the detectors time-channel stamps dataFrame
        det_1_df_channels = pd.read_table(dir_energy_cdb_data + '/detector_1_data', sep='\t')
        det_2_df_channels = pd.read_table(dir_energy_cdb_data + '/detector_2_data', sep='\t')
        # calibrate the channels of the detectors time-channel stamps data  into time-energy stamps
        det_1_df = df_time_channel_to_time_energy(det_1_df_channels, energy_calib_poly_det_1)
        det_2_df = df_time_channel_to_time_energy(det_2_df_channels, energy_calib_poly_det_2)
        # get the list of all the valid cdb instances
        cdb_pairs = cdb_pairs_parser(det_1_df, det_2_df,
                                     energy_resolution_det_1, energy_resolution_det_2,
                                     time_interval_limit)
        cdb_2d_spectrum_hist, first_det_edges, second_det_edges = cdb_pairs_df_to_hist(cdb_pairs, grid_dim_size,
                                                                                       two_dim_mesh_parm)
        doppler_broadening = cdb_horizontal_cut(cdb_2d_spectrum_hist, first_det_edges)
        resolution_profile = cdb_vertical_cut(cdb_2d_spectrum_hist, second_det_edges)
        dir_energy_cdb = cdb_spectra_dir + '/' + str(energy)
        os.mkdir(dir_energy_cdb)
        os.mkdir(dir_energy_cdb+'/histogram2d')
        os.mkdir(dir_energy_cdb + '/db')
        os.mkdir(dir_energy_cdb + '/resolution')
        save_cdb_spectrum(dir_energy_cdb+'/histogram2d', cdb_2d_spectrum_hist, first_det_edges, second_det_edges)
        db.db_save_spectrum_file(dir_energy_cdb + '/db', 'db', doppler_broadening)
        db.db_save_spectrum_file(dir_energy_cdb + '/resolution', 'res', resolution_profile)
        print(f'finished energy {energy}')
    return 0


def df_time_channel_to_time_energy(df, energy_calibration_poly):
    """
    takes time and channel stamps DataFrame and change the channel column into energy column
    return the new DataFrame
    """
    # Create an uncertainty dataset with 'channel' as a dimension
    df['channel'] = energy_calibration_poly(df['channel'])
    df = df.rename(columns={'channel': 'energy'})
    return df


def cdb_pairs_df_to_hist(cdb_cases, grid_dim_size, mesh_interval):
    """ return the cdb pyspectrum from all the cdb pairs in dataframe format
        input:
        - cdb_cases : dataframe in the format of index, energy_1, energy_2
        where in each index the energy_1 and energy_2 instance are a measurement pair
        - mesh_interval : the interval (###later we need to change it to a full mesh )

        return:
         2d histogram, edges of the x axis, edges of y axis """
    # create the mesh
    # Adjust the range and number of bins as needed
    bin_edges_x = np.arange(-grid_dim_size, grid_dim_size, mesh_interval)
    bin_edges_y = np.arange( -grid_dim_size, grid_dim_size, mesh_interval)

    # Create 2D histogram
    db = (cdb_cases['energy_1'] - cdb_cases['energy_2'])/2
    res = (cdb_cases['energy_1'] + cdb_cases['energy_2']) - 2*ELECTRON_REST_MASS
    hist, x_edges, y_edges = np.histogram2d(res, db, bins=[bin_edges_x, bin_edges_y])

    return hist, x_edges, y_edges


def cdb_pairs_parser(det_1_data_time_energy, det_2_data_time_energy,
                     det_1_energy_resolution, det_2_energy_resolution, time_interval_limit):
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
    index = 0
    det_1_index = 0
    det_2_index = 0
    det_2_index_lim = len(det_2_data_time_energy) - 1
    det_1_time = 0
    det_2_time = 0

    for det_1_index, det_1_time in enumerate(det_1_data_time_energy['time']):
        while det_1_time > det_2_data_time_energy['time'][det_2_index] and det_2_index < det_2_index_lim:
            if abs(det_1_time - det_2_data_time_energy['time'][det_2_index]) < time_interval_limit:
                coin_pair = [det_1_data_time_energy['energy'][det_1_index],
                             det_2_data_time_energy['energy'][det_2_index]]
                if energy_coincidence_check(coin_pair, det_1_energy_resolution, det_2_energy_resolution):
                    coincidence_list.append(coin_pair)
            det_2_index = det_2_index + 1
    data = np.array(coincidence_list)
    return pd.DataFrame({'energy_1': data[:, 0], 'energy_2': data[:, 1]})


def energy_coincidence_check(coincidence_pair, det_1_fwhm, det_2_fwhm):
    """ checks if the coincidence pair is indeed a coincidence instance
        this is done by comparing the pair energy sum to 2*ELECTRON_REST_MASS which suppose to be equal
        if up to 3 sigma the energy sum there is difference in the energies than the count is not cdb
        input:
        - coincidence_pair: [E_1, E_2]
        - det_i_fwhm: the resolution of the i'th detector
        return:
        boolean
        """
    energy_1 = coincidence_pair[0]
    energy_2 = coincidence_pair[1]
    # this calculation is expensive but is constant through all the measurement, i need to move it
    sig_1 = det_1_fwhm / (2 * np.log(2)) ** 0.5
    sig_2 = det_2_fwhm / (2 * np.log(2)) ** 0.5
    # are the difference between the sum of the 2 energies from 1022 is larger than three time the resolution
    # from each detector center? (six sigmas in total
    flag = abs(energy_1 + energy_2 - 2*ELECTRON_REST_MASS) < 3 * (sig_2 ** 2 + sig_1 ** 2) ** 0.5
    return flag


def save_cdb_spectrum(path, histogram_2d, first_detector_bins, second_detector_bins):
    """saves the 3 elements from cdb pyspectrum -
    histogram, doppler broadening profile, resolution profile"""
    histogram_2d.tofile(path + '/histogram', sep='\t', format='%s')
    first_detector_bins.tofile(path + '/first_detector_energy_bins', sep='\t', format='%s')
    second_detector_bins.tofile(path + '/second_detector_energy_bins', sep='\t', format='%s')
    return 0


def cdb_write_id_file(cdb_type_directory, name, energy_calib_poly_det_1, energy_calib_poly_det_2,
                      energy_resolution_det_1, energy_resolution_det_2,
                      two_dim_mesh_parm, time_interval_limit):
    id_df = pd.DataFrame(data=['CDB', name, energy_resolution_det_1, energy_resolution_det_2,
                               [list(energy_calib_poly_det_1)], [list(energy_calib_poly_det_2)],
                               two_dim_mesh_parm, time_interval_limit],
                         index=['data_base_type', 'name', 'energy_resolution_det_1', 'energy_resolution_det_2',
                                'energy_calibration_det_1', 'energy_calibration_det_2',
                                'mesh_energy_interval_kev', 'time_interval_for_coincidence'])
    id_df.to_csv(cdb_type_directory + '/id', sep='\t', index=True, header=False)
    return 0


def cdb_horizontal_cut(cdb_2d_spectrum_hist, det_energy_hist_edges):
    """calculate the horizontal cut of the 2d cdb histogram
    return:
    doppler broadening
    """
    db = cdb_2d_spectrum_hist[0, :] * 0
    for i in range(len(cdb_2d_spectrum_hist[:, 0])):
        db = db + cdb_2d_spectrum_hist[i, :]
    domain = [(det_energy_hist_edges[i + 1] + det_energy_hist_edges[i]) / 2
              for i in range(len(det_energy_hist_edges) - 1)]
    return xr.DataArray(db, {'energy': domain}, dims=['energy'], name='counts')


def cdb_vertical_cut(cdb_2d_spectrum_hist, det_energy_hist_edges):
    """calculate the vertical cut of the 2d cdb histogram
    return:
    resolution pyspectrum
    """
    res = cdb_2d_spectrum_hist[:, 0] * 0
    for i in range(len(cdb_2d_spectrum_hist[0, :])):
        res = res + cdb_2d_spectrum_hist[:, i]
    domain = [(det_energy_hist_edges[i + 1] + det_energy_hist_edges[i]) / 2
              for i in range(len(det_energy_hist_edges) - 1)]
    return xr.DataArray(res, {'energy': domain}, dims=['energy'], name='counts')
