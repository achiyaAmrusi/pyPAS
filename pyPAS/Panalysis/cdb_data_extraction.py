import os
import pandas as pd
from pyspectrum import load_spectrum
from pas_objects import id_file
from Panalysis import database_tools

ELECTRON_REST_MASS = 511


def cdb_create_s_w_file(cdb_directory_path, s_w_file_path, energy_domain_peak, energy_domain_s,
                        energy_domain_w_l, energy_domain_w_r):
    """create 2d spectra of cdb, resolution pyspectrum and db file in CDB database for each measurement in pyspectrum file.
    Also, update id file to include 2d_spectra and 2D CDB
    input :
    - data_directory : directory of cdb data
    - time_interval_limit : the time such that if two counts counted in this time differential
                            they practically where counted in the same time.
                            this means if |time_of_count_det1-time_of_count_det2|<time_interval_limit
                            the counts where in the same time.
                            for huji the cdb time interval is 10 nanosecond
    """
    # is the directory exist and 'CDB' type
    try:
        database_type = id_file.is_in_id_file(cdb_directory_path, 'data_base_type')
    except ValueError:
        raise FileNotFoundError(f"The given data directory path '{cdb_directory_path}' not in format .")
    if not (database_type == "CDB"):
        print("wrong database type")
        return 1

    # open data directory - filled with energy directories, in each one, cdb_histogram, db and resolution
    data_files = cdb_directory_path + '/cdb_spectra_files'
    try:
        # energy directories
        energies_list = os.listdir(data_files)
    except ValueError:
        raise ValueError(f"The given data directory path '{data_files}' not in format .")
    cdb_files_path = [data_files + '/' + energy + '/db/db_spectrum' for energy in energies_list]
    s_w_data = pd.DataFrame({'energy': [], 's': [], 'w': []})
    # for on the db files in the cdb db path
    for index, cdb_file_name in enumerate(cdb_files_path):
        # file name and energy
        file_energy = int(energies_list[index])
        # load the pyspectrum from the file
        spectrum = load_spectrum.load_spectrum_file_to_xarray_spectrum(cdb_file_name)

        # calculate the s and w parameters from the pyspectrum
        (s_parm, w_parm) = database_tools.spectrum_s_w_calculation(spectrum, energy_domain_peak, energy_domain_s,
                                                                   energy_domain_w_l, energy_domain_w_r)
        s_w_data = pd.concat([s_w_data, pd.DataFrame({'energy': [file_energy], 's': [s_parm], 'w': [w_parm]})],
                             ignore_index=True)
    s_w_data.to_csv(s_w_file_path)
    id_file.add_to_id_file(cdb_directory_path, 's_w', True)
    return 0


def load_data_from_file_energy(path, energy_calibration_poly):
    """
    load data of spectrometer into pandas data frame
    input:
    data file path in format - Time, Channel
    energy calibration polynomial - np.polyd1 such that the polynom is the calibration
    """
    data = pd.read_table(path, sep='\t')
    data_detector_spec = data.rename(columns={'channel': 'energy'})
    data_detector_spec['energy'] = energy_calibration_poly(data_detector_spec['energy'])
    return data_detector_spec
