import os
from uncertainties import unumpy
import pandas as pd
from data_types import id_file
from Panalysis import database_tools
from pyspectrum import load_spectrum

ELECTRON_REST_MASS = 511


def db_create_s_w_file(db_directory_path, s_w_file_path, energy_domain_peak, energy_domain_s,
                       energy_domain_w_l, energy_domain_w_r):
    """create s and w parameter file in DB database for each measurement in pyspectrum file
    also update id file to include s_w"""

    # is the directory exist and 'DB' type
    try:
        database_type = id_file.is_in_id_file(db_directory_path, 'data_base_type')
    except ValueError:
        raise FileNotFoundError(f"The given data directory path '{db_directory_path}' not in format .")
    if not (database_type == "DB"):
        print(database_type)
        print("wrong database type")
        return 1
    # tries to open the data directory
    spectra_directory = db_directory_path + '/db_spectra_files'
    try:
        file_names = os.listdir(spectra_directory)
    except ValueError:
        raise FileNotFoundError(f"The given data directory path '{db_directory_path}' not in the .")
    s_w_data = pd.DataFrame({'energy': [], 's': [], 's_error': [], 'w': [], 'w_error': []})
    # extract s and w for each file and add that to s_w file
    for file_name in file_names:
        # file name and energy
        file_energy = int(file_name)
        file_path = spectra_directory + '/' + file_name

        # load the pyspectrum from the file
        spectrum = load_spectrum.load_spectrum_file_to_xarray_spectrum(file_path)

        # calculate the s and w parameters from the pyspectrum
        (s_parm, w_parm) = database_tools.spectrum_s_w_calculation(spectrum, energy_domain_peak, energy_domain_s,
                                                                   energy_domain_w_l, energy_domain_w_r)
        s_w_data = pd.concat([s_w_data, pd.DataFrame({'energy': [file_energy],
                                                      's': [unumpy.nominal_values(s_parm)],
                                                      's_error': [unumpy.std_devs(s_parm)],
                                                      'w': [unumpy.nominal_values(w_parm)],
                                                      'w_error': [unumpy.std_devs(w_parm)]})],
                             ignore_index=True)
    s_w_data.to_csv(s_w_file_path, sep='\t', index=False)
    return 0
