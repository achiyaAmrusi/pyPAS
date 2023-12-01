# functions to adjust input files into the relevant file
# each lab and equipment different and thus  will have their own format.
# Thus, a function to adjust the forma is very individualistic function.
# e.g. in our system, the data output comes with 5 lines before data and third column for pileup, and we clean it up
# the format to be use is the following:
# id file - data_base type
# basic_results
# advance_results
# the basic results file changes for each measurement DB\CDB\PALS
# the advanced results is dependent on the analysis function used.
# id file format :
# name sample_name
# data_base_type DB/CDB/PALS
# basic_results :
# dir_name DB/CDB
# for doppler broadening
# spectrum_files - positrons energy and spectrum
#
import os
import pandas as pd
from uncertainties import unumpy
from data_types import id_file
from spectrum import s_tools, load_spectrum

ELECTRON_REST_MASS = 511


def db_make_data_dir(format_data_path, db_directory_path, name, energy_calibration_poly, energy_resolution,
                     detector_id):
    """make data directory which is created according to the format.
        The format contains the following -
        id file: data type (DB_f,CDB_f,PALS_f) , sample name, if there is energy calibration
        spectrum_files: positron energy with the spectrum measured in the detector
        input :
        data_dir_path - the data directory in the default format of huji results
        with results of 1 detector only!
        db_directory_path - the resulted dir path which contain the measurement data given resolution and calibration,
        the directory is the db type data which can contain the s and w data and support the functions Panalysis
        name - sample name in id file
        detector_name - detector number
        """
    # tries to open the data directory and get the spectra file names
    if not check_if_db_dir_in_format(format_data_path):
        print("directory is not in format")
        return 1
    spectra_data_directory = format_data_path + '/db_spectra_files'
    files_name = os.listdir(spectra_data_directory)

    db_directory = db_directory_path
    spectra_dir = db_directory + '/db_spectra'
    os.mkdir(db_directory)
    os.mkdir(spectra_dir)
    db_write_id_file(db_directory, name, detector_id, energy_resolution, energy_calibration_poly)

    for file_name in files_name:
        energy = int(file_name[file_name.find('_') + 1:])

        spectrum = load_spectrum.load_spectrum_file_to_xarray_spectrum(spectra_data_directory + '/' + file_name,
                                                                       energy_calibration_poly)
        spectrum_no_bg = s_tools.subtract_background_from_spectra_peak(spectrum,
                                                                       energy_resolution, ELECTRON_REST_MASS)
        db_save_spectrum_file(spectra_dir, file_name[file_name.find('_') + 1:], spectrum_no_bg)
        print(f'finished energy {energy}')

    return 0


def check_if_db_dir_in_format(format_data_path):
    """check if the path is to  doppler broadening format directory
      the function checks the id file, the existence of the directory and the spectra directory """
    if not os.path.exists(format_data_path):
        print(f"directory {format_data_path} do not exist")
        return False
    id_type = id_file.is_in_id_file(format_data_path, 'data_base_type', 0)
    if not (id_type == 'DB_FORMAT'):
        print(f"id file shows DB_FORMAT")
        return False
    if not os.path.exists(format_data_path + '/db_spectra_files'):
        print(f"spectra directory {format_data_path} do not exist")
        return False
    return True


def db_write_id_file(db_directory_path, name, energy_calibration_poly, energy_resolution, detector_id=None):
    if not (detector_id is None):
        id_df = pd.DataFrame(data=['DB', name, detector_id, energy_resolution, [list(energy_calibration_poly)]],
                             index=['data_base_type', 'name', 'detector_id', 'energy_resolution', 'energy_calibration'])
    else:
        id_df = pd.DataFrame(data=['DB', name, energy_resolution, [list(energy_calibration_poly)]],
                             index=['data_base_type', 'name', 'energy_resolution', 'energy_calibration'])
    id_df.to_csv(db_directory_path + '/id', sep='\t', index=True, header=False)
    return 0


def db_save_spectrum_file(spectrum_directory, file_name, spectrum_no_bg):
    spectrum_no_bg_df = spectrum_no_bg.to_dataframe()
    spectrum_no_bg_df['counts'] = unumpy.nominal_values(spectrum_no_bg_df['counts'])
    spectrum_no_bg_df.to_csv(spectrum_directory + '/' + file_name, sep='\t')
    return 0
