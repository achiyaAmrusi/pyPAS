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
# spectrum_files - positrons energy and pyspectrum
#
import os
import numpy as np
import pandas as pd

from Panalysis import database_tools
from huji_to_format import huji_to_format_tools


ELECTRON_REST_MASS = 511


def db_make_data_dir_format(data_dir_path, db_data_in_format_path, sample_name, detector_id):
    """make data directory which is created according to the format.
        The format contains the following -
        id file: data type (DB,CDB,PALS) , sample name, if there is energy calibration
        spectrum_files: positron energy with the pyspectrum measured in the detector
        s_w : s and w parameters
        input :
        data_dir_path - the data directory in the default format of huji results
        with results of 1 detector only!
        db_data_in_format_path - the data directory in which the data directory is saved
        sample_name - name of the sample
        detector_id - detector number
        """
    # define new data in format directory
    format_dir = db_data_in_format_path + '\\' + sample_name
    # tries to open the data directory
    try:
        file_names = os.listdir(data_dir_path)
    except ValueError:
        raise ValueError(f"The given data directory path '{data_dir_path}' does not exist.")
    # tries to make the in format data directory
    # if it can, it will also prepare spectra directory and id file
    try:
        os.mkdir(format_dir)
    except FileExistsError:
        print(f"Directory '{format_dir}' already exists.")
        return 1
    except Exception as e:
        print(f"An error occurred during the creation of the directory: {e}")
    huji_to_format_tools.make_format_id_file(format_dir, sample_name, 'DB_FORMAT', detector_id)
    spectra_dir = format_dir + '\\db_spectra_files'
    os.mkdir(spectra_dir)
    for file_name in file_names:
        db_save_file_in_format(data_dir_path, spectra_dir, sample_name, file_name)


def db_filter_data_file_to_spectrum(input_file, num_lines_to_remove=5):
    """filter data from pile up and non-relevant lines in the beginning
        returns simple pyspectrum of a detector
    """
    # 3 data bins for each row
    # The following commands takes the data from the file and form it into pyspectrum.
    # 1. Filter the (time, counts ,pile up) DataFrame to keep only rows where the third column is 0
    # 2. (time, counts ,pile up) -> pyspectrum with only the channels that had any count
    # 3. fix the error of large count in the last channel
    # 4. change the pyspectrum to have all channels
    # step 1 happens in function filter_data_file and the rest in counts_in_time_into_spectrum
    filtered_data = huji_to_format_tools.data_file_to_filtered_dataframe(input_file, num_lines_to_remove=num_lines_to_remove)
    # 2
    full_spectrum = huji_to_format_tools.counts_in_time_into_spectrum(filtered_data)

    return full_spectrum


def db_save_file_in_format(data_dir_path, spectra_dir, sample_name, file_name):
    """ takes the data from the directory which obey the format of huji.
        Then, the function save it in the format in data_base directory
        """
    # the format of the files name is Xev_channel00i where X is the energy in ev and 'i' is 0 or 1.
    ev_index = file_name.find('eV')
    # Extract the substring before 'ev' if it's found and writing the pyspectrum
    if ev_index != -1:
        substring_before_ev = file_name[:ev_index]
        db_data = db_filter_data_file_to_spectrum(data_dir_path + '/' + file_name)
        spect_text = (db_data.reset_index()).to_csv(spectra_dir + '/' + 'energy_' + substring_before_ev,
                                                    index=False, sep='\t')
        print(f'Measurement with positron energy of {substring_before_ev}[eV] is loaded')
    else:
        print('the file name is not in the format XeV_channel00i')

