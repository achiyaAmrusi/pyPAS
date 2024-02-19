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
# cdb data files are large

import os
from pyPAS.detector_parser import huji_to_format_tools


ELECTRON_REST_MASS = 511


def cdb_make_data_dir_format(cdb_data_path:str, cdb_data_in_format_path, sample_name:'str'):
    """make data directory which is created according to the format.
        The format contains the following -
        id file: data type (DB,CDB,PALS) , sample name, if there is energy calibration
        spectrum_files: cdb 2d pyspectrum files
        huji data files are using the template name is "****ev_ch000* where the last figure is 1 or 0
        the cdb function check if the files are common for two

        """
    # tries to open the given data directory
    try:
        file_names = os.listdir(cdb_data_path)
    except ValueError:
        raise ValueError(f"The given data directory path '{cdb_data_path}' does not exist.")
    # energies measured
    energy_list = get_measurement_energy_list(file_names)
    # define the new data in directory format
    format_dir = cdb_data_in_format_path + '\\' + sample_name
    try:
        os.mkdir(format_dir)
    except FileExistsError:
        print(f"Directory '{format_dir}' already exists.")
        return 1
    except Exception as e:
        print(f"An error occurred during the creation of the directory: {e}")
    # id file
    huji_to_format_tools.make_format_id_file(format_dir, sample_name, 'CDB_FORMAT')
    # spectra directory for cdb
    spectra_dir = format_dir + '\\cdb_data_files'
    os.mkdir(spectra_dir)
    # makes the spectra directory and files for each energy inside the spectra directory
    for measured_energy in energy_list:
        # detector data file names according to huji convention
        file_name_1_huji = cdb_data_path + '/' + measured_energy + "ev_ch000.txt"
        file_name_2_huji = cdb_data_path + '/' + measured_energy + "ev_ch001.txt"
        # save the coincidence data for the measured energy
        if os.path.isfile(file_name_1_huji) and os.path.isfile(file_name_2_huji):
            data_detector_1 = huji_to_format_tools.data_file_to_filtered_dataframe(file_name_1_huji)
            data_detector_2 = huji_to_format_tools.data_file_to_filtered_dataframe(file_name_2_huji)

            spectre_dir_energy = spectra_dir + '/' + measured_energy
            os.mkdir(spectre_dir_energy)
            data_detector_1[['time', 'channel']].to_csv(spectre_dir_energy + '/detector_1_data', sep='\t', index=False)
            data_detector_2[['time', 'channel']].to_csv(spectre_dir_energy + '/detector_2_data', sep='\t', index=False)
            print(f"energy {measured_energy} is loaded")
        else:
            print(f"energy {measured_energy} dose not exist for both detectors")
    return 0


def get_measurement_energy_list(huji_files_names):
    """function gets the energies of all the elements in the directory with form
    xxxxeV_ch000i"""
    energy_list = [filename.split('eV')[0] for filename in huji_files_names if 'eV_ch' in filename]
    # the energies appear twice for each detector thus we use set to filter the repetitive elements
    energy_list = list(energy_list)
    return energy_list




