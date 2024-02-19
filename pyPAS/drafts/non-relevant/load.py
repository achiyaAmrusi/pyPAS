import pandas as pd
import numpy as np
from pyspectrum.spectrum import Spectrum
from pas_measurement_objects.pydb import PASdb
from pas_measurement_objects.pycdb import PAScdb


def db_from_file(file_path, energy_calibration_poly=np.poly1d([1, 0])):
    """
    load spectrum from a file which has 2 columns which tab between them
    first column is the channels/energy and the second is counts
    function return Spectrum
    input :
    spectrum file - two columns with tab(\t) between them.
     first line is column names - channel, counts
     energy_calibration - numpy.poly1d([a, b])
    """
    # Load the pyspectrum file in form of DataFrame
    try:
        data = pd.read_csv(file_path, sep='\t')
    except ValueError:
        raise FileNotFoundError(f"The given data file path '{file_path}' do not exist.")
    return PASdb(data[data.columns[1]].to_numpy(), data[data.columns[0]].to_numpy(), energy_calibration_poly)


def cdb_from_file(file_path):
    """
    load spectrum from a file which has 2 columns with \t between them.
    each line in the file represents coincidence measurement.
    The first column is the energy measured in the first detector and the energy in the second column is the
    energy measured in the second detector.
    input :
        file path - the path to the coincidence measurements list
    return:
        CDBpas of the measurement
    """
    try:
        data = pd.read_csv(file_path, sep='\t')
    except ValueError:
        raise FileNotFoundError(f"The given data file path '{file_path}' do not exist.")
    return PAScdb(data)
