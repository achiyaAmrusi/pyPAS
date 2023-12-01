import numpy as np
import pandas as pd
import ast

# File include function which are common on the database type.
# e.g. DB and CDB need energy calibration function, id file and so on.


ELECTRON_REST_MASS = 511


def add_energy_calibration(database_path, poly):
    """add an energy calibration to a DB and CDB database """
    try:
        idfile = pd.read_csv(database_path + '/id', header=None, sep='\t', index_col=0)
    except FileExistsError:
        print(f"wrong format - id file doesn't exist")
        return 1
    idfile.loc["energy_calibration"] = [list(poly)]
    idfile.columns = ['']
    idfile.index.name = ''
    idfile.to_csv(database_path + '/id', sep='\t', index=True, header=False)
    return 0


def energy_calibration(database_path):
    """return the energy calibration to a DB and CDB database
    input:
    database directory"""
    try:
        idfile = pd.read_csv(database_path + '/id', header=None, sep='\t', index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"wrong format directory - id file is not found")
    try:
        idfile.loc["energy_calibration"]
    except KeyError:
        raise KeyError(f"No energy calibration")
    return np.poly1d(ast.literal_eval(idfile.loc["energy_calibration"].values[0]))


def is_in_id_file(database_path, feature_name, raise_error=1):
    """ check if the feature exist in id file
     if it does it return the value, otherwise None"""
    try:
        idfile = pd.read_csv(database_path + '/id', header=None, sep='\t', index_col=0)
    except FileExistsError:
        print(f"wrong format - id file doesn't exist")
        return None
    try:
        idfile.loc[feature_name]
    except KeyError:
        if raise_error:
            print(f"{feature_name} is not in id file or there is another error")
        return None
    return idfile.loc[feature_name].values[0]


def add_to_id_file(database_path, feature_name, feature_value):
    """ add feature into id file  """
    idfile = pd.read_csv(database_path + '/id', header=None, sep='\t', index_col=0)
    idfile.loc[feature_name] = feature_value
    idfile.columns = ['']
    idfile.index.name = ''
    idfile.to_csv(database_path + '/id', sep='\t', index=True, header=False)
    return 0


def remove_from_id_file(database_path, feature_name):
    """ add feature into id file  """
    idfile = pd.read_csv(database_path + '/id', header=None, sep='\t', index_col=0)
    try:
        idfile = idfile.drop(feature_name)
    except KeyError:
        raise KeyError(f"The feature in the id is not exist")
    idfile.columns = ['']
    idfile.index.name = ''
    idfile.to_csv(database_path + '/id', sep='\t', index=True, header=False)
    return 0
