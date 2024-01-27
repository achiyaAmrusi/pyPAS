import numpy as np
import pandas as pd
from pyPAS.pyPdb import PASdb, PAScdb

ELECTRON_REST_MASS = 511


class BeamDbSet:
    """
    This object holds a dataframe of energies and doppler broadening spectrum measured in beam
    Using the object the energy dependency of PAS parameters can be extracted.
    Parameters:
    - db_list: list of PASdb
    - energy_list:  !aligned! list of doppler broadening energy

    """

    def __init__(self, db_list, energy_list):
        """ Constructor of BeamDbSet measurement"""
        self.defect_parameters = pd.DataFrame({'energy': [], 'S': [], 'W': []})
        self.beam_db = pd.DataFrame({'energy': energy_list, 'PASdb': db_list})
        self.beam_db = self.beam_db.set_index('energy')

    def calculate_s_parameter(self, energy_list, energy_domain_total, energy_domain_s,
                              peak_energy_resolution=1, background_subtraction=True):
        s_parm_list = []
        new_energy_list = []
        beam_energy_list_set = set(self.beam_db.index)
        for ind, energy in enumerate(energy_list):
            if energy in beam_energy_list_set:
                if any(self.defect_parameters['energy'] == energy):
                    w_parm = self.defect_parameters.loc[self.defect_parameters['energy'] == energy]['W']
                else:
                    w_parm = None
                s_parm = self.beam_db.loc[energy]['PASdb'].s_parameter_calculation(peak_energy_resolution,
                                                                                   energy_domain_total,
                                                                                   energy_domain_s,
                                                                                   background_subtraction).values
                s_parm_list.append(s_parm)
                new_energy_list.append(energy)
                if any(self.defect_parameters['energy'] == energy):
                    self.defect_parameters.iloc[self.defect_parameters['energy'] == energy] = pd.Series({
                        'energy': energy_list[ind], 'S': s_parm, 'W': w_parm})
                else:
                    self.defect_parameters = pd.concat([self.defect_parameters,
                                                        pd.DataFrame({'energy': [energy_list[ind]],
                                                                      'S': [s_parm], 'W': [w_parm]})],
                                                       ignore_index=True)
            print(f'S {energy} finised')
        return pd.DataFrame({'energy': new_energy_list, 'S': s_parm_list}).set_index('energy')

    def calculate_w_parameter(self, energy_list, energy_domain_total, energy_domain_w_left, energy_domain_w_right,
                              peak_energy_resolution=1, background_subtraction=True):
        w_parm_list = []
        new_energy_list = []
        beam_energy_list_set = set(self.beam_db.index)
        for ind, energy in enumerate(energy_list):
            if energy in beam_energy_list_set:
                if any(self.defect_parameters['energy'] == energy):
                    s_parm = self.defect_parameters.loc[self.defect_parameters['energy'] == energy]['S']
                else:
                    s_parm = None
                w_parm = self.beam_db.loc[energy]['PASdb'].w_parameter_calculation(peak_energy_resolution,
                                                                                   energy_domain_total,
                                                                                   energy_domain_w_left,
                                                                                   energy_domain_w_right,
                                                                                   background_subtraction).values
                w_parm_list.append(w_parm)
                new_energy_list.append(energy)
                if any(self.defect_parameters['energy'] == energy):
                    self.defect_parameters.iloc[self.defect_parameters['energy'] == energy] = pd.Series({
                        'energy': energy_list[ind], 'S': s_parm, 'W': w_parm})
                else:
                    self.defect_parameters = pd.concat([self.defect_parameters,
                                                        pd.DataFrame({'energy': [energy_list[ind]],
                                                                      'S': [s_parm], 'W': [w_parm]})])
                    print(f'W {energy} finised')
        return pd.DataFrame({'energy': new_energy_list, 'W': w_parm_list}).set_index('energy')

    def update_all_defects_parameters(self, energy_domain_total,
                                      energy_domain_s, energy_domain_w_left, energy_domain_w_right,
                                      peak_energy_resolution=1, background_subtraction=True):
        energy_list = self.beam_db.index
        self.calculate_w_parameter(energy_list, energy_domain_total, energy_domain_w_left, energy_domain_w_right,
                                   peak_energy_resolution, background_subtraction)
        self.calculate_s_parameter(energy_list, energy_domain_total, energy_domain_s,
                                   peak_energy_resolution, background_subtraction)
        return self.defect_parameters

    def save(self, files_path):
        """save the data in the se in a file """

    @classmethod
    def from_files(cls, files_path, positron_energies, energy_calibration):
        """work, it loads   """
        if isinstance(energy_calibration, np.poly1d):
            energy_calibration = [energy_calibration for dummy in positron_energies]
        elif all([isinstance(i_energy_calibration, np.poly1d) for i_energy_calibration in energy_calibration]):
            pass
        else:
            print('energy calibration is not in format')
        db = [PASdb.from_file(files_path[ind], energy_calibration[ind]) for ind in range(len(files_path))]
        return BeamDbSet(db, positron_energies)


class BeamCdbSet:
    """
    This object holds a dataframe of energies and doppler broadening spectrum measured in beam
    Using the object the energy dependency of PAS parameters can be extracted.
    Parameters:
    - cdb_pair_list: list of pair
    - energy_list:  !aligned! list of doppler broadening energy

    """

    def __init__(self, cdb_list, energy_list):
        """ Constructor of BeamDbSet measurement"""
        self.defect_parameters = pd.DataFrame({'energy': [], 'S': [], 'W': []})
        self.beam_cdb = pd.DataFrame({'energy': energy_list, 'pair_list': cdb_list})
        self.beam_cdb = self.beam_cdb.set_index('energy')

    def calculate_s_parameter(self, energy_list, energy_dynamic_range, mesh_interval,
                              energy_domain_total, energy_domain_s,
                              peak_energy_resolution=1, background_subtraction=True):
        db_list = [self.beam_cdb.loc[energy]['pair_list'].doppler_broadening_spectrum(self,
                                                                                      energy_dynamic_range,
                                                                                      mesh_interval)
                   for energy in energy_list]
        return BeamDbSet(db_list, energy_list).calculate_s_parameter(energy_list, energy_domain_total,
                                                                     energy_domain_s,
                                                                     peak_energy_resolution, background_subtraction)

    def calculate_w_parameter(self, energy_list, energy_dynamic_range, mesh_interval,
                              energy_domain_total, energy_domain_w_left, energy_domain_w_right,
                              peak_energy_resolution=1, background_subtraction=True):
        db_list = [self.beam_cdb.loc[energy]['pair_list'].doppler_broadening_spectrum(self,
                                                                                      energy_dynamic_range,
                                                                                      mesh_interval)
                   for energy in energy_list]
        return BeamDbSet(db_list, energy_list).calculate_w_parameter(energy_list, energy_domain_total,
                                                                     energy_domain_w_left, energy_domain_w_right,
                                                                     peak_energy_resolution, background_subtraction)

    def update_all_defects_parameters(self, energy_domain_total,
                                      energy_domain_s, energy_domain_w_left, energy_domain_w_right,
                                      peak_energy_resolution=1, background_subtraction=True):
        energy_list = self.beam_cdb.index
        self.calculate_w_parameter(energy_list, energy_domain_total, energy_domain_w_left, energy_domain_w_right,
                                   peak_energy_resolution, background_subtraction)
        self.calculate_s_parameter(energy_list, energy_domain_total, energy_domain_s,
                                   peak_energy_resolution, background_subtraction)
        return self.defect_parameters

    @classmethod
    def from_files(cls, files_path, positron_energies):
        """work, it loads   """
        cdb_list = [PAScdb.from_file(files_path[ind]) for ind in range(len(files_path))]
        return BeamCdbSet(cdb_list, positron_energies)
