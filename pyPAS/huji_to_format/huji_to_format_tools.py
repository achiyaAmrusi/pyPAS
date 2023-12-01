import numpy as np
import pandas as pd


def make_format_id_file(directory_path, sample_name, format_type, detector_name=None):
    """make id_file"""
    id_df = pd.DataFrame(data=[sample_name, format_type], index=['name', 'data_base_type'])
    if not (detector_name is None):
        id_df = pd.concat([id_df, pd.DataFrame(data=[detector_name], index=['detector_id'])])
    id_df.to_csv(directory_path + '/id', sep='\t', index=True, header=False)


def data_file_to_filtered_dataframe(input_file, num_lines_to_remove=5):
    """filter data from pile up and non-relevant lines in the beginning of the counts in time file
    """
    # 3 data bins for each row
    # 3 data bins for each row
    column_names = ['time', 'channel', 'pileUp']
    # Read the data from the input file into a DataFrame without the first lines
    try:
        data = pd.read_csv(input_file, skiprows=num_lines_to_remove, sep=' ', names=column_names, usecols=range(3))
    except FileNotFoundError:
        raise FileNotFoundError(f"The given data file path '{input_file}' does not exist.")

    filtered_data = data[(data.iloc[:, 2] == 0) & (data.iloc[:, 1] >= 0)]
    return filtered_data


def counts_in_time_into_spectrum(counts_in_time, detector_num_of_channels=2 ** 14):
    """takes a data frame with time of count, count and pileup and turn into detector spectrum"""
    # construct spectrum
    spectrum_counts = counts_in_time['channel'].value_counts()
    partial_spectrum = pd.DataFrame({'channel': spectrum_counts.index, 'counts': spectrum_counts.values})
    partial_spectrum = partial_spectrum.set_index('channel')
    # take the last bin to be 0 (bug)
    partial_spectrum.loc[detector_num_of_channels - 1] = 0
    # fills the spectrum with all the channels that didn't got counts
    full_spectrum = pd.DataFrame({'counts': np.zeros(detector_num_of_channels - 1)}, index=list(range(1, 2 ** 14)))
    full_spectrum.index.name = 'channel'
    full_spectrum.update(partial_spectrum)
    return full_spectrum
