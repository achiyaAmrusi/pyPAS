import pandas as pd
import numpy as np
import xarray as xr
from uncertainties import unumpy
from pyspectrum import spectrum


def load_spectrum_file_to_spectrum_class(file_path, energy_calibration_poly=np.poly1d([1, 0])):
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
    return spectrum.Spectrum(data[data.columns[1]].to_numpy(), data[data.columns[0]].to_numpy(), energy_calibration_poly)


def load_spectrum_file_to_xarray_spectrum(file_path, energy_calibration_poly=np.poly1d([1, 0])):
    """
    load spectrum from a file
    return Spectrum class of the spectrum
    input :
    pyspectrum file - two columns with tab(\t) between them .
     first line is column names - channel, counts
     energy_calibration - numpy.poly1d([a, b])
    """
    # Load the pyspectrum file in form of DataFrame
    try:
        data = pd.read_csv(file_path, sep='\t')
    except ValueError:
        raise FileNotFoundError(f"The given data file path '{file_path}' do not exist.")
    domain = data.columns[0]
    # Create an uncertainty dataset with 'channel' as a dimension
    energy = energy_calibration_poly(data[domain].to_numpy())
    counts = data['counts'].to_numpy()
    counts_with_error = unumpy.uarray(counts, abs(counts) ** 0.5)
    return xr.DataArray(counts_with_error, coords={'energy': energy}, dims=['energy'], name='counts')
