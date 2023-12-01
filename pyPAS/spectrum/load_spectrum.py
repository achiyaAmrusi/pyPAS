import pandas as pd
import numpy as np
import xarray as xr
from uncertainties import unumpy


def load_spectrum_file_to_xarray_spectrum(file_path, energy_calibration_poly=np.poly1d([1, 0])):
    """
    load spectrum from a file in a DB database directory
    return the spectra in xarray form
    if energy calibration is given (numpy.poly1d), then calibration is preformed.
    input :
    spectrum file - two columns with tab(\t) between them .
     first line is column names - channel, counts
     energy_calibration - numpy.poly1d([a, b])
    """
    # Load the spectrum file in form of DataFrame
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
