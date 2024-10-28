import pandas as pd
import xarray as xr
from pyPAS.sample.sample import Sample


def profile_annihilation_fraction(positron_profile: xr.DataArray, sample: Sample):
    """
    Function calculate the rates of positron annihilation per annihillation rate.
    The rates are taken from Sample.Layer.Material.rates
    Parameters
    ----------
    - positron_profile: xr.DataArray
     The positron positron_implantation_profile in the sample
    -  sample: Sample
     The sample which contains the different annihilation rate

     Return
     -------
     DataFrame
     table of the annihilation rate for each layer and annihilation spot type
    """
    layers = sample.layers
    layers_location = [(layer.start, layer.start + layer.width) for layer in layers]

    total_annihilation_fractions = pd.DataFrame({'layer': [], 'annihilation_type': [], 'annihilation_fraction': []})

    # surface
    surface_annihilation_fraction = (positron_profile[0] * sample.surface_capture_rate).item()
    total_annihilation_fractions = pd.concat([total_annihilation_fractions,
                                              pd.DataFrame({'layer': [0],
                                                            'annihilation_type': ['surface'],
                                                            'annihilation_fraction': [surface_annihilation_fraction]
                                                            })],
                                             ignore_index=True)
    for i, layer in enumerate(layers):
        layer_positron_profile = positron_profile.sel(x=slice(layers_location[i][0], layers_location[i][1]))
        positron_fraction_in_layer = layer_positron_profile.integrate('x')

        for annihilation_type in layer.material.rates:
            annihilation_fraction = (
                    layer.material.rates[annihilation_type] * positron_fraction_in_layer).item()
            # add all the bulk annihilation fraction of annihilation_type
            total_annihilation_fractions = pd.concat([total_annihilation_fractions,
                                                      pd.DataFrame({'layer': [i],
                                                                    'annihilation_type': [annihilation_type],
                                                                    'annihilation_fraction': [annihilation_fraction]
                                                                    })],
                                                     ignore_index=True)

    # normalization
    total_annihilation_fractions.annihilation_fraction = (total_annihilation_fractions.annihilation_fraction /
                                                          total_annihilation_fractions.annihilation_fraction.sum())
    return total_annihilation_fractions.set_index(['layer', 'annihilation_type'])



