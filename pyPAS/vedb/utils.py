from pyPAS.sample.sample import Sample


def material_in_location(sample: Sample, location: float):
    """
    find the material of the sample layer in the location
    Parameters
    ----------
    - sample: Sample
    The sample
    - location: float
    The location at which the type of material is found
    Returns
    -------
    Material
    The material at the location
    """
    layer_in_location = sample.find_layer(location)
    return layer_in_location.material
