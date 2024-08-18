from pyPAS.pyvedb.sample import Sample


def material_in_location(sample: Sample, location: float):
    """
    find the material of the sample layer in the location
    :param sample:
    :param location:
    :return:
    """
    layer_in_location = sample.find_layer(location)
    return layer_in_location.material