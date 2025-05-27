from pyPAS.sample.material import Material


class Layer:
    """
    Describe a layer in a sample which consist one material continuously.
    The class include spatial properties as beginning and end location, and a material type.

    Parameters
    ----------
    - start: float [nm]
    The location where the layer begin
    - width: float
    - width: float
    The width of the sample
    - material: Material
    The material in the layer

    Attributes
    ----------
    - start: float [nm]
    The location where the layer begin
    - width: float [nm]
    The width of the sample
    - material: Material
    The material in the layer
    """

    def __init__(self, starting_point: float, width: float, material: Material):
        self.start = starting_point
        self.width = width
        self.material = material
