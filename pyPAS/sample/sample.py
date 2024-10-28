from pyPAS.sample.layer import Layer


class Sample:
    """
    Describe multiple layers in which the positron propagate.
    In the first layer there is surface capture rate and the last layer continues to infinity
    Parameters
    ----------
    - layers: float
    list of the layers
    - surface_capture_rate: float
    The surface capture rate

    Attributes
    ----------
    - layers: float
    list of the layers
    - size: float
    The sample size [nm]
    - surface_capture_rate: float
    The surface capture rate
    """

    def __init__(self, layers: list, surface_capture_rate: float):
        self.layers = []
        location = 0
        for ind, layer in enumerate(layers):
            layer.start = location
            location = location + layer.width
            self.layers.append(layer)
        self.size = location
        # insert the surface capture rates
        self.surface_capture_rate = surface_capture_rate

    def find_layer(self, x: float) -> Layer:
        """
        find in which layer x is included
        - x: float

        Return
        ------
         x_layer: Layer
         the Layer at which x is included
        """
        x_layer = self.layers[-1]
        for layer in self.layers:
            if layer.start < x <= (layer.start + layer.width):
                x_layer = layer
                break
        return x_layer
