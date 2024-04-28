import numpy as np


class Material:
    """
    This is a class that defines a materials in which the diffused positrons propagate
     It has the following parameters:
    - diffusion constant,
    - positron mobility,
    - bulk effective annihilation rate,
    - defects effective annihilation rate
    It should be noted that these are a lot of parameters.
    Technically the mobility and the diffusion can be regarded as the same parameter from Einstein relation
    Parameters
    ----------
    - diffusion: float
    diffusion constant
    - mobility: float
    positron mobility in presence of electrical field (the default is 0 for no electric field)
    - eff_annihilation_rate_bulk:
    The effective annihilation rate is the sum of the multiplication of the different atom density,
     and their counterpart annihilation rate per atom, meaning sum(n_i*k_i) where
     - n_i: concentration of the bulk atoms (the i'th type of atom)
     - k_i: annihilation rate per atom s (the i'th type of atom)
    - eff_annihilation_rate_defects: list
    The effective annihilation rate is the sum of the multiplication of the different defects density,
     and their counterpart annihilation rate per defects, meaning sum(n_i*k_i) where
     - n_i: concentration of the defects (the i'th type of defects)
     - k_i: annihilation rate per defect (the i'th type of defects)

    Attributes
    ----------
    - diffusion: float
    diffusion constant
    - mobility: float
    positron mobility in presence of electrical field (the default is 0 for no electric field)
    - eff_annihilation_rate_bulk: float
        The effective annihilation rate of positrons in the sample bulk
    - eff_annihilation_rate_defects: float
      The effective annihilation rate of positrons in the sample defects

    [1]
    """

    def __init__(self, diffusion, mobility, eff_annihilation_rate_bulk, eff_annihilation_rate_defects, **kwargs):
        self.diffusion = diffusion
        self.mobility = mobility
        self.eff_annihilation_rate_bulk = eff_annihilation_rate_bulk
        self.eff_annihilation_rate_defects = eff_annihilation_rate_defects
        self.annihilation_rates = {'bulk': eff_annihilation_rate_bulk, 'defects': eff_annihilation_rate_defects}
        self.annihilation_rates.update(kwargs)


class Layer:
    """
    Layer is a class that describe a layer in a sample which consist the same material continuously.
    The class include spatial properties as beginning and end location, and a material type.

    """

    def __init__(self, width: float, material: Material, starting_point=0.0):
        self.width = width
        self.material = material
        self.start = starting_point


class Sample:
    """
    Sample a class that describe a multiple layers in which the positron propagate.
    between each two layers there is surface capture rate for the surface overlap
    p.
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
        l = self.layers[-1]
        for layer in self.layers:
            if layer.start < x <= (layer.start + layer.width):
                l = layer
                return l
        return l
