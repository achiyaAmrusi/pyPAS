class Material:
    """
    This is a class that defines a materials in which the diffused positrons propagate
     It has the following parameters:
    - diffusion constant,
    - positron mobility,
    - bulk effective annihilation rate,
    - defects effective capturing rate

    Parameters
    ----------
    - diffusion: float
    positron diffusion constant.
    can be taken as 1 where the effective annihilation rate is then 1/L**2
    - mobility: float
    positron mobility in presence of electrical field (the default is 0 for no electric field)
    - annihilation_rate_bulk:
    The annihilation rate in the bulk λ_b
    - **kwargs
    The kwargs parameters are used to insert different capturing rates.
    for example (eff_defects_capture_rate =  κ_d) or (defects_1= κ_d_1, defects_2 = κ_d_2).
    The parameters can be used for studies on defect density or positronium annihilation rate.

    Note, The defect capturing rate is often described as proportional to the defect density.
    Also, the total change in the probability of a positron being free is dn_b/dx = -(λ_b+κ_d)*n_b + e_d*n_d
    where n_b probability of being free, e_d the escape rate from defect and n_d the density of defects
    The term e_d*n_d is often emitted in diffusion length analysis because n_d is roughly proportional to n_b.
    Thus using effective κ_d is enough.

    Attributes
    ----------
    - diffusion: float
    positron diffusion constant.
    - mobility: float
    positron mobility in presence of electrical field (the default is 0 for no electric field)
    - rates: float
    The effective annihilation and capture rates of positrons in the sample bulk

    """

    def __init__(self, diffusion, mobility, annihilation_rate_bulk, **kwargs):
        self.diffusion = diffusion
        self.mobility = mobility
        self.rates = {'bulk': annihilation_rate_bulk}
        self.rates.update(kwargs)
#        self.effective_length = self.diffusion/sum(self.rates.values())

