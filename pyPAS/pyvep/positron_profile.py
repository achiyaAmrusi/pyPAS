from math import floor
import numpy as np
import xarray as xr
from scipy.integrate import solve_bvp
from pyPAS.pyvep.sample import Sample
from pyPAS.pyvep.utils import material_in_location
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time


def scipy_positrons_annihilation_profile(positron_implementation_profile: xr.DataArray,
                                         sample: Sample,
                                         electric_field: xr.DataArray = None,
                                         num_of_mesh_cells=1000):
    """
    The solution for positron profile boundary problem for given sample and positron energy.
    The solution method here uses scipy solve_bvp function in order to solve the self-consistent problem.
    because of its self-consisting nature this method is slow comparable with the matrix solution.

    Parameters
    ----------
    - positron_implementation_profile: xarray.DataArray
    A function of the thermal positrons per micron in the sample. see also ghosh_profile, makhov_profile
    - sample: Sample
    The sample for which the profile is calculated,
      it is advised to define the last layer to be
        at least 3 mean free path from the end of the profile (or where the implementation is negligible).
    - electric_field: xarray.DataArray
    The electric field value if it exists,
    If None, taken to be 0
    - num_of_mesh_cells: int
    The number of mesh cells for the system
    The function takes the total size of he sample, divide by num_of_mesh_cells,
     and this is the size of ever mesh cell
     default is 1000
    Returns
    -------
    The thermalized positron distribution in units of positrons per micron
    """

    # define the diffusion ODE system for scipy solver
    def ode_system(location, density):
        """
        The ode for the positron diffusion-trapping-annihilation ode[]:
        D*(d**2c(x)/dx**2 )-d/dx(v_d(x)*c(x))+I(x)-sum(labda_defects)*c(x)-lambda_bulk*c(x) = 0
        The diffusion and and the annihilation rate depends on the material which depend on the location.
        density[0] is the positron density
        density[1] is the positron density derivative
        Parameters
        ----------
        - location: list
        location in which the ode is calculated
        - density: list
        the density and the density derivitive

        Returns
        -------
        np.ndarray
        The density derivative and the density second derivative in compliance with the equation above.
        """

        # check in which material the point is, and get the annhilation rates
        materials = [sample.find_layer(x).material for x in location]
        annhilation_rate_bulk = np.array([material.eff_annihilation_rate_bulk for material in materials])
        annhilation_rate_defects = np.array([material.eff_annihilation_rate_defects for material in materials])

        eff_annihilation_rate = np.array([sum(material.annihilation_rates.values()) for material in materials])

        # positron influx in the locations
        I = positron_implementation_profile.interp(x=location)
        I = I.fillna(0)

        # electric field in the location
        if electric_field is None:
            E = np.zeros_like(location)
            dE_dx = np.zeros_like(location)
        else:
            electric_field_deriv = electric_field.differentiate('x')
            E = electric_field.interp(x=location)
            dE_dx = electric_field_deriv.interp(x=location)
        # Derivitives calculation
        # First derivitive
        dc_dx = density[1]
        # second derivitive
        d_2_c_dx_2 = [(1 / material.diffusion) * ( \
                    - material.mobility * (density[1][i] * E[i] + density[0][i] * dE_dx[i]) \
                    + (eff_annihilation_rate[i]) * density[0][i] \
                    - I[i].values) for i, material in enumerate(materials)]
        return np.vstack((dc_dx, d_2_c_dx_2))

    # define the boundary condition for the ode system
    def boundary_conditions(density_in_surface, density_in_deep_bulk):
        """
        The boundary condition definition for the beginning and end of the sample/
       The boundary conditions given are
        (bc 1) c(inf) = 0
        (bc 2) Ddc(0)/dx = $\alpha_s$*c(0) - > radiative condition
        Note: I'd like to make from both vacuum condition and see if it is more compatible
        """
        L_a = sample.layers[0].material.diffusion / sample.surface_capture_rate
        return np.array(
            [density_in_surface[1] - density_in_surface[0] / L_a, density_in_deep_bulk[0]])  # Boundary conditions

    # The mesh array
    mesh = np.linspace(0, sample.size, num_of_mesh_cells)

    # The initial guess of the positron profile is the solution from the fast solve, we can see in scipy if it converge
    initial_guess = np.array([positron_implementation_profile.interp(x=x) for x in mesh])
    initial_guess = xr.DataArray(np.nan_to_num(initial_guess), coords={'x': mesh})

    sol = solve_bvp(fun=ode_system, bc=boundary_conditions, x=mesh,
                    y=np.vstack((initial_guess, initial_guess.differentiate('x'))),
                    max_nodes=max(num_of_mesh_cells + 1, 1000))
    return sol


def fast_positrons_annihilation_profile(positron_implementation_profile: xr.DataArray,
                                        sample: Sample,
                                        electric_field: xr.DataArray = None,
                                        num_of_mesh_cells=1000):
    """
    Fast solver for positron profile boundary problem for given sample and positron energy.
    The solution method here uses finite differences method in order to solve the boundary condition problem.

    TODO: add the electric field in the boundry condition, and the parameters for annhilation between layers (surface)
    Parameters
    ----------
    - positron_implementation_profile: xarray.DataArray
    A function of the thermal positrons per micron in the sample. see also ghosh_profile, makhov_profile
    - sample: Sample
    The sample for which the profile is calculated,
      it is advised to define the last layer to be
        at least 3 mean free path from the end of the profile (or where the implementation is negligible).
    - electric_field: xarray.DataArray
    The electric field value if it exists,
    If None, taken to be 0
    - num_of_mesh_cells: int
    The number of mesh cells for the system
    The function takes the total size of he sample, divide by num_of_mesh_cells,
     and this is the size of ever mesh cell
     default is 1000
    Returns
    -------
    The thermalized positron distribution in units of positrons per micron
    """
    # The mesh array
    mesh = np.linspace(0, sample.size, num_of_mesh_cells)

    # Construct finite difference matrix for the positrons diffusion
    dx = mesh[1] - mesh[0]

    main_diag = np.zeros_like(mesh)
    off_diag_up = np.zeros(len(mesh) - 1)
    off_diag_down = np.zeros(len(mesh) - 1)

    if electric_field is None:
        electric_field = xr.DataArray(np.zeros_like(mesh), coords={'x': mesh})

    # find the index for each material
    layers_indices = []
    for i, layer in enumerate(sample.layers):
        start = floor(layer.start / sample.size * num_of_mesh_cells)
        end = floor((layer.start + layer.width) / sample.size * num_of_mesh_cells)
        layers_indices.append((start, end))

    for i, layer_index in enumerate(layers_indices):
        # material in the range
        mat = sample.layers[i].material
        lambda_eff = sum(mat.annihilation_rates.values())
        diffusion = mat.diffusion
        mobility = mat.mobility
        padding_index_off_up = 0 if i > 0 else 1
        padding_index_off_down = 0 if i < (len(sample.layers) - 1) else -1
        main_diag[layer_index[0]: layer_index[1]] = (-2 * diffusion / dx ** 2 -
                                                     lambda_eff +
                                                     np.nan_to_num(electric_field.interp(x=mesh[layer_index[0]:layer_index[1]])) * mobility / dx)
        off_diag_up[layer_index[0] - 1 + padding_index_off_up: layer_index[1] - 1] = (diffusion / dx ** 2 -
                                                                                      np.nan_to_num(electric_field.interp(x=mesh[layer_index[0] - 1 + padding_index_off_up:layer_index[1] - 1])) * mobility / dx)
        off_diag_down[layer_index[0] + 1: layer_index[1] + 1 + padding_index_off_down] = diffusion / dx ** 2

    # boundary condition
    # x = 0
    L_a = sample.layers[0].material.diffusion / sample.surface_capture_rate
    main_diag[0] = - 1 / dx - 1 / L_a
    off_diag_up[0] = 1 / dx
    # x = end
    main_diag[-1] = 1
    off_diag_up[-1] = 0
    off_diag_down[-1] = 0

    finite_diff_matrix = sp.diags([main_diag, off_diag_up, off_diag_down], [0, 1, -1], shape=(len(mesh), len(mesh)))

    # positron implementation profile
    positron_implementation = positron_implementation_profile.interp(x=mesh)
    positron_implementation = np.nan_to_num(positron_implementation)

    # solve for the positron distribution
    final_positron_distribution = sp.linalg.spsolve(A=finite_diff_matrix.tocsr(), b=-positron_implementation)

    return xr.DataArray(final_positron_distribution, coords={'x': mesh})
