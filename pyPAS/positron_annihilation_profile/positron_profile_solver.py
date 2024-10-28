from math import floor
import numpy as np
import xarray as xr
from pyPAS.sample.sample import Sample
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def positrons_annihilation_profile_solver(positron_implementation_profile: xr.DataArray,
                                          sample: Sample,
                                          electric_field: xr.DataArray = None,
                                          mesh_size=10000):
    """
    Finite differences solver for the positron annihilation profile in a sample, c(z).
    The solver solves a transformed positron diffusion equation which is expressed using diffusion lengths on the sample.
    This is achieved by the transformation c'(z) - > c(z)/D and solving for c'(z) rather than c(z)
    detailed description of the solver will be published in future paper [].
    TODO: add the electric field in the boundary condition, and the parameters for annihilation between layers (surface)

    Parameters
    ----------
    - positron_implementation_profile: xarray.DataArray
    Thermal positron implementation profile [positron/second/micrometer] in the sample.
    Note that in the code the profile is linearly interpolated into the mesh points
     See ghosh_profile, makhov_profile
    - sample: Sample
    The sample for which c(z) is calculated,
      it is advised to define the last layer to be very large such that c(z) is expected to be negligible at the end of the sample
    - electric_field: xarray.DataArray
    The electric field in the sample
    If None, taken to be 0
    - mesh_size: int (default 10000)
    Specifies the number of cells used to discretize the 1D domain in the finite element method.

    Returns
    -------
    The positron annihilation profile [annihilation/micron/s]
    """
    # The mesh array
    mesh_points = np.linspace(0, sample.size, mesh_size)
    dz = mesh_points[1] - mesh_points[0]
    diffusion_vector = np.zeros_like(mesh_points)

    # If the field is not defined it is taken to be 0
    if electric_field is None:
        electric_field = xr.DataArray(np.zeros(mesh_points.size), coords={'x': mesh_points})

    # define the differential equation operator matrix
    finite_diff_matrix, diffusion_vector = finite_differences_matrix(sample, electric_field, mesh_points)

    # define the thermal positron profile
    positron_implementation = positron_implementation_profile.interp(x=mesh_points)
    positron_implementation = np.nan_to_num(positron_implementation)
    positron_implementation[-1] = 0
    positron_implementation[0] = positron_implementation[0] * dz / 2

    # solve for the positron distribution
    final_positron_distribution = sp.linalg.spsolve(A=finite_diff_matrix.tocsr(), b=-positron_implementation)

    return xr.DataArray(final_positron_distribution*diffusion_vector, coords={'x': mesh_points})


def finite_differences_matrix(sample: Sample, electric_field: xr.DataArray, mesh_points: np.ndarray):
    """
    Construct the discrete operator form of the positron diffusion equation using central finite differences method.
    Also, return the diffusion constant on the mesh (The operator is for c'(z) == c(z)/D where c(z) the annihilation profile)

    Parameters
    ----------
    - sample: Sample
    The sample for which c(z) is calculated,
      it is advised to define the last layer to be very large such that c(z) is expected to be negligible at the end of the sample
    - electric_field: xarray.DataArray
    The electric field in the sample
    If None, taken to be 0
    - mesh_points: np.ndarray
    mesh vector which is evenly spaced (for example np.linspace(0, sample.size, mesh_size))
    The solver does not support uneven mesh
    Returns
    -------
    (sparse matrix: sp.dia_matrix, diffusion_vector: np.ndarray)
    """
    # definitions
    dz = mesh_points[1] - mesh_points[0]

    main_diag = np.zeros_like(mesh_points)
    off_diag_up = np.zeros(mesh_points.size - 1)
    off_diag_down = np.zeros(mesh_points.size - 1)
    # The diffusion constant at each mesh point
    diffusion_vector = np.zeros_like(mesh_points)

    # calculate the boundary indices for each material
    layers_indices = []
    for i, layer in enumerate(sample.layers):
        start = floor(layer.start / sample.size * mesh_points.size)
        end = floor((layer.start + layer.width) / sample.size * mesh_points.size )
        layers_indices.append((start, end))

    # Calculate the 3 diagonals of the differential equation operator
    # (note: We use central second order finite differences o(dz**2))

    # The bulk part of the operator (Not the boundary) #
    for i, layer_index in enumerate(layers_indices):

        # material definitions in the range of the layer
        # Note: mobility_eff == e*electric_field/k_bT
        material = sample.layers[i].material
        l_sq_inv = sum(material.rates.values()) / material.diffusion
        mobility_eff = material.mobility / material.diffusion

        # padding definitions for the indices boundary
        padding_index_off_up = 1 if i == 0 else 0
        padding_index_off_down = 1 if i == 0 else 0
        padding_index_off_down_boundary = 1 if i == (len(layers_indices) - 1) else 0

        # substitute of the discrete operator values in the diagonals
        main_diag[layer_index[0]: layer_index[1]] = -2 / dz ** 2 - l_sq_inv
        off_diag_up[layer_index[0] - 1 + padding_index_off_up: layer_index[1] - 1] = (
                1 / dz ** 2 -
                np.nan_to_num(electric_field.interp(
                        x=mesh_points[layer_index[0] - 1 + padding_index_off_up: layer_index[1] - 1])) * mobility_eff / dz / 2)
        off_diag_down[layer_index[0] + 1 - padding_index_off_down: layer_index[1] + 1 - 2 * padding_index_off_down_boundary] = (
                1 / dz ** 2 +
                np.nan_to_num(electric_field.interp(
                    x=mesh_points[layer_index[0] + 1 - padding_index_off_down: layer_index[1] + 1 - 2 * padding_index_off_down_boundary])) * mobility_eff / dz / 2)
        diffusion_vector[layer_index[0]: layer_index[1]] = material.diffusion

    # boundary condition #

    # surface boundary
    material = sample.layers[0].material
    l_surface_inv = sample.surface_capture_rate / material.diffusion
    l_sq_inv = sum(material.rates.values()) / material.diffusion

    main_diag[0] = - l_surface_inv - l_sq_inv * dz / 2 - 1 / dz
    off_diag_up[0] = 1 / dz

    # infinity boundary
    main_diag[-1] = 1
    off_diag_down[-1] = 0
    return sp.diags([main_diag, off_diag_up, off_diag_down],
                                  [0, 1, -1], shape=(mesh_points.size, mesh_points.size)), diffusion_vector

