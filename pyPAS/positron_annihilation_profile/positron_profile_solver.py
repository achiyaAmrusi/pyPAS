import numpy as np
import xarray as xr
from pyPAS.sample.sample import Sample
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def positrons_annihilation_profile_solver(positron_implantation_profile: xr.DataArray,
                                          sample: Sample,
                                          electric_field: xr.DataArray = None,
                                          mesh_size=10000):
    """
    Finite differences solver for the positron annihilation profile in a sample, c(z).
    The solver solves a transformed positron diffusion equation which is expressed using diffusion lengths on the sample.
    This is achieved by the transformation c'(z) - > c(z)/D and solving for c'(z) rather than c(z)
    detailed description of the solver will be published in future paper [].

    Parameters
    ----------
    - positron_implantation_profile: xarray.DataArray
    Thermal positron implantation profile [positron/second/micrometer] in the sample.
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

    # If the field is not defined it is taken to be 0
    if electric_field is None:
        electric_field = xr.DataArray(np.zeros(mesh_points.size), coords={'x': mesh_points})

    # define the differential equation operator matrix
    finite_diff_matrix = finite_differences_matrix(sample, mesh_points, electric_field)

    # define the thermal positron profile
    positron_implantation = positron_implantation_profile.interp(x=mesh_points)
    positron_implantation = np.nan_to_num(positron_implantation)
    # solve for the positron distribution
    final_positron_distribution = sp.linalg.spsolve(A=finite_diff_matrix.tocsr(), b=-positron_implantation)

    return xr.DataArray(final_positron_distribution, coords={'x': mesh_points})


def finite_differences_matrix(sample: Sample, mesh_points: np.ndarray, electric_field=None):
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
    dx = mesh_points[1] - mesh_points[0]

    if electric_field is not None:
        if not isinstance(electric_field, xr.DataArray):
            raise ValueError("electric field needs to be an xarray.DataArray")
    diag = np.zeros_like(mesh_points)
    diag_upper = np.zeros(mesh_points.size - 1)
    diag_lower = np.zeros(mesh_points.size - 1)

    # The diffusion constant at each mesh point
    diffusion = np.zeros_like(mesh_points)
    drift = np.zeros_like(mesh_points)
    lambda_eff = np.zeros_like(mesh_points)

    # calculate the boundary indices for each layer
    layers_indices = np.zeros((len(sample.layers), 2))
    for i, layer in enumerate(sample.layers):
        start = int(np.round(layer.start / sample.size * mesh_points.size))
        end = int(np.round((layer.start + layer.width) / sample.size * mesh_points.size))
        layers_indices[i] = (int(start), int(end))


    # set the coefficients
    for i in range(layers_indices.shape[0]):
        start, end = int(layers_indices[i][0]), int(layers_indices[i][1])
        diffusion[start:end] = sample.layers[i].material.diffusion
        lambda_eff[start:end] = sum(sample.layers[i].material.rates.values())
        if electric_field is not None:
            drift[start:end] = sample.layers[i].material.mobility * electric_field.interp(x=mesh_points[start:end])

    # Calculate the 3 diagonals of the differential equation operator
    # (note: We use central central finite differences o(dz**2))

    diag[1:-1] = -((diffusion[2:] + 2*diffusion[1:-1]+ diffusion[:-2]) / 2 / dx ** 2 + lambda_eff[1:-1])
    diag_upper[1:] = (diffusion[2:] + diffusion[1:-1]) / 2 / dx ** 2 - drift[2:] / 2 / dx
    diag_lower[:-1] = (diffusion[1:-1] + diffusion[:-2]) / 2 / dx ** 2 + drift[:-2] / 2 / dx

    if sample.surface_capture_rate == 0:
        l_a = np.inf  # Effectively means no surface capture
    else:
        l_a = diffusion[0] / sample.surface_capture_rate
    if lambda_eff[-1] == 0:
        l_bulk = np.inf
    else:
        l_bulk = (diffusion[-1] / lambda_eff[-1]) ** 0.5
    # boundary conditions are taken on the centers of the cells and not the edges for stability
    diag[0] = -( 2*diffusion[0] / dx ** 2 + lambda_eff[0] + (diffusion[0] / dx ** 2 + drift[0] / 2 / dx) * 2 * dx /l_a)
    diag_upper[0] = 2*diffusion[0] / dx ** 2


    diag[-1] = -(2*diffusion[-1] / dx ** 2 + lambda_eff[-1] + (diffusion[-1] / dx ** 2 - drift[-1] / 2 / dx) * (2 * dx / l_bulk))
    diag_lower[-1] = 2*diffusion[-1] / dx ** 2

    return sp.diags([diag, diag_upper, diag_lower],
                    [0, 1, -1], shape=(mesh_points.size, mesh_points.size))

