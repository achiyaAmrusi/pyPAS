import numpy as np
from scipy.optimize import nnls

from scipas.core.lifetime import PASLifetime, TimeResolution
from scipas.analysis.lifetime.generator import _convolved_decay


def _components_response_basis(time: np.ndarray, lifetime_components: np.ndarray,
                    resolution: TimeResolution, t0: float) -> np.ndarray:
    """
    Matrix whose column j is the unit-intensity, IRF-convolved decay density
    for "lifetime_components[j]" (t0 enters through the resolution). Shape (len(time), n).

    Columns are densities in 1/time, each with unit integral (depends on the time length); multiply by the
    bin width to get counts per bin.
    """
    basis = np.empty((len(time), len(lifetime_components)))
    for j, tau in enumerate(lifetime_components):
        basis[:, j] = _convolved_decay(
            time,
            lifetimes=np.array([tau]),
            intensities=np.array([1.0]),
            resolution=resolution,
            t0=t0,
        )
    return basis


def _weighted_system(pals: PASLifetime, lifetime_components: np.ndarray, t0: float,
                     background: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Poisson-weighted linear equations system whose least-squares solution is the intensities.
    The function return the response to the lifetime_components A, and the measurements spectrum,
    weighted by the statistical error of the decay measurement.

    Returns
    ------
    (design / counts_std, (counts - background) / counts_std) tuple
      Inputs are not
    validated.
    """
    counts = pals.lifetime.counts
    time = pals.lifetime.energy.values
    dt = time[1] - time[0]

    total = counts.sum() - background * len(counts)
    weight = 1.0 / np.sqrt(np.maximum(counts, 1.0))

    design = (dt * total) * _components_response_basis(time, lifetime_components, pals.resolution, t0)
    return design * weight[:, None], (counts - background) * weight


def solve_intensities(pals: PASLifetime, lifetime_components: np.ndarray, t0: float,
                      background: float) -> np.ndarray:
    """
    Recover the intensities for one nonlinear trial.

    For fixed "(taus, t0, background)" the model is linear in the intensities:

        counts_k ~ background +  I_i * [dt * T * A_i(t_k)],   T =  counts − background·M

    where "A_i" is the unit-intensity IRF-convolved decay density for "tau_i",
    "dt" the bin width and "M" the number of bins. Scaling the columns by
    "dt*T" makes the solved coefficients the intensities themselves, so they
    sum to ~ 1 without the sum rule being imposed. The solver is NNLS, which
    keeps I >= 0. Data rows carry Poisson weights "1/std".

    Parameters
    ----------
    pals : PASLifetime
        Measured spectrum, supplying the time grid, counts and resolution.
    lifetime_components : np.ndarray
        Lifetimes of every component, in component order.
    t0 : float
        Time-zero of the current trial (applied through the resolution).
    background : float
        Per-bin background of the current trial.

    Returns
    -------
    np.ndarray
        Intensity of every component, in component order.
    """
    design, rhs = _weighted_system(pals, lifetime_components, t0, background)
    intensities, _ = nnls(design, rhs)
    return intensities


def solve_intensities_with_covariance(pals: PASLifetime, lifetime_components: np.ndarray, t0: float,
                                      background: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Intensities and their covariance, for one fixed (taus, t0, background).

    Solves the same Poisson-weighted linear system as "intensities.solve_intensities",
    and additionally returns the covariance of the intensity estimate by that equation system.

    The weighted system is Wy = WA@I + eps, i.e. X@I = y_w with X = WA,
    y_w = Wy and Cov(eps) ~ Unity_matrix (the Poisson weights make the noise ~unit
    variance per bin). The least-squares solution is

        I_hat = (X^T X)^-1 @ X^T @ y_w

    Since I_hat is linear in y_w, its covariance follows directly by
    propagating Cov(y_w) = I through that linear map:

        Cov(I_hat) = [(X^T X)^-1 X^T] Cov(y_w) [(X^T X)^-1 X^T]^T
                   = (X^T X)^-1 X^T X (X^T X)^-1
                   = (X^T X)^-1

    This covariance is conditional on holding `lifetime_components`, `t0`
    and `background` exact, so it carries only the Poisson noise of the
    linear solve.

    Parameters
    ----------
    pals : PASLifetime
        Measured spectrum, supplying the time grid, counts and resolution.
    lifetime_components : np.ndarray
        Lifetimes of every component, in component order.
    t0 : float
        Time-zero (applied through the resolution).
    background : float
        Per-bin background.

    Returns
    -------
    tuple of np.ndarray
        " (intensities, cov)"  of shapes " (n,)"  and " (n, n)" .
    """
    design, rhs = _weighted_system(pals, lifetime_components, t0, background)
    intensities, _ = nnls(design, rhs)
    return intensities, np.linalg.pinv(design.T @ design)
