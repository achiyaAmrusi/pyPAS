import numpy as np
from scipy.optimize import nnls

from scipas.core.lifetime import PASLifetime, TimeResolution
from scipas.analysis.lifetime.generator import _convolved_decay


def _response_basis(time: np.ndarray, taus: np.ndarray,
                    resolution: TimeResolution, t0: float) -> np.ndarray:
    """Basis whose column j is the unit-intensity, IRF-convolved decay density
    for ``taus[j]`` (t0 enters through the resolution). Shape (len(time), n).

    Columns are densities in 1/time, each with unit integral; multiply by the
    bin width to get counts per bin.
    """
    basis = np.empty((len(time), len(taus)))
    for j, tau in enumerate(taus):
        basis[:, j] = _convolved_decay(
            time,
            lifetimes=np.array([tau]),
            intensities=np.array([1.0]),
            resolution=resolution,
            t0=t0,
        )
    return basis


def solve_intensities(pals: PASLifetime, taus: np.ndarray, t0: float,
                      background: float) -> np.ndarray:
    """
    Recover the intensities for one nonlinear trial.

    For fixed ``(taus, t0, background)`` the model is linear in the intensities:

        counts_k ≈ background + Σ_i I_i · [dt · T · A_i(t_k)],   T = Σ counts − background·M

    where ``A_i`` is the unit-intensity IRF-convolved decay density for ``τ_i``,
    ``dt`` the bin width and ``M`` the number of bins. Scaling the columns by
    ``dt·T`` makes the solved coefficients the intensities themselves, so they
    sum to ≈ 1 without the sum rule being imposed. The solve is NNLS, which
    keeps I ≥ 0. Data rows carry Poisson weights ``1/σ`` with ``σ² = max(counts, 1)``.

    Parameters
    ----------
    pals : PASLifetime
        Measured spectrum, supplying the time grid, counts and resolution.
    taus : np.ndarray
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
    counts = pals.lifetime.counts
    time = pals.lifetime.energy.values
    dt = time[1] - time[0]

    total = counts.sum() - background * len(counts)
    weight = 1.0 / np.sqrt(np.maximum(counts, 1.0))

    basis = _response_basis(time, taus, pals.resolution, t0)   # (M, n) densities
    design = (dt * total) * basis                              # counts per unit intensity
    rhs = counts - background

    intensities, _ = nnls(design * weight[:, None], rhs * weight)
    return intensities
