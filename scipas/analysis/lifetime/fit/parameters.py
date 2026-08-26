import numpy as np
from dataclasses import dataclass, replace
from scipy.constants import pico, nano
from warnings import warn
from typing import Sequence

TAU_LOWER_BOUND = pico / nano  # 1 ps, in ns


@dataclass
class FitParameter:
    """
    A single fittable parameter with optional bounds and fixed-value support.

    Parameters
    ----------
    value : float
        Initial guess (if free) or fixed value (if fixed).
    fixed : bool
        If True, parameter is held constant during fitting.
    lower : float
        Lower bound for optimization.
    upper : float
        Upper bound for optimization.
    """
    value: float
    fixed: bool = False
    lower: float = -np.inf
    upper: float = np.inf


def _check_boundary(parm: FitParameter, acceptable_lower: float,
                    acceptable_upper: float) -> tuple[FitParameter, bool]:
    """
    Clamp a parameter's bounds into a physically acceptable range.
    Returns a new FitParameter; adjusted (boolean).
    The boolean is True only when a finite bound was moved, so that the default infinite
    bounds are clamped silently.
    """
    lower = max(parm.lower, acceptable_lower)
    upper = min(parm.upper, acceptable_upper)
    adjusted = ((np.isfinite(parm.lower) and lower != parm.lower)
                or (np.isfinite(parm.upper) and upper != parm.upper))
    return replace(parm, lower=lower, upper=upper), adjusted


def _check_feasible(parm: FitParameter, label: str) -> None:
    """Raise if a parameter's value lies outside its own bounds."""
    if not (parm.lower <= parm.value <= parm.upper):
        raise ValueError(
            f"Initial value of '{label}' ({parm.value}) lies outside its "
            f"bounds [{parm.lower}, {parm.upper}]."
        )


class ParameterMap:
    """
    Bookkeeping for the nonlinear parameters of the lifetime fit.
    The main point is to keep record of the fixed parameters,
     which are needed for the model but aren't part of the optimization.

    Slot layout is [tau_0..tau_{n-1}, t0, background] with length n + 2.
    free_mask is the boolean mask over these slots and is the single
    source for what enters the optimizer vector.

    Parameters
    ----------
    lifetime_components : sequence of FitParameter
        The lifetime of each component, in ns, in the order they appear in the
        slot layout. Each is clamped to the physical range
        "[TAU_LOWER_BOUND, inf)" on construction.
    t0 : FitParameter
        Time-zero, in ns.
    background : FitParameter
        Flat background, in counts per bin.

    Attributes
    ----------
    parameter_names : list of str
        "["tau_0", ..., "t0", "background"]" — labels for the slots. Components
        are identified by index; they carry no user-facing name.
    free_mask : np.ndarray of bool
        Length "n + 2" mask over the slots. True where free.
    bounds_lower, bounds_upper : np.ndarray of float
        Bounds of the free parameters, in slot order.
    """

    def __init__(self, lifetime_components: Sequence[FitParameter],
                 t0: FitParameter, background: FitParameter):
        self._lifetimes = []
        for i, tau in enumerate(lifetime_components):
            tau, adjusted = _check_boundary(
                tau, acceptable_lower=TAU_LOWER_BOUND, acceptable_upper=np.inf)
            if adjusted:
                warn(f"The bounds of tau_{i} were adjusted to the physical "
                     f"range [{TAU_LOWER_BOUND}, inf) ns.")
            self._lifetimes.append(tau)

        self._t0 = t0
        self._background = background

        # Nonlinear slot layout: [tau_0..tau_{n-1}, t0, background].
        nl_params = self._lifetimes + [t0, background]
        for parm, label in zip(nl_params, self.parameter_names):
            _check_feasible(parm, label)

        self._values = np.array([p.value for p in nl_params], dtype=float)
        lower = np.array([p.lower for p in nl_params], dtype=float)
        upper = np.array([p.upper for p in nl_params], dtype=float)
        self.free_mask = ~np.array([p.fixed for p in nl_params], dtype=bool)
        self.bounds_lower = lower[self.free_mask]
        self.bounds_upper = upper[self.free_mask]

    @property
    def parameter_names(self) -> list[str]:
        return [f"tau_{i}" for i in range(self.n_lifetime)] + ["t0", "background"]

    @property
    def n_free(self) -> int:
        return int(self.free_mask.sum())

    @property
    def n_lifetime(self) -> int:
        return len(self._lifetimes)

    def initial_vector(self) -> np.ndarray:
        """Flat vector of initial values for the free nonlinear parameters."""
        return self._values[self.free_mask]

    def pack(self, lt_vals, t0_val, bg_val) -> np.ndarray:
        """Collect full-slot nonlinear values into the flat optimizer vector."""
        all_vals = np.concatenate([np.asarray(lt_vals, dtype=float), [t0_val, bg_val]])
        return all_vals[self.free_mask]

    def unpack(self, x) -> tuple[np.ndarray, float, float]:
        """Expand the flat optimizer vector to (lifetimes, t0, background).

        Fixed slots keep their stored values.
        """
        n = self.n_lifetime
        all_vals = self.full_vector(x)
        return all_vals[:n], float(all_vals[n]), float(all_vals[n + 1])

    def full_vector(self, x) -> np.ndarray:
        """Expand the flat optimizer vector over all slots, fixed ones included.

        Free slots take their value from "x" in order; fixed slots keep the
        value they were constructed with.
        """
        all_vals = self._values.copy()
        all_vals[self.free_mask] = x
        return all_vals

    def embed_covariance(self, cov_free) -> np.ndarray:
        """Place a free-parameter covariance into the full slot layout.

        "cov_free" is the "(n_free, n_free)" covariance in free-slot order, as
        returned by the optimizer. The result is "(n + 2, n + 2)" and is zero on
        every row and column of a fixed slot, a fixed parameter carrying no
        uncertainty.
        """
        n_slots = len(self._values)
        cov = np.zeros((n_slots, n_slots), dtype=float)
        index = np.flatnonzero(self.free_mask)
        #  ix_ makes a cross product for index X index for the free parameters.
        cov[np.ix_(index, index)] = cov_free
        return cov
