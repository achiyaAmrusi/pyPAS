import numpy as np
import xarray as xr
from scipy.optimize import least_squares
from typing import Sequence
from scipas.analysis.lifetime.fit.intensities import solve_intensities
from scipas.analysis.lifetime.fit.result import FitResult, OptimizerOutput
from scipas.core.lifetime import PASLifetime
from scipas.analysis.lifetime.generator import _convolved_decay
from scipas.analysis.lifetime.fit.parameters import FitParameter, ParameterMap
from warnings import warn
from typing import Literal

#: least_squares options "fit" forwards. An allowlist rather than a blocklist:
#: every name here can only change how the optimizer reaches the minimum, never
#: what the Jacobian at the minimum means, so "estimate_cov" stays valid.
#: Excluded on purpose, because each silently corrupts the covariance --
#: "loss" and "f_scale" minimise something other than a sum of squares,
#: "diff_step" sets the finite-difference step, "jac_sparsity" declares Jacobian
#: entries structurally zero so they are never computed. All three yield a
#: plausible fit and a wrong pcov with no error. Also excluded: "fun", "x0",
#: "bounds" and "args", which "fit" sets itself; "kwargs", which would collide
#: with the args tuple "residuals" expects; and "workers", which needs the
#: residual to pickle and is untested here. "method" is an explicit parameter of
#: "fit", so it binds there and never reaches these keywords.
_FORWARDED_LEAST_SQUARES_ARGS = frozenset(
    {"jac", "ftol", "xtol", "gtol", "x_scale", "max_nfev", "verbose",
     "tr_solver", "tr_options"})


class LifetimeFitter:
    """
    Discrete multi-exponential lifetime spectrum fitter.

    Fits the model:
        N(t) = T * dt * (IRF(t - t0) X [Sum_i (I_i / tau_i) * exp(-t / tau_i) * H(t)]) + bg

    with T = Sum counts - bg * M pinned to the measured spectrum and X convolution on time.

    The fit is separable to the intensities and the nonlinear parameters.
    Because of this, only the nonlinear parameters (tau_i, t0, background)
    reach the optimizer; each of the nonlinear parameters can be free, bounded, or fixed.
    The model is linear in the intensities, so for every nonlinear trial they are recovered
    by a NNLS solver against the data rather than being searched over.

    Notes
    -----
    Residuals are Neyman-weighted: divided by "sqrt(max(observed-counts, 1))".
    The weights are therefore constants of the data rather than functions of the parameters,
    which is what allows the intensities to be recovered by a single linear NNLS solve
    and keeps the fit separable.
    The known cost is a downward bias of about one count on a level parameter.
    It lands on the background, which is fixed by the low-count tail where the
    weighting distorts most, and leaves the lifetimes and t0 effectively
    unchanged.
    A background parameter that must be free of this bias may be measured outside
    the fit and passed fixed or bounded.
    """

    def estimate_cov(self, lsq_result, pmap):
        """
        Function extract the jacobian from scipy.optimize.least_squares results,
        and calculate the covariance matrix.
        The covariance is calculated according to the approximation done in gauss-newton optimization method -
         Hessian = J.T @ J
        The inverse is taken from the singular value decomposition of J rather than
        by forming J.T @ J, whose condition number is the square of J's. Singular
        values below "eps * s[0]" are dropped, so a rank-deficient
        Jacobian yields a pseudo-inverse instead of raising.
        Parameters
        ----------
        lsq_result : scipy.optimize.OptimizeResult
            Return value of "least_squares"; supplies the Jacobian at the optimum.
        pmap : ParameterMap
            Slot layout, used only for the shape of the fallback returned when
            the decomposition fails.

        Returns
        -------
        covariance_matrix np.ndarray
            Covariance of the free parameters, shape "(n_free, n_free)". All
            entries are inf if the Jacobian carries no usable information.

        Warns
        -----
        UserWarning
            When the covariance could not be estimated and inf is returned.
        """
        # Covariance from Jacobian
        try:
            J = lsq_result.jac
            # SVD of J itself: forming J.T @ J squares the condition number
            _, singular_values, vt = np.linalg.svd(J, full_matrices=False)
            keep = singular_values > np.finfo(float).eps * singular_values[0]
            singular_values, vt = singular_values[keep], vt[keep]
            if singular_values.size == 0:
                raise np.linalg.LinAlgError("Jacobian has no significant singular values")

            cov = (vt.T / singular_values ** 2) @ vt

        except np.linalg.LinAlgError:
            cov = np.full((pmap.n_free, pmap.n_free), np.inf)
            warn("Covariance of the parameters could not be estimated")
        return cov

    def residuals(self,
                  parms: np.ndarray,
                  pmap: ParameterMap,
                  pals,
                  sigma)->np.ndarray:
        """
        Neyman-weighted residual of one nonlinear trial.

        Unpacks "parms" onto the full slot layout, solves the intensities at
        those values using nnls optimization and evaluates the lifetime model on the time grid.
        Finally, the function returns "(counts - model) / sigma".

        Parameters
        ----------
        parms : np.ndarray
            Free nonlinear parameters, in the order "pmap" packs them.
        pmap : ParameterMap
            Slot layout supplying the fixed values and the free mask.
        pals : PASLifetime
            Measured spectrum.
        sigma : np.ndarray
            Per-bin standard deviation, "sqrt(max(counts, 1))".

        Returns
        -------
        np.ndarray
            Weighted residual, one entry per bin.
        """
        dt = pals.lifetime.axis[1] - pals.lifetime.axis[0]
        lt_vals, t0_val, bg_val = pmap.unpack(parms)
        I_vals = solve_intensities(pals=pals,
                                   lifetime_components=lt_vals,
                                   t0=t0_val,
                                   background=bg_val)
        total_counts = pals.lifetime.counts.sum() - bg_val * len(pals.lifetime.counts)
        predicted_signal = total_counts*dt*_convolved_decay(time=pals.lifetime.axis,
                                                         lifetimes=lt_vals,
                                                         intensities=I_vals,
                                                         t0=t0_val,
                                                         resolution=pals.resolution)
        predicted_signal += bg_val
        return (pals.lifetime.counts - predicted_signal) / sigma

    def _build_result(self, pmap, pals, lsq_result, cov_free) -> FitResult:
        """
        Assemble a FitResult from the optimizer output.

        Values and covariance are expanded onto the full slot layout, so fixed
        parameters appear with the value they were fixed to and a zero row and
        column in the covariance.

        Parameters
        ----------
        pmap : ParameterMap
            Slot layout of the fit; supplies the parameter names and free mask.
        pals : PASLifetime
            Spectrum that was fitted.
        lsq_result : scipy.optimize.OptimizeResult
            Return value of "least_squares".
        cov_free : np.ndarray
            Covariance of the free parameters, shape "(n_free, n_free)".

        Returns
        -------
        FitResult
        """
        names = pmap.parameter_names

        popt = xr.DataArray(pmap.full_vector(lsq_result.x),
                            dims="parameter",
                            coords={"parameter": names})
        pcov = xr.DataArray(pmap.embed_covariance(cov_free),
                            dims=("parameter", "parameter0"),
                            coords={"parameter": names, "parameter0": names})
        free = xr.DataArray(pmap.free_mask.copy(),
                            dims="parameter",
                            coords={"parameter": names})

        optimizer = OptimizerOutput(success=bool(lsq_result.success),
                                    status=int(lsq_result.status),
                                    message=str(lsq_result.message),
                                    nfev=int(lsq_result.nfev))

        return FitResult(popt=popt, pcov=pcov, free=free, pals=pals,
                         optimizer=optimizer)

    def fit(self,
            pals: PASLifetime,
            lifetime_components: Sequence[FitParameter],
            t0: FitParameter | None = None,
            background: FitParameter | None = None,
            method: Literal["trf", "dogbox"] = "trf",
            **least_squares_kwargs,
            ) -> FitResult:
        """
        Fit a discrete multi-exponential model to a lifetime spectrum.
        Note that Intensities are not given as a parameter, they are solved for.
        Parameters
        ----------
        pals : PASLifetime
            Measured lifetime spectrum with resolution function.
        lifetime_components : list of FitParameter
            One per component, holding the lifetime parameter, its initial
            guess, and its value if fixed. Bounded below by 1 ps.
        t0 : FitParameter, optional
            Time-zero, in ns, applied through the resolution.
            Default: FitParameter(0.0), free and unbounded.
        background : FitParameter, optional
            Background level (counts per bin). Default: FitParameter(0.0, lower=0.0).
        method : str
            Optimization method for least_squares. Default "trf" (Trust Region
            Reflective, supports bounds).
        **least_squares_kwargs
            Additional options forwarded to "scipy.optimize.least_squares",
            Arguments this method sets itself are rejected -
             "fun", "x0", "bounds", "args", "method"  and "loss".
            quantity something else than a sum of squares, and "estimate_cov"
            reads the covariance off the Gauss-Newton Hessian of one.
            Note "jac=\"cs\"" does not work, since the intensity solve is not
            defined for complex input.

        Returns
        -------
        FitResult
            Best-fit values and covariance of the characteristic lifetime, t0 and background.
             The object holds the spectrum that was fitted and the
            optimizer outcome.
             Furthermore, the object is used to calculate the intensities with "opt_parameters",
              and parameters can be drawn randomly from the fit results with "sample".

        Raises
        ------
        ValueError
            If "lifetime_components" is empty, if every parameter is fixed so
            there is nothing to optimize, if an unsupported option is passed in
            "least_squares_kwargs", or if any initial value lies outside
            its own bounds. The last case includes a lifetime below the 1 ps
            floor, since the bounds are clamped to that floor before the value
            is checked against them.
        """
        if len(lifetime_components) == 0:
            raise ValueError("At least one component is required")

        if t0 is None:
            t0 = FitParameter(0.0)
        if background is None:
            background = FitParameter(0.0, lower=0.0)

        sigma = (np.sqrt(np.maximum(pals.lifetime.counts, 1.0)) if pals.lifetime.counts_err is None else pals.lifetime.counts_err)
        pmap = ParameterMap(lifetime_components, t0, background)

        if pmap.n_free == 0:
            raise ValueError("No free parameters - nothing to fit")

        rejected = set(least_squares_kwargs) - _FORWARDED_LEAST_SQUARES_ARGS
        if rejected:
            raise ValueError(
                f"{sorted(rejected)} cannot be forwarded to least_squares. "
                f"Allowed: {sorted(_FORWARDED_LEAST_SQUARES_ARGS)}. The rest are "
                f"either set by fit itself or would leave estimate_cov reading a "
                f"covariance off a Jacobian that no longer means what it assumes."
            )

        options = {"max_nfev": 10000} | least_squares_kwargs

        # noinspection PyTypeChecker
        result = least_squares(self.residuals,
                               pmap.initial_vector(),
                               bounds=(pmap.bounds_lower, pmap.bounds_upper),
                               args=(pmap, pals, sigma),
                               method=method,
                               **options)

        cov = self.estimate_cov(lsq_result=result, pmap=pmap)

        return self._build_result(pmap=pmap, pals=pals, lsq_result=result,
                                  cov_free=cov)
