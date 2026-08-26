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

class LifetimeFitter:
    """
    Discrete multi-exponential lifetime spectrum fitter.

    Fits the model:
        N(t) = T * dt * (IRF(t) X [Σ_i (I_i / τ_i) · exp(-(t - t0) / τ_i) · H(t - t0)]) + bg

    with T = Σ counts − bg · M pinned to the measured spectrum and X convolution on time.

    The fit is separable to the intensities and the nonlinear parameters.
    Because of this, only the nonlinear parameters (τ_i, t0, background)
    reach the optimizer; each of the nonlinear parameters can be free, bounded, or fixed.
    The model is linear in the intensities, so for every nonlinear trial they are recovered
    by a NNLS solver against the data rather than being searched over.
    Currently, there aren't constraints on the intensities, but it will be added in later version.
    """

    def estimate_cov(self, lsq_result, pmap):
        """
        Function extract the jacobian from scipy.optimize.least_squares results,
        and calculate the covariance matrix.
        The covariance is calculated according to the approximation done in gauss-newton optimization method -
         Hessian = J.T @ J
        Parameters
        ----------
        lsq_result
        pmap

        Returns
        -------
        covariance_matrix np.ndarray
        """
        # Covariance from Jacobian
        try:
            J = lsq_result.jac
            Hessian = J.T @ J

            cov = np.linalg.pinv(Hessian, hermitian=True)

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
        Poisson-weighted residual of one nonlinear trial.

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
        dt = pals.lifetime.energy.values[1] - pals.lifetime.energy.values[0]
        lt_vals, t0_val, bg_val = pmap.unpack(parms)
        I_vals = solve_intensities(pals=pals,
                                   taus=lt_vals,
                                   t0=t0_val,
                                   background=bg_val)
        total_counts = pals.lifetime.counts.sum() - bg_val * len(pals.lifetime.counts)
        predicted_signal = total_counts*dt*_convolved_decay(time=pals.lifetime.energy.values,
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
            ) -> FitResult:
        """
        Fit a discrete multi-exponential model to a lifetime spectrum.
        Note that Intensities are not given as a parameter, they are solved for.
        The function will support constraints on the intensities in the future.
        Parameters
        ----------
        pals : PASLifetime
            Measured lifetime spectrum with resolution function.
        lifetime_components : list of FitParameter
            One per component, holding the lifetime parameter, its initial
            guess, and its value if fixed. Bounded below by 1 ps.
        t0 : FitParameter, optional
            Time-zero parameter. .
        background : FitParameter, optional
            Background level (counts per bin). Default: FitParameter(0.0, lower=0.0).
        method : str
            Optimization method for least_squares. Default "trf" (Trust Region
            Reflective, supports bounds).

        Returns
        -------
        FitResult
            Best-fit values and covariance of the charectaristic lifetime, t0 and background.
             The object holds the spectrum that was fitted and the
            optimizer outcome.
             Furthermore, the object is used to calculate the intensities with "opt_parameters",
              and parameters can be drawned randomly from the fit results with "sample".
        """
        if len(lifetime_components) == 0:
            raise ValueError("At least one component is required")

        if t0 is None:
            t0 = FitParameter(0.0)
        if background is None:
            background = FitParameter(0.0, lower=0.0)

        sigma = np.sqrt(np.maximum(pals.lifetime.counts, 1.0))
        pmap = ParameterMap(lifetime_components, t0, background)

        if pmap.n_free == 0:
            raise ValueError("No free parameters — nothing to fit")

        # noinspection PyTypeChecker
        result = least_squares(self.residuals,
                               pmap.initial_vector(),
                               bounds=(pmap.bounds_lower, pmap.bounds_upper),
                               args=(pmap, pals, sigma),
                               method=method,
                               max_nfev=10000)

        cov = self.estimate_cov(lsq_result=result, pmap=pmap)

        return self._build_result(pmap=pmap, pals=pals, lsq_result=result,
                                  cov_free=cov)
