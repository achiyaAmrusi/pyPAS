import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Sequence
from warnings import warn

from scispectrum import Spectrum
from scipas.core.lifetime import PASLifetime
from scipas.analysis.lifetime.generator import _convolved_decay
from scipas.analysis.lifetime.fit.intensities import (
    solve_intensities,
    solve_intensities_with_covariance,
)


MAX_INTENSITY_DRAWS = 100


@dataclass
class OptimizerOutput:
    """
    Outcome of the "least_squares" call that produced a fit.

    Parameters
    ----------
    success : bool
        Whether the optimizer reported convergence.
    status : int
        Termination code returned by "least_squares".
    message : str
        Human-readable termination reason.
    nfev : int
        Number of residual evaluations.
    """
    success: bool
    status: int
    message: str
    nfev: int


@dataclass
class FitResult:
    """
    Result of a discrete multi-exponential lifetime fit.

    Holds the nonlinear parameters the optimizer solved for which include the lifetimes,
    time-zero and background, together with their covariance.
    Note that the intensities are not stored, they are a linear function of these parameters
    and the measured spectrum, recovered by "opt_parameters" or drawn by "sample".

    Parameters
    ----------
    popt : xr.DataArray
        Best-fit values over the "parameter" coordinate
        "[tau_0..tau_{n-1}, t0, background]". Fixed parameters carry the value
        they were fixed to.
    pcov : xr.DataArray
        Covariance over "(parameter, parameter0)", same layout as "popt". Rows
        and columns of fixed parameters are zero.
    free : xr.DataArray
        Boolean over "parameter"; False where a parameter was held fixed.
    pals : PASLifetime
        The measured spectrum the fit was performed on. Required because the
        intensities depend on the counts, not only on "popt".
    optimizer : OptimizerOutput
        Convergence information from the optimizer.
    """
    popt: xr.DataArray
    pcov: xr.DataArray
    free: xr.DataArray
    pals: PASLifetime
    optimizer: OptimizerOutput

    @property
    def n_components(self) -> int:
        """Number of lifetime components."""
        return self.popt.sizes["parameter"] - 2

    def _split(self, values) -> tuple[np.ndarray, float, float]:
        """Split a full slot vector into (lifetimes, t0, background)."""
        n = self.n_components
        values = np.asarray(values, dtype=float)
        return values[:n], float(values[n]), float(values[n + 1])

    def opt_parameters(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Best-fit parameters.

        The lifetimes, time-zero and background come from "popt"; the
        intensities are solved at them for the best-fit decay spectrum.

        Returns
        -------
        tuple
            "(lifetimes, intensities, t0, background)", the first two of shape
            "(n,)" and the last two scalars.
        """
        lifetimes, t0, background = self._split(self.popt.values)
        intensities = solve_intensities(self.pals, lifetimes, t0, background)
        return lifetimes, intensities, t0, background

    def generate(self,
                 lifetimes: Sequence[float],
                 intensities: Sequence[float],
                 t0: float,
                 background: float) -> PASLifetime:
        """
        Evaluate the model on the measured time grid.

        The model is "T * dt * decay + background" with
        "T = counts − background * M", the same expression the optimizer
        minimised against, so the returned spectrum is directly comparable with
        "self.pals".

        Every parameter is required, intensities included. They are never solved
        here: solving them would return the best-fit intensities for the given
        lifetimes and discard any that were drawn, leaving a band that carries
        only the lifetime uncertainty. Use "opt_parameters" or "sample" to obtain
        a complete set.

        Parameters
        ----------
        lifetimes : array-like
            Lifetime of every component, in ns.
        intensities : array-like
            Intensity of every component, same length as "lifetimes".
        t0 : float
            Time-zero, in ns.
        background : float
            Flat background, in counts per bin.

        Returns
        -------
        PASLifetime
            Model counts per bin, on the time axis and resolution of "self.pals".
        """
        lifetimes = np.atleast_1d(np.asarray(lifetimes, dtype=float))
        intensities = np.atleast_1d(np.asarray(intensities, dtype=float))
        if lifetimes.size != intensities.size:
            raise ValueError(
                f"lifetimes and intensities must have the same length, got "
                f"{lifetimes.size} and {intensities.size}")

        counts = self.pals.lifetime.counts
        time = self.pals.lifetime.energy.values
        dt = time[1] - time[0]
        total = counts.sum() - float(background) * len(counts)

        model = total * dt * _convolved_decay(
            time, lifetimes, intensities, self.pals.resolution, float(t0)
        ) + float(background)

        return PASLifetime(
            lifetime=Spectrum(counts=model, axis_calib=self.pals.lifetime.axis_calib),
            resolution=self.pals.resolution,
        )

    def _draw_intensities(self, nonlinear, generator) -> tuple[np.ndarray, int]:
        """One conditional intensity draw for a single nonlinear parameter vector.

        Solves the intensities at "nonlinear", then draws from
        "N(I, cov(I|tau))". A draw with a negative component is rejected and
        redrawn; after "MAX_INTENSITY_DRAWS" failures the last draw is clipped
        at zero.

        Parameters
        ----------
        nonlinear : np.ndarray
            Full slot vector "[tau..., t0, background]".
        generator : np.random.Generator
            Source of the draw.

        Returns
        -------
        tuple
            "(intensities, rejected)" — the draw and how many were discarded.
        """
        taus, t0, background = self._split(nonlinear)
        centre, intensity_fit_cov = solve_intensities_with_covariance(
            self.pals, taus, t0, background)

        for rejected in range(MAX_INTENSITY_DRAWS):
            draw = generator.multivariate_normal(centre, intensity_fit_cov)
            if np.all(draw >= 0.0):
                return draw, rejected
        return np.maximum(draw, 0.0), MAX_INTENSITY_DRAWS

    def sample(self, size: int, rng=None):
        """
        Draw parameter sets from the fit, intensities included.

        Two stages per draw. The nonlinear parameters come from
        "N(popt, pcov)"; the intensities are then solved at those parameters and
        drawn from "N(I, cov(I|tau))". Sampling the second stage rather than
        taking the solved intensities is what carries the Poisson noise of the
        linear solve — without it the spread would hold only the part induced by
        the lifetime uncertainty. Together the two stages realize

            cov(I) = E[cov(I|tau)] + cov(E[I|tau])

        and, because the stages stay paired within a draw, the lifetime-intensity
        cross terms as well.

        Intensity draws with a negative component are rejected and redrawn, the
        Gaussian being a poor description near the non-negativity floor of the
        solve. Draws still negative after "MAX_INTENSITY_DRAWS" attempts are
        clipped at zero.

        Parameters
        ----------
        size : int
            Number of draws.
        rng : int or np.random.Generator, optional
            Seed or generator, for reproducible draws.

        Returns
        -------
        tuple
            "(lifetimes, intensities, t0, background)", the first two of shape
            "(size, n)" and the last two of shape "(size,)". Draw "i" is
            "generate(lifetimes[i], intensities[i], t0[i], background[i])".
        """
        generator = np.random.default_rng(rng)
        n = self.n_components

        nonlinear = generator.multivariate_normal(
            self.popt.values, self.pcov.values, size=size)

        intensities = np.empty((size, n))
        rejected = 0
        for k in range(size):
            intensities[k], attempts = self._draw_intensities(nonlinear[k], generator)
            rejected += attempts

        if rejected > size:
            warn(f"{rejected} intensity draws were rejected for {size} samples; "
                 f"a component is close to the non-negativity floor and the "
                 f"sampled intensity distribution is not Gaussian.")

        return nonlinear[:, :n], intensities, nonlinear[:, n], nonlinear[:, n + 1]
