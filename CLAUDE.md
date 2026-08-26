# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**SciPAS** is a Python library for Positron Annihilation Spectroscopy (PAS) analysis. It covers Doppler broadening (DB) and coincidence Doppler broadening (CDB) spectrum analysis, positron implantation profiling, finite-difference transport simulation, and variable-energy Doppler broadening (VEDB) diffusion-length fitting.

**Project name:** SciPAS  
**PyPI / import name:** `scipas` (the package directory is `scipas/`)  
**GitHub:** `achiyaAmrusi/scipas`  
**Author:** Achiya Yosef Amrusi  
**Email:** ahia.amrosi@mail.huji.ac.il (Hebrew University of Jerusalem)  
**License:** MIT  
**Python:** 3.11+ (SPEC 0 policy; tested through 3.13)  
**Build:** `setuptools.build_meta` via `pyproject.toml` (no `setup.py`)  
**Status:** Active development. DB/CDB analysis, transport, and VEDB fitting are stable. Lifetime analysis lives on the `lifetime` branch only.

---

## Install and Develop

```bash
pip install -e "/home/owner/gitProjects/scipas[dev]"
```

The companion spectrum library **SciSpectrum** (`scispectrum>=0.3` on PyPI) is listed in `pyproject.toml` dependencies and installed automatically.

---

## Running Tests

```bash
cd /home/owner/gitProjects/scipas
pytest tests/
```

All test files carry the `test_` prefix and are discovered by pytest. The full suite (97 tests on main, 168 on lifetime) passes on Python 3.11–3.13. Test directories mirror the package layout (`tests/analysis/lifetime/fit/`, `.../inversion/`). There are no `__init__.py` files under `tests/`, so pytest puts each test directory on `sys.path` and shared helpers are imported as `from conftest import ...`; adding `__init__.py` would break that. `tests/transport/diffusion/scipy_positron_profile_solver.py` is a reference/validation helper, not a test file.

CI: `.github/workflows/tests.yml` runs pytest on Python 3.11, 3.12, 3.13 on every push and PR.

---

## Architecture

### Full PAS workflow

```
Raw detector files (list-mode .txt)
  └─ scispectrum TimeChannelParser → Spectrum

DB pipeline:
  Spectrum
    └─ DB.from_spectrum()  → DB (Domain slice around 511 keV peak)
         ├─ .s_parameter_calculation() → ufloat
         └─ .w_parameter_calculation() → ufloat

CDB pipeline:
  PasCoincidenceFilter.time_coincidence_filter()   → coincident pairs
  PasCoincidenceFilter.energy_coincidence_filter() → energy-validated pairs
  CDB(pairs, energy_min, energy_max, mesh_interval)
    ├─ .doppler_broadening() → DB
    └─ .resolution()         → Domain

Transport + VEDB fitting:
  makhov_profile / ghosh_profile  (one per beam energy) [nm-depth, positrons/nm]
    └─ multilayer_implantation_profile → list of xr.DataArray
         └─ profile_solver(profile, sample, electric_field) → annihilation profile [positrons/nm]
              └─ compute_annihilation_fractions(profile, sample) → fractions per layer
                   └─ DiffusionLengthOptimization(profiles, s_measurement, initial_sample)
                        └─ .optimize_diffusion_length() → (best_fit [nm], covariance)
```

### Module map

```
scipas/
├── __init__.py              public API exports
├── core/
│   ├── db.py               DB — extends Domain; S/W parameter extraction
│   ├── cdb.py              CDB — 2D coincidence histogram; DB/resolution projections
│   ├── lifetime.py         PASLifetime (LIFETIME BRANCH ONLY)
│   ├── time_resolution.py  TimeResolution, MeasuredRF, MultiGaussianRF (LIFETIME BRANCH ONLY)
│   └── const.py            ELECTRON_REST_MASS_KEV (computed from scipy.constants)
├── filter/
│   ├── pas_coincidence.py  PasCoincidenceFilter — time + energy coincidence for CDB
│   └── pals_coincidence.py PALSCoincidenceFilter — STUB
├── libs/
│   └── positron_profile/   Ghosh & Makhov parameter tables (.txt CSV files)
├── transport/
│   ├── implantation/
│   │   ├── profiles.py           makhov_profile, ghosh_profile
│   │   ├── multilayer.py         multilayer_implantation_profile
│   │   └── material_parameters.py loads tables via importlib.resources from scipas.libs.positron_profile
│   └── diffusion/
│       └── positron_profile_solver.py  profile_solver — 1D FD diffusion-drift-annihilation solver
├── model/
│   ├── material.py         Material dataclass (diffusion, mobility, bulk_annihilation_rate, defects)
│   ├── layer.py            Layer dataclass (material, start, width)
│   ├── sample.py           Sample dataclass (layers, absorption_length)
│   └── lifetime.py         LifetimeModel dataclass (lifetimes, intensities)
└── analysis/
    ├── vedb/
    │   ├── annihilation_fractions.py  compute_annihilation_fractions
    │   ├── diffusion_length.py        DiffusionLengthOptimization
    │   ├── lineshape.py               compute_s_lineshape, compute_w_lineshape
    │   └── ve_implanation.py          variable_energy_implantation_profiles
    └── lifetime/                      LIFETIME BRANCH ONLY — not in main
        ├── generator.py         synthetic spectra (analytical + Poisson-sampled)
        ├── fit/
        │   ├── parameters.py   FitParameter, ParameterMap
        │   ├── intensities.py  solve_intensities, solve_intensities_with_covariance
        │   ├── fitter.py       LifetimeFitter (forward model + least squares)
        │   └── result.py       FitResult, OptimizerOutput
        └── inversion/
            ├── base.py             LifetimeInvert — common interface
            ├── tikhonov.py         TikhonovRegularization
            ├── maximum_entropy.py  MaximalEntropyInversion (MELT, Bryan 1990)
            ├── gp_regression.py    GPRegression — Gaussian-process inversion
            └── utils.py            _response_matrix, _svd_truncate, t0_scan
```

---

## Critical Dependency — SciSpectrum

SciSpectrum (on PyPI as `scispectrum`, source at `/home/owner/gitProjects/scispectrum`) provides the foundational types:

| Type | Role in SciPAS |
|---|---|
| `Spectrum` | Universal 1D count array with calibrated axis and Poisson errors |
| `Domain` | Contiguous slice of a `Spectrum` — **`DB` inherits from it** |
| `AxisCalibration` | `channel → energy` callable |
| `ResolutionCalibration` | Models detector FWHM vs energy; required by `SNRFinder` |
| `SNRFinder`, `Convolution`, `gaussian_2_dev` | Automatic 511 keV peak detection inside `DB.from_spectrum` |
| `center_estimator`, `sum_under` | Peak centroid finding and windowed integration for S/W |

Import with `from scispectrum import ...` (lowercase package name).

Key SciSpectrum invariants:
- `Domain` uses lazy background subtraction — background is stored and applied only when `.data` is accessed.
- All `Spectrum` / `Domain` arithmetic propagates uncertainties via the `uncertainties` library.
- `SNRFinder` requires `ResolutionCalibration` attached to the `Spectrum` before calling.

---

## Non-Obvious Design Rules

### Units — these are strict throughout
- Depth / position: **nm**
- Diffusion coefficient: **nm²/ps**
- Mobility: **nm²/(ps·V)**
- Annihilation rate (λ): **1/ps**
- Electric field: **V/nm**
- Implantation profiles: **positrons/nm** (must be normalised so integral = 1)
- Implantation energy for profiles: **keV**
- Material density (for Ghosh/Makhov profiles): **g/cm³**

### Governing PDE (`profile_solver`)
```
d/dz[ D(z) dc/dz ] − μ(z) E(z) dc/dz − λ(z) c(z) = −g(z)
```
Boundary conditions are **radiative** at both surfaces:
- `dc/dz|_{z=0}  =  c(0) / L_a`  (surface absorption length from `Sample.absorption_length`)
- `dc/dz|_{z=L}  = −c(L) / L_bulk`  (bulk diffusion length `sqrt(D/λ)`)

The FD solver handles discontinuous material coefficients at layer interfaces correctly. The scipy BVP solver (`tests/transport/diffusion/scipy_positron_profile_solver.py`, used only for validation) does not handle discontinuous interfaces well.

### Sample / Layer construction
`Sample.__post_init__` auto-computes `Layer.start` from widths in list order — never set `start` manually when building a `Sample`. The last layer must be thick enough that `c(z) ≈ 0` at its far end.

### DB and Domain
`DB` inherits `spectrum`, `start`, `stop`, `background`, and `data` from `Domain`. `DB` itself only adds `s_parameter_calculation`, `w_parameter_calculation`, `from_spectrum`, `from_domain`, and `recenter`. When modifying `DB`, be aware that `self.data` is background-subtracted lazily.

### `ELECTRON_REST_MASS_KEV`
Computed from `scipy.constants`, not hardcoded as 511.

### xarray coordinate name
All depth-dependent `xr.DataArray` objects (implantation profiles, electric field, solver output) use coordinate name **`'x'`** in nm. `profile_solver` interpolates the input profile onto its mesh via `.interp(x=mesh_points)`.

### DiffusionLengthOptimization — normalized trial samples
Inside the optimizer, trial samples are constructed with `D=1`, `λ = 1/L²` so that `L_eff = sqrt(D/λ) = L`. Absolute D and λ are not independently identifiable from VEDB data.

### Lifetime — forward model and the two problems

A measured PALS spectrum is counts vs. time,

```
y(t) = N · ∫ R(t; τ) f(τ) dτ + bg,     R(t; τ) = IRF(t) ⊗ (1/τ)e^{-t/τ}Θ(t)
```

`f(τ)` is the lifetime distribution, `R(t; τ)` the detector response to a single
exponential component, `N` the total counts, `bg` a flat background. Two problems
follow, and they are the two subpackages:

1. **Discrete fitting** (`fit/`) — a few known components; find τ_i and I_i.
2. **Inversion** (`inversion/`) — no parametric assumption; recover `f(τ)`. This
   is a Fredholm integral equation of the first kind (exponential / Laplace-type),
   so it is ill-posed and every method differs *only in how it regularizes*.

All times are in **nanoseconds**; intensities are dimensionless and their sum is
not constrained to 1 (see the result-layout and Decisions sections below).

### Lifetime fit — result layout and the four-array parameter set

`LifetimeFitter.fit` returns a `FitResult` holding `popt` / `pcov` / `free` as
`xr.DataArray` over a `parameter` coordinate `[τ_0..τ_{n-1}, t0, background]`,
plus `pals` and an `OptimizerOutput`. Both arrays are **full slot-sized**: a
fixed parameter keeps the value it was fixed to and gets a zero row and column
in `pcov`. Consequences worth knowing:

- `sample` needs no special case for fixed parameters — a zero-variance
  direction reproduces the fixed value in every draw.
- Anything propagating `pcov` (e.g. `∂I/∂θ · pcov · ∂I/∂θᵀ`) gets the right
  answer for free, since fixed slots contribute nothing.

Intensities are **never stored** — they are a linear function of `popt` and the
counts. `opt_parameters()` and `sample(size)` both return the same four things,
differing only by a leading draw axis:

```
(lifetimes, intensities, t0, background)      shapes (n,) (n,) scalar scalar
(lifetimes, intensities, t0, background)      shapes (size,n) (size,n) (size,) (size,)
```

`generate(lifetimes, intensities, t0, background)` requires **all four** and
never solves the intensities itself. That is deliberate: solving would return
the best-fit intensities for the given lifetimes and silently discard a drawn
one, collapsing an uncertainty band to only its lifetime term. To evaluate at
lifetimes of your own, call `solve_intensities` and pass the result.

### Lifetime inversion — what each method regularizes

All inverters share one interface, `SomeInversion(time_grid, characteristic_time_grid)`
then `.invert(pals, bg_est=..., t0_shift=...)`. The response matrix is built
internally by `utils._response_matrix`: column j is the IRF-convolved decay for
τ_j evaluated on the t0-shifted time grid.

**Return signatures are not uniform** — `TikhonovRegularization` returns
`(q, OptimizeResult)` and `GPRegression` returns `(f_density, metadata)`, but
`MaximalEntropyInversion` returns `(alpha_opt, f_hat)`, distribution **second**.
That is why `utils.t0_scan` carries
`q = result[0] if isinstance(result[0], np.ndarray) else result[1]`.

- **Tikhonov.** Minimizes `‖R f − y‖² + α‖D²f‖²` subject to `f ≥ 0`, solved as one
  augmented NNLS; α by the discrepancy principle, `min |χ²/N − 1|`, over `log α`.
  Fast, and over-smooths narrow features. Its docstring records an unexplained
  sensitivity to sub-bin shifts: if the tail deviates, shifting the spectrum half
  a bin can fix it.
- **MELT** (Bryan 1990). Maximizes `αS(f) − ½χ²` with S the Shannon entropy against
  a flat prior, inside the truncated-SVD subspace. The parametrization
  `f_i = m_i·exp((U u)_i)` keeps `f > 0` structurally and reduces the search from
  `n_τ` to the SVD rank; α is found by Powell in `log α` with `u` warm-started
  between steps. Good peak localization, no uncertainty estimate. Normalization
  uses `np.trapezoid`, not a rectangular sum, so `response @ f` can sit slightly
  off 1 through the boundary weights.
- **GP.** Models the **log** of the distribution as a GP, making positivity
  structural rather than a constraint: `K = exp(log_amp)·exp(−(Δ log τ)²/2ℓ²)`,
  `c = e^g`, `y = N·(RM @ c)`. Three choices carry the method: the kernel lives on
  **log τ**, matching the ratio-resolution of exponential analysis; `RM` is
  column-sum-normalized so `c` are probability weights and the free `N` carries the
  counts, decoupling shape from scale; the prior mean `log(1/n_τ)` pulls toward flat
  where the data say nothing. MAP of `[log N, g]` by L-BFGS with analytic gradients,
  then a Laplace posterior `cov(g) = (W + K⁻¹)⁻¹` and the delta method `σ_f = f·σ_g`
  — the only calibrated error bars of the three. ℓ and amplitude are chosen by the
  Laplace evidence `−log Z ≈ ½χ² + ½(g−m)ᵀK⁻¹(g−m) + ½ log det(I + K·W)`, whose
  log-det term is the Occam penalty; the grid is walked smooth→flexible with warm
  starts, and models within 1 nat of the best are treated as ties with the smoothest
  ℓ winning. **Effective resolution therefore tracks the statistics** — more counts,
  smaller supported ℓ, previously merged components separate.

| | Tikhonov | MELT | GP |
|---|---|---|---|
| Peak localization | fair | good | good |
| Narrow features | over-smoothed | good | good |
| Uncertainty estimate | – | – | ±σ (Laplace) |
| Resolution adapts to statistics | – | – | yes (evidence) |
| Cost | seconds | seconds | ~minute (hyperparameter grid) |

Every inverter takes `t0_shift`; `utils.t0_scan(inverter, pals, t0_values, **kw)`
scans it and returns `best_t0` / `best_result` / `chi_squared`. In discrete
fitting t0 is instead a free `FitParameter`.

### `annihilation_fractions` layer coordinate convention
`compute_annihilation_fractions` returns an `xr.DataArray` with coordinate `'layer'`:
- `layer = -1` : surface annihilation
- `layer = 0, 1, 2, …` : bulk layers in depth order

---

## Branch Strategy

- **`main`** — paper submission branch. No lifetime module.
- **`lifetime`** — active development for positron lifetime spectrum analysis. All `analysis/lifetime/` work happens here. Do not merge back to `main` without discussion.

---

## Known Issues (publication deadline 2026-07-03)

All reviewer code issues for the CPC revision are resolved. Remaining items are future work.

### Stub files needing implementation
- `filter/pals_coincidence.py` — PALS timing coincidence filter (out of scope for CPC revision)

---

## To Do (lifetime branch)

Discrete fitting is wired end to end: `fit` returns a `FitResult`, intensities
are solved by variable projection, and uncertainties come from sampling. What
remains is listed under *Pending*.

### Fitter — implemented

- **t0 lives in the resolution.** `TimeResolution.convolve(signal, time, t0=0)` evaluates the IRF on the grid of *differences*, `(arange(n) - zero_point_index)*dt - t0` with `zero_point_index = round(-time[0]/dt)` — the same index at which the `np.convolve` output is sliced back onto `time`. `MultiGaussianRF` shifts analytically; `MeasuredRF.evaluate` interpolates (`np.interp`, 0 outside support). Dissolves the discrete-onset problem — no post-convolution t0 interpolation in the fitter.
- **Shared kernel `_convolved_decay(time, lifetimes, intensities, resolution, t0=0)`** in `generator.py` — validation-free, returns a **density**, so counts per bin are `dt·T·density + bg`. `generate_analytical_lt_spectrum` / `generate_random_lt_spectrum` and the fitter all route through it.
- **Exact onset integration.** Each bin holds the decay's mean over the bin, `exp(-t/τ)(1-exp(-dt/τ))/dt`, with the bin straddling `t=0` integrated from 0 rather than from its left edge (`k0 = searchsorted(time, 0.0, "right") - 1`). Sampling at the bin edge overestimates by `dt/2τ` — a Euler–Maclaurin boundary term, 5% at τ=0.1 ns, dt=0.01 ns.
- **Lifetimes are plain `FitParameter`s**, passed as `fit(pals, lifetime_components=[...])`; intensities are not configured. Components carry no name — they are referenced by index, and slot labels are derived positionally as `tau_0 … tau_{n-1}`. `ParameterMap.__init__` clamps every lifetime to the 1 ps floor (`TAU_LOWER_BOUND`) via `_check_boundary`, which returns a new `FitParameter` (`dataclasses.replace`) and warns only when a *finite* bound was moved, so the default `-inf` clamps silently; the warning names the component by index. Clamping runs **before** the feasibility check, so a sub-picosecond τ raises rather than being silently raised to the floor. `ParameterMap.__init__` also rejects an initial value outside its own bounds rather than letting `least_squares` fail with "x0 is infeasible".
- **Separable fit (variable projection).** The optimizer sees only `[τ_i, t0, background]`. `ParameterMap` owns the slot layout with `free_mask` as the single source of truth; `pack` / `unpack` / `full_vector` / `embed_covariance` / `bounds_*` / `n_free` all derive from it.
- **Intensity solve (`fit/intensities.py`).** `solve_intensities` builds the unit-intensity response basis via `_convolved_decay`, pins `T = Σcounts − bg·M`, scales columns by `dt·T` and solves by **NNLS** with Poisson row weights `1/√max(counts,1)`. `solve_intensities_with_covariance` returns the intensities *and* `cov(I|τ) = (DᵀD)⁻¹` from one build of the weighted system — call it rather than solving twice, which is what `_draw_intensities` does per draw.
- **Result and sampling.** See *Lifetime fit — result layout* under Non-Obvious Design Rules.

### Fitter — pending

- **`core/lifetime.py` and `core/time_resolution.py` have no tests.** `tests/core/test_lt.py` holds fixtures and zero test functions, so `convolve` — where two independent bugs were fixed (the `zero_point_index` off-by-one and the lag-axis offset) — is unguarded. A regression there is silent and corrupts every fit.
- **Inverter return signatures disagree.** `MaximalEntropyInversion.invert` returns `(alpha_opt, f_hat)` while `TikhonovRegularization` and `GPRegression` return the distribution first. `utils.t0_scan` papers over it with an `isinstance` test on `result[0]`. Settle on distribution-first and fix the caller.
- **Docstring cleanup.** `estimate_cov` has an empty parameter template and no `Returns`, and omits the one non-obvious fact about it — the VP Jacobian makes `(JᵀJ)⁻¹` already the marginal covariance with intensities profiled out, with no `χ²_red` rescaling. Both the `LifetimeFitter` class docstring and `fit` carry a roadmap sentence promising intensity constraints, which belongs here, not in a contract. `fit` documents `background`'s default but not `t0`'s, has a stray `. .` and the typos "charectaristic" / "drawned", and no `Raises` despite two `ValueError`s. The class docstring writes the model as `exp(-(t-t0)/τ)·H(t-t0)`, putting t0 in the decay where the code puts it in the IRF — equivalent under convolution, but it sends the reader looking for a post-convolution shift. In `generator.py` the private `_convolved_decay` has full numpydoc while both public generators have two lines, and `generate_random_lt_spectrum` neither documents `num_events` nor mentions that it draws from the global `np.random` with no seed argument, unlike `FitResult.sample`. Code is quoted as `"..."` in `parameters.py` / `result.py` and as ``` ``...`` ``` in `intensities.py`.

### Future work — deliberately deferred

Not on the current plan. Recorded so the design is not re-derived from scratch.

- **User intensity constraints (equality only).** Tie intensities with linear equalities, e.g. `I_defect = I_bulk/2`, or pin one for source correction. All of it is one system `C I = d`: fixing is a row of the identity, a ratio a two-term row, the sum rule the ones row. Planned surface: `IntensityConstraint(terms={index: coeff}, value)` (+ `.fixed(i, v)` / `.ratio(a, b, k)`), passed as `constraints=[...]`; `ParameterMap` exposes `constraint_matrix()/constraint_rhs()`; `solve_intensities` appends them as heavily-weighted penalty rows and reports the max violation (exact equality with `I ≥ 0` would need a QP). Validate `rank(C) < n`, consistency (`rank([C|d]) == rank(C)`), and unknown indices at construction. Inequalities are out of scope. Note the interaction with component ordering under *Decisions*: a constraint references components by index, so it is the user's job to set non-overlapping τ bounds before using one.
- **Simultaneous fitting of several spectra** with parameters shared across them (a temperature or dose series with common τ, or a source correction constrained across samples). `LifetimeFitter.fit` takes a single `PASLifetime`, and `ParameterMap` owns one slot layout, so this needs shared-parameter infrastructure across spectra rather than an extra argument. Left to the user to compose from the existing pieces for now.

### Decisions

- **The sum rule is not imposed.** `T` is pinned to the data and the basis columns have unit integral, so `Σ I ≈ 1` falls out of the solve. Forcing it destroys information: off the optimum `Σ I` drifts (0.977 at τ=[0.25,1.3] against 0.999 at the truth), which makes it a free diagnostic of a wrong model or wrong scaling. **Calibrate the baseline before reading it: `Σ I` sits at `1 − M/T`, not at 1**, because of the Neyman background bias above — 0.998 at M=1700 and T=10⁶, but 0.992 at 8000 bins on the same counts. The offset scales with binning, so a finely binned spectrum reads as "wrong model" if the baseline is assumed to be 1.000.
- **Uncertainties by hierarchical sampling, not a joint Jacobian.** `sample` draws θ from `N(popt, pcov)`, then per draw solves the intensities and draws them from `N(I, cov(I|τ))`. This realizes `cov(I) = E[cov(I|τ)] + cov(E[I|τ])` — both the direct and the τ-induced term — plus the τ–I cross terms, since the stages stay paired within a draw. **The second stage is not optional:** taking `solve_intensities` deterministically per draw yields `cov(E[I|τ])` alone and silently understates the spread.
- **Neyman weighting is deliberate; the −1 count background bias is accepted.** `residuals` and `_weighted_system` both divide by `sqrt(max(counts,1))` — the *observed* counts, so the weights are data constants. That is the textbook Neyman χ², biased low by exactly 1 count; the alternatives measured over 100 synthetic fits are `bias(bg)` = −1.075 (Neyman), +0.472 (Pearson), −0.020 (Poisson deviance), which are the harmonic mean, the RMS and the arithmetic mean of the counts respectively. **The bias is an absolute offset, not a relative one** — it stays at −1 whether the level is 10 counts or 1000 — and PALS runs at ≥1M events, so it lands on `bg` (a nuisance parameter) while τ and t0 barely move. Its one path onward is the **bin-count amplification**: `bg` enters `T = ΣN − bg·M` multiplied by M, so a −1 count/bin bias makes `T` high by `M` counts and every `I_i` low by the same relative amount, `≈ M/T`. Measured over 12 fits at M=1700, T=10⁶: `bias(bg)` = −1.117 ± 0.075, predicted `Σ I` = 0.99810, measured 0.99829 — the mechanism accounts for ~90% of the departure from 1. For an individual intensity that is 0.13σ against `σ(I) ≈ 0.0085`, negligible; for `Σ I` it is ~1.8σ against `σ(ΣI) ≈ 0.001`, i.e. the dominant term (see the sum-rule decision below). Rejected the deviance swap because it is not local: the deviance is not quadratic in `I`, so the inner solve stops being one exact `nnls` call and becomes IRLS (weights `1/μ` rebuilt from the previous pass, a **fixed** iteration count — a convergence tolerance would make the residual discontinuous and wreck the finite-difference Jacobian). That breaks the single-objective consistency that makes variable projection valid and would force re-verification of `estimate_cov` and `sample`. Pearson is worse still here: `bg` defaults to `FitParameter(0.0, lower=0.0)` and grids start well before t0, so `μ → 0` over a few hundred pre-onset bins and `1/μ` explodes.
- **The redundant convolution in `residuals` is left in place.** `residuals` calls `solve_intensities` (which builds the `n` basis columns) and then `_convolved_decay` again, and by linearity of convolution that second call is exactly `basis @ I` (verified to 2.5e-16 relative). Removing it means threading an optional prebuilt `basis` through `solve_intensities` and `_weighted_system` — an argument that can silently disagree with the `lifetime_components`/`t0` passed beside it, with nothing to check it. Measured cost of keeping it: 3 ms of a 0.165 s fit at 1700 bins by `nfev` alone, a few percent once Jacobian evaluations are counted. Not worth a footgun. Note the basis route is only cheaper as *reuse* — standing alone it is `n` convolutions against `_convolved_decay`'s one, so `FitResult.generate` should keep calling `_convolved_decay` directly. The remaining duplication between the two is the two-line normalization `T = ΣN − bg·M`, `T·dt·decay + bg`, which is small enough to leave.
- **Draws outside the physical domain are the user's problem, not the library's.** Nothing detects a parameter sitting on a bound at the optimum, and `sample` does not reject nonlinear draws that violate the bounds — so with the default `background=FitParameter(0.0, lower=0.0)` a background-free spectrum converges onto the bound and `sample` will emit negative backgrounds. This is accepted rather than guarded. `pcov` is a Gaussian (Laplace) approximation of the posterior around the optimum; such an approximation can always place mass outside the physical domain, and a bound is only the most visible way it shows. Removing that properly needs a full Bayesian treatment, which is out of scope. The audience is scientists, who are expected to know that a variance estimate can leave the domain of its parameter. Consistent with *The fitter fits* below. **Intensities are the deliberate exception**: `pcov` is returned and the user can inspect it, whereas `cov(I|τ)` is built and consumed inside `_draw_intensities` and never surfaces, and the NNLS enforces `I ≥ 0` structurally — so a negative draw contradicts the estimator that produced its centre. The library guards what it does not expose; the user handles what they can see. The reason is in `sample`'s docstring.
- **Component order is the user's responsibility, and that is why components carry no name.** The likelihood is invariant under permuting components (label switching), so nothing in the model distinguishes slot 0 from slot 1 and the optimizer can land either way depending on where it started. The fix is non-overlapping τ bounds, which the user must set anyway the moment they use constraints, since those reference components by index. Naming components was rejected for exactly this reason: a name would promise an identity the fit does not guarantee, while the user has to notice the switching regardless. `tests/analysis/lifetime/fit/test_fitter.py` sorts before asserting for the same reason. Consistent with *The fitter fits* below.
- **The fitter fits.** Goodness-of-fit, correlation matrices and conditioning diagnostics stay out of `LifetimeFitter` — the user derives them from the result, following the `analysis/vedb/` pattern of separate modules.
- **`pinv` in the intensity covariance is for duplicate lifetimes, not for the pinned sum.** `Σ I` being tightly determined makes a *large* eigenvalue of `DᵀD`, which is harmless. Measured `cond(DᵀD)`: 2.3 at τ=[0.2,1.5], 7.0 at [0.2,0.45], 482 at [0.2,0.22], 1.8e7 at [0.2,0.2001], exactly singular at [0.2,0.2] where `inv` raises. `pinv` exists so an unlucky draw inside `sample` cannot kill the whole loop — at the cost of absorbing a genuinely degenerate model silently.

### Verified numerically

Each of these was checked against a reference that shares no machinery with the code under test. None of them live in the suite; they were one-off scripts.

- **`convolve`** against a brute-force transcription of the definition, `out[m] = Σ_j sig[j]·IRF(time[m] − time[j] − t0)·dt`: worst relative error 5e-16 over 63 combinations of 7 grids, 3 IRF types and 3 t0 values. Caveat: it shares `evaluate`, so it validates the convolution bookkeeping, not `evaluate` itself; and it is not a reference for *truncation*, since `brute` slides its lag range with `m` while `convolve` uses one fixed lag array.
- **`_convolved_decay` normalization**: `|integral − 1| < 1.1e-13` over 64 combinations of `dt` (0.05→0.001), grid offset and τ (0.5→0.02, i.e. `dt/τ = 0.5`). The only residual on a truncated window is the analytic tail `Σ I_i exp(-T/τ_i)`.
- **`estimate_cov`** against 600 independent synthetic fits: `est/MC` = 1.008 / 1.022 / 1.005 / 1.020 against a ±0.029 band (the standard error of an MC standard deviation at n=600 is `1/√(2(n−1))`), Mahalanobis `d²` mean 3.889 ± 0.115 against 4, and every element of the 4×4 covariance inside its bootstrap 95% CI. The VP Jacobian `least_squares` returns and the full joint Jacobian in `[τ, t0, bg, I]` agree to 5 digits, i.e. `(JᵀJ)⁻¹` is already the **marginal** covariance with the intensities profiled out (Golub–Pereyra), needing no correction term. No `χ²_red` rescaling belongs in it — the residuals carry the known Poisson scale. The inverse is now taken from the SVD of `J` rather than by forming `JᵀJ`, whose condition number is the square; the two routes agree to 3.3e-14 at the measured `cond(J) = 1285`, so this is protection for the near-degenerate-τ regime rather than a fix. It costs 149 µs against a 0.165 s fit — the `JᵀJ` route is genuinely ~4× faster (a single GEMM plus a 4×4 solve, against a QR and bidiagonalization of 1700×4), but `estimate_cov` runs once per fit.
- **`sample`** against 400 independent fits: intensity σ 0.008673 (MC) vs 0.008451 (sampler), all six parameters within 2σ, and the correlation structure reproduced.
- **Intensity anticorrelation is not universal.** `σ(ΣI) ≈ 1e-3` regardless of configuration — that part is structural, from `T` being pinned. But `ρ(I_0,I_1) = −(σ_0² + σ_1² − σ_sum²)/(2σ_0σ_1)` reaches −1 only when the lifetimes are close enough that `σ_i ≫ σ_sum`: −0.99 at τ=[0.2,0.45], −0.64 at τ=[0.2,1.5].

### GP inversion

- **Hyperparameters are system-dependent, not measurement-dependent.** ℓ and amplitude characterize the setup, not a given spectrum. The module should support inverting with *given* hyperparameters (adjusted only for counting statistics), in addition to finding them from the data.
- **Real physical bounds on ℓ and amplitude** instead of the hardcoded candidate grid (`ell_grid`/`amp_grid`) — optimize continuously within physical limits rather than iterating over fixed values.

---

## Notes

- Do not read example data files (`.txt`, `.nc`, `.csv` in `examples/`).
- Worked lifetime usage lives in `examples/lifetime/`: `generate_lifetime_spectrum`, `discrete_fitting`, `lifetime_inversion` (Tikhonov + MELT), `gp_inversion`, `load lifetime spectrum`. The lifetime module has no README — this file is its documentation, and user-facing prose belongs in the top-level `README.md`.
- Data files in `scipas/libs/` are package data loaded via `importlib.resources`. They are included in the distribution via `[tool.setuptools.package-data]` in `pyproject.toml`. The `__init__.py` files in `libs/` subdirectories are required for `importlib.resources.files()` to treat them as packages.
- `tests/transport/diffusion/scipy_positron_profile_solver.py` is a reference/validation implementation, not a test file. It is used by `test_fd_vs_scipy.py`.
- `tests/transport/diffusion/test_fd_analytical.py` tests the FD solver against a closed-form analytical solution (< 0.01% error threshold).
