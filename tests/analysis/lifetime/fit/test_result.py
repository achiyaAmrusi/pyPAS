import numpy as np
import pytest
from scipas.analysis.lifetime.fit import LifetimeFitter, FitParameter, FitResult
from scipas.core.lifetime import PASLifetime

from conftest import components, reduced_chi_squared


@pytest.fixture
def fitted(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=components(0.15, 1.2),
        t0=FitParameter(0.0, lower=-0.5, upper=0.5),
        background=FitParameter(bg, lower=0.0),
    )
    return pals, result


def test_structure(fitted):
    pals, result = fitted
    assert isinstance(result, FitResult)
    assert result.n_components == 2
    assert list(result.popt.coords["parameter"].values) == ["tau_0", "tau_1", "t0", "background"]
    assert result.popt.shape == (4,)
    assert result.pcov.shape == (4, 4)
    assert result.pcov.dims == ("parameter", "parameter0")
    assert result.free.dtype == bool
    assert np.allclose(result.pcov.values, result.pcov.values.T)
    assert result.pals is pals
    assert isinstance(result.optimizer.success, bool)
    assert result.optimizer.nfev > 0


def test_opt_parameters_layout(fitted):
    _, result = fitted
    taus, intensities, t0, background = result.opt_parameters()
    assert taus.shape == (2,)
    assert intensities.shape == (2,)
    assert np.isscalar(t0) or isinstance(t0, float)
    assert isinstance(background, float)
    # lifetimes and t0/background agree with the stored slots
    assert np.allclose(taus, result.popt.values[:2])
    assert t0 == pytest.approx(float(result.popt.sel(parameter="t0")))
    assert background == pytest.approx(float(result.popt.sel(parameter="background")))


def test_generate_matches_measured_grid(fitted):
    pals, result = fitted
    model = result.generate(*result.opt_parameters())
    assert isinstance(model, PASLifetime)
    assert model.lifetime.counts.shape == pals.lifetime.counts.shape
    assert np.allclose(model.lifetime.axis, pals.lifetime.axis)
    assert reduced_chi_squared(pals, result) < 2.0


def test_generate_requires_matching_lengths(fitted):
    _, result = fitted
    with pytest.raises(ValueError, match="same length"):
        result.generate([0.2, 1.5], [1.0], 0.0, 50.0)


def test_sample_shapes_and_draw_is_generatable(fitted):
    pals, result = fitted
    size = 25
    taus, intensities, t0, background = result.sample(size, rng=1)
    assert taus.shape == (size, 2)
    assert intensities.shape == (size, 2)
    assert t0.shape == (size,)
    assert background.shape == (size,)
    assert np.all(intensities >= 0.0)

    # a draw feeds straight back into generate, no reshaping
    drawn = result.generate(taus[0], intensities[0], t0[0], background[0])
    assert drawn.lifetime.counts.shape == pals.lifetime.counts.shape


def test_sample_is_reproducible(fitted):
    _, result = fitted
    for a, b in zip(result.sample(10, rng=1), result.sample(10, rng=1)):
        assert np.array_equal(a, b)


def test_sample_differs_across_seeds(fitted):
    _, result = fitted
    assert not np.array_equal(result.sample(10, rng=1)[0], result.sample(10, rng=2)[0])


def test_sample_reproduces_pcov(fitted):
    """The nonlinear draws must reproduce pcov, in scale and in structure.

    Two assertions rather than one comparison of covariance matrices: the
    diagonal spans three orders of magnitude and wants a relative tolerance,
    while the off-diagonal correlations sit near zero, where a relative
    tolerance means nothing.
    """
    _, result = fitted
    taus, _, t0, background = result.sample(2000, rng=1)
    drawn = np.column_stack([taus, t0, background])
    scale = np.sqrt(np.diag(result.pcov.values))

    assert np.allclose(drawn.std(axis=0, ddof=1), scale, rtol=0.1)
    assert np.allclose(np.corrcoef(drawn.T),
                       result.pcov.values / np.outer(scale, scale), atol=0.05)


def test_sampled_intensity_sum_is_pinned(fitted):
    """T is pinned to the observed counts, so sum(I) barely varies across draws.

    A property of the scaling, not of the components: it holds whether or not
    the individual intensities are well determined. How strongly I_0 and I_1
    anticorrelate does depend on the components,

        rho = -(s0^2 + s1^2 - s_sum^2) / (2 s0 s1)

    which approaches -1 only when the lifetimes are close enough that the
    individual intensities are degenerate, s_i >> s_sum.

    Both assertions are one-sided bounds, chosen to separate the pinned case
    from the unpinned one rather than to measure anything: sum(I) varies by
    about 1e-3 here, while an unpinned sum would vary on the scale of the
    individual intensities, about 8e-3. The 5e-3 bound sits between the two.
    500 draws are therefore plenty - the standard error of a standard deviation
    is sigma/sqrt(2(n-1)), around 3e-5.
    """
    _, result = fitted
    _, intensities, _, _ = result.sample(500, rng=1)
    total = intensities.sum(axis=1)
    assert abs(total.mean() - 1.0) < 5e-3
    assert total.std(ddof=1) < 5e-3


def test_sample_holds_fixed_parameters_constant(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    result = LifetimeFitter().fit(
        pals,
        lifetime_components=[FitParameter(0.2, fixed=True),
                             FitParameter(1.2)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(bg, lower=0.0),
    )
    taus, _, t0, _ = result.sample(50, rng=1)
    assert np.all(taus[:, 0] == 0.2)
    assert np.all(t0 == 0.0)
