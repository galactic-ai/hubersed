"""Why the 5 PSD hyperparameters must stay frozen in a MAP fit, as a test.

Two claims, both load-bearing for every continuum-only sample fit:

  1. The MAP objective is UNBOUNDED with the hyperparameters free.
     ``hyperparameters.py:53-62`` scores ``logsfr_ratios`` with the NORMALISED
     ``scipy.stats.multivariate_normal`` pdf, and Sigma is linear in ``sigma_reg**2``
     and ``sigma_dyn**2`` (``hyperparam_transforms.py:120-136``). Shrink both sigmas
     and the ratios by the same factor: the Mahalanobis term is exactly invariant
     while ``-0.5*ln|Sigma|`` grows by ``(nbins-1)*ln(10)`` per decade, forever. The
     optimiser therefore walks the sigmas to their LogUniform floors and, in doing so,
     squeezes the SFH prior onto a constant SFH -- the opposite of what a quiescent
     galaxy needs.

  2. ``build_continuum_model`` yields exactly the 15 free parameters the sample run
     assumes, with the hyperparameters already ``isfree=False``.

Test 1 needs only numpy + prospect's transforms. Test 2 needs a model build, so it
skips where FSPS is unavailable.

Run with ``uv run pytest tests/test_stochastic_prior_hypers.py`` (never ``uvx``).
"""

import numpy as np
import pytest

transforms = pytest.importorskip("prospect.models.hyperparam_transforms")

# 10 bins => 9 log SFR ratios, matching make_stochastic_agebins.
AGEBINS = np.log10(
    np.array([[0.001, 0.005], [0.005, 0.01]]
             + [[a, b] for a, b in zip(np.geomspace(0.01, 12.0, 9)[:-1],
                                       np.geomspace(0.01, 12.0, 9)[1:])]) * 1e9)
HYPERS = np.array([0.17, 2.5, 13.0, 0.005, 0.025])   # DEFAULT_SET_VALS ordering


def ratio_covar(psd):
    return transforms.sfr_covar_to_sfr_ratio_covar(
        transforms.get_sfr_covar(psd, agebins=AGEBINS))


def mvn_logpdf(x, S):
    """scipy.stats.multivariate_normal(...).logpdf, spelled out via slogdet."""
    sign, logdet = np.linalg.slogdet(S)
    assert sign > 0, "Sigma is not positive definite"
    return -0.5 * (logdet + len(x) * np.log(2 * np.pi) + x @ np.linalg.solve(S, x))


def test_sigma_scales_the_covariance_quadratically():
    """The premise of the whole argument: Sigma is homogeneous of degree 2 in sigma."""
    p = HYPERS.copy()
    S1 = ratio_covar(p)
    q = p.copy(); q[0] *= 3.0; q[3] *= 3.0
    S3 = ratio_covar(q)
    np.testing.assert_allclose(S3, 9.0 * S1, rtol=1e-10)


def test_map_objective_is_unbounded_in_the_sigma_funnel():
    """Shrinking (sigma, ratios) together buys (nbins-1)*ln(10) nats per decade."""
    rng = np.random.default_rng(0)
    x0 = rng.normal(scale=0.3, size=len(AGEBINS) - 1)

    def lnp(scale):
        p = HYPERS.copy(); p[0] *= scale; p[3] *= scale
        return mvn_logpdf(x0 * scale, ratio_covar(p))

    mahal = []
    for s in (1.0, 1e-1, 1e-2, 1e-3):
        p = HYPERS.copy(); p[0] *= s; p[3] *= s
        S = ratio_covar(p)
        mahal.append((x0 * s) @ np.linalg.solve(S, x0 * s))

    # Mahalanobis term exactly invariant -- nothing pushes back on the shrinkage.
    np.testing.assert_allclose(mahal, mahal[0], rtol=1e-8)

    # ...while the log-determinant term grows without bound, at a rate set only by
    # the dimension. This is the number quoted in fit_one's docstring: 9*ln(10)=20.72.
    expected = (len(AGEBINS) - 1) * np.log(10.0)
    for a, b in zip((1.0, 1e-1, 1e-2), (1e-1, 1e-2, 1e-3)):
        assert np.isclose(lnp(b) - lnp(a), expected, rtol=1e-6)
    assert lnp(1e-5) > lnp(1.0) + 100.0


def test_continuum_model_is_15_free_with_hypers_frozen():
    pytest.importorskip("fsps")
    from hubersed.fitting.config import build_continuum_model

    model, tmpl = build_continuum_model(0.0187021)
    labels = model.theta_labels()
    assert len(labels) == 15, f"expected 15 free, got {len(labels)}: {labels}"

    named = {lab.rsplit("_", 1)[0] if lab.startswith("logsfr_ratios_") else lab
             for lab in labels}
    assert named == {"logzsol", "dust2", "logmass", "logsfr_ratios",
                     "dust_ratio", "dust_index", "sigma_smooth"}
    assert sum(lab.startswith("logsfr_ratios_") for lab in labels) == 9

    for k in ("sigma_reg", "tau_eq", "tau_in", "sigma_dyn", "tau_dyn"):
        assert tmpl[k]["isfree"] is False, f"{k} is free -- the objective is unbounded"
        assert k not in model.theta_index
