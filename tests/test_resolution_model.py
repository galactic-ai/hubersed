"""The prospect model that applies a DESI resolution matrix, on a synthetic stellar population.

A stand-in source gives a C3K_ER-like library (10 km/s log-uniform samples, 12.7 km/s
resolution) and a few strong lines, so the model runs without building FSPS or Cue. Prospect
reads SPS_HOME on import, hence the fsps mark.
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.fsps

C_KMS = 299792.458
LIB_KMS = C_KMS / (2.355 * 10000.0)
ZRED = 0.025
LINES = {
    "[O II] 3726": 0.6,
    "[O II] 3729": 0.8,
    "Ba-beta 4861": 1.0,
    "[O III] 5007": 2.0,
    "Ba-alpha 6563": 3.0,
    "[N II] 6584": 1.0,
    "[S II] 6716": 0.5,
}


class StandInSource:
    """A source with a fixed library spectrum and fixed line luminosities.

    The spectrum is a smooth f_nu continuum with absorption lines of 15 km/s on a grid of 10 km/s
    steps, uniform in log wavelength at rest 3000-11000 A, at the library resolution LIB_KMS.
    """

    def __init__(self, emline_info, line_scale=1.0):
        self.wave = np.exp(np.arange(np.log(3000.0), np.log(11000.0), 10.0 / C_KMS))
        v = np.linspace(0, 1, 40)
        centres = 3300.0 * (10000.0 / 3300.0) ** v
        rng = np.random.default_rng(3)
        depth = np.zeros_like(self.wave)
        for c, d in zip(centres, rng.uniform(0.1, 0.5, centres.size), strict=True):
            depth += d * np.exp(-0.5 * (C_KMS * np.log(self.wave / c) / 15.0) ** 2)
        self.spec = 1e-14 * (self.wave / 5000.0) ** 1.5 * (1 - np.clip(depth, 0, 0.9))
        self.spectral_resolution = np.full(self.wave.size, LIB_KMS)
        names = np.char.strip(np.asarray(emline_info["name"]).astype(str))
        self.ewave = np.asarray(emline_info["wave"], dtype=float)
        self.elum = np.zeros(self.ewave.size)
        for name, lum in LINES.items():
            self.elum[names == name] = line_scale * lum * 1e-3

    def get_galaxy_spectrum(self, **params):
        """Return the library wavelengths, spectrum and surviving mass fraction."""
        return self.wave, self.spec.copy(), 0.6

    def get_galaxy_elines(self):
        """Return the line wavelengths and luminosities."""
        return self.ewave, self.elum.copy()


def model_params(fbroad=0.2):
    """Return a small parameter set for TwoCompLineModel with the broad and forbidden widths."""
    from prospect.models import priors

    from hubersed.sps import broadline

    params = {
        "zred": dict(N=1, isfree=False, init=ZRED),
        "mass": dict(N=1, isfree=False, init=1e9),
        "sigma_smooth": dict(
            N=1, isfree=True, init=60.0, prior=priors.TopHat(mini=10.0, maxi=400.0)
        ),
        "nebemlineinspec": dict(N=1, isfree=False, init=False),
        "add_neb_emission": dict(N=1, isfree=False, init=True),
        "eline_sigma": dict(
            N=1, isfree=True, init=20.0, prior=priors.TopHat(mini=10.0, maxi=250.0)
        ),
    }
    params = broadline.add_broad_params(params, forbidden_broad="shared", sigma_split=45.0)
    params["eline_fbroad"]["init"] = fbroad
    params["eline_sigma_forb"]["init"] = 30.0
    params["eline_sigma_broad"]["init"] = 200.0
    params["eline_vbroad"]["init"] = -40.0
    return params


def gaussian_banded(wave, sigma_kms, ndiag=11):
    """Return a banded R whose rows are Gaussians of ``sigma_kms`` sampled at the pixels."""
    from hubersed.sps.resolution_matrix import banded_offsets

    n = wave.size
    c = ndiag // 2
    dpix = wave[1] - wave[0]
    d = np.arange(-c, c + 1)
    s_pix = sigma_kms * wave / C_KMS / dpix
    rows = np.exp(-0.5 * (d[None, :] / s_pix[:, None]) ** 2)
    rows /= rows.sum(axis=1, keepdims=True)
    data = np.zeros((ndiag, n))
    for k, off in enumerate(banded_offsets(ndiag)):
        i = np.arange(n) - off
        ok = (i >= 0) & (i < n)
        data[k, ok] = rows[i[ok], off + c]
    return data


@pytest.fixture(scope="module")
def setup():
    """Return the parameters, a source, the DESI grid in float32 and the median LSF."""
    from hubersed.sps import broadline
    from hubersed.sps.lsf import desi_resolution
    from hubersed.sps.parameter_file import WAVE_OBS

    params = model_params()
    src = StandInSource(broadline.TwoCompLineModel(params).emline_info)
    lsf = (C_KMS / (2.355 * desi_resolution(WAVE_OBS))).astype(np.float64)
    return params, src, WAVE_OBS, lsf


def predict(model, obs, src):
    """Return the model spectrum for one observation at the model's initial parameters."""
    return model.predict(model.theta.copy(), observations=[obs], sps=src)[0][0]


def test_rejects_resolution_and_bad_grids(setup):
    """The observation refuses a Gaussian resolution, a mismatched grid and prospect's smoothing."""
    from hubersed.sps.resolution_model import ResolutionMatrixSpectrum

    params, src, wave, lsf = setup
    data = gaussian_banded(wave.astype(float), lsf)
    flux = np.ones(wave.size)
    kw = dict(wavelength=wave, resolution_matrix=data, flux=flux, uncertainty=flux)
    with pytest.raises(ValueError, match="Gaussian resolution"):
        ResolutionMatrixSpectrum(resolution=lsf, **kw)
    with pytest.raises(ValueError, match="does not match"):
        ResolutionMatrixSpectrum(resolution_wave=np.linspace(3600.1, 9824.1, wave.size), **kw)
    with pytest.raises(ValueError, match="columns"):
        ResolutionMatrixSpectrum(**{**kw, "resolution_matrix": data[:, :-1]})
    obs = ResolutionMatrixSpectrum(**kw)
    with pytest.raises(NotImplementedError):
        obs.instrumental_smoothing(wave, flux)


def test_pickles_without_the_cached_matrix(setup):
    """The observation pickles, leaves R' out, and rebuilds the same R' after unpickling."""
    from hubersed.sps.resolution_model import build_resolution_obs

    params, src, wave, lsf = setup
    flux = np.ones(wave.size)
    (obs,) = build_resolution_obs(flux, flux, np.ones(wave.size, bool), gaussian_banded(wave, lsf))
    rp = obs.resampled()[0]
    blob = pickle.dumps(obs)
    assert len(blob) < 2e6
    back = pickle.loads(blob)
    assert back._rprime is None
    assert (back.resampled()[0] != rp).nnz == 0


def test_lines_masked_at_centre_still_reach_fitted_pixels(setup):
    """A line whose centre and core are masked is still added and lights its fitted neighbours."""
    from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs

    params, src, wave, lsf = setup
    w64 = wave.astype(float)
    mu = 5008.3137 * (1 + ZRED)
    mask = np.abs(w64 - mu) > 1.3  # masks the three central pixels
    flux = np.ones(wave.size)
    (obs,) = build_resolution_obs(flux, flux, mask, gaussian_banded(w64, lsf))
    model = ResolutionMatrixModel(params)
    predict(model, obs, src)
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    assert model._valid_eline[names == "[O III] 5007"].all()
    near = mask & (np.abs(w64 - mu) < 4.0)
    assert near.any() and np.all(model._resolved_lines[near] > 0)


def test_zero_row_at_fitted_pixel_is_refused(setup):
    """A fitted pixel whose row of R is zero raises, so it cannot get a zero model silently."""
    from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs

    params, src, wave, lsf = setup
    data = gaussian_banded(wave.astype(float), lsf)
    j = 2000
    off = 11 // 2 - np.arange(11)
    data[np.arange(11), j + off] = 0.0  # row j
    flux = np.ones(wave.size)
    (obs,) = build_resolution_obs(flux, flux, np.ones(wave.size, bool), data)
    with pytest.raises(ValueError, match="row sums"):
        predict(ResolutionMatrixModel(params), obs, src)


def test_library_resolution_must_be_constant(setup):
    """Fitted pixels that reach a part of the library with another resolution are refused."""
    from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs

    params, src, wave, lsf = setup
    low = StandInSource.__new__(StandInSource)
    low.__dict__.update(src.__dict__)
    low.spectral_resolution = np.where(src.wave < 3550.0, 636.6, LIB_KMS)
    flux = np.ones(wave.size)
    (obs,) = build_resolution_obs(flux, flux, np.ones(wave.size, bool), gaussian_banded(wave, lsf))
    with pytest.raises(ValueError, match="library resolution varies"):
        predict(ResolutionMatrixModel(params), obs, low)
    rest = wave / (1 + ZRED)
    (obs,) = build_resolution_obs(flux, flux, rest > 3560.0, gaussian_banded(wave, lsf))
    assert np.all(np.isfinite(predict(ResolutionMatrixModel(params), obs, low)))


def test_intrinsic_widths_and_no_instrument_in_lines(setup):
    """Lines are built at the TwoCompLineModel widths with no instrumental or library part."""
    from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs

    params, src, wave, lsf = setup
    flux = np.ones(wave.size)
    (obs,) = build_resolution_obs(flux, flux, np.ones(wave.size, bool), gaussian_banded(wave, lsf))
    model = ResolutionMatrixModel(params)
    predict(model, obs, src)
    prof, wide, shift, inst2 = model._profile_widths()
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    assert np.all(inst2 == 0)
    assert prof[names == "Ba-alpha 6563"][0] == 20.0
    assert prof[names == "[O III] 5007"][0] == 30.0
    assert prof[names == "[N II] 6584"][0] == 30.0
    # the shared forbidden broad fraction gives the forbidden lines the broad component too
    assert np.allclose(model._broad_fraction()[np.isin(names, list(LINES))], 0.2)
    # the fine-grid line spectrum integrates to the Cue line fluxes
    fine = obs.resampled()[1]
    lines, added = model.fine_lines(fine)
    step = (fine[-1] - fine[0]) / (fine.size - 1)
    total = np.sum(lines * 2.998e18 / fine**2) * step
    elum = model._eline_lum * model.flux_norm() / (1 + model._zred)
    assert np.isclose(total, elum[added].sum(), rtol=1e-9)


def test_line_widths_with_last_bit_rounding(setup):
    """An eline_sigma whose square rounds differently in numpy and Python still builds lines.

    fine_lines sets every width to eline_sigma, so the instrumental part sig0**2 - eline_sigma**2
    is zero up to rounding. For this value it is 5.7e-14 on the Mac, which failed an exact check.
    """
    from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs

    params, src, wave, lsf = setup
    params = dict(params, eline_sigma=dict(params["eline_sigma"], init=21.06017450160843))
    flux = np.ones(wave.size)
    (obs,) = build_resolution_obs(flux, flux, np.ones(wave.size, bool), gaussian_banded(wave, lsf))
    model = ResolutionMatrixModel(params)
    assert np.all(np.isfinite(predict(model, obs, src)))
    assert np.all(model._profile_widths()[3] < 1e-9)


def test_quadrature_rule_for_the_continuum(setup):
    """Below the library resolution the continuum is not smoothed; above it the kernel adds."""
    from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs

    params, src, wave, lsf = setup
    flux = np.ones(wave.size)
    (obs,) = build_resolution_obs(flux, flux, np.ones(wave.size, bool), gaussian_banded(wave, lsf))
    fine = obs.resampled()[1]
    out = {}
    for sig in (8.0, LIB_KMS, 60.0):
        model = ResolutionMatrixModel(params)
        theta = model.theta.copy()
        theta[model.theta_index["sigma_smooth"]] = sig
        model.predict(theta, observations=[obs], sps=src)
        out[sig] = model.fine_continuum(fine, obs.reach())
    # sigma_smooth at or below lib leaves the library as it is
    assert np.array_equal(out[8.0], out[LIB_KMS])
    assert not np.allclose(out[60.0], out[8.0], rtol=1e-3)
    # in the observed frame and in maggies, and smoothing keeps the flux in 50 A bins
    raw = np.interp(fine, src.wave * (1 + ZRED), src.spec * model.flux_norm())
    sel = (fine > 4000) & (fine < 9000)
    nbin = 1000  # 50 A of 0.05 A steps
    n = sel.sum() // nbin * nbin

    def binned(x):
        return x[sel][:n].reshape(-1, nbin).mean(axis=1)

    for sig in (8.0, 60.0):
        assert np.allclose(binned(out[sig]), binned(raw), rtol=2e-3)


def test_gaussian_r_reproduces_prospect_smoothing(setup):
    """With R built from the median Gaussian LSF the model is close to prospect's own.

    The two differ in known ways. Prospect samples each line at the pixel centres and rescales
    it to unit flux with 3e18 A/s, which lowers its line fluxes by 0.07% against the 2.998e18 of
    its own profile. R' integrates each line against a spline through the R row, which samples a
    0.6-0.9 pixel Gaussian, so line profiles differ by a few percent of their peak between
    pixels. Prospect resamples the continuum onto a coarse grid before smoothing, which shifts
    the absorption-line depths by a few 0.1%. The total width is the same in both,
    sqrt(sigma_smooth^2 + inst^2), since sigma_smooth is above the library resolution.
    """
    from hubersed.sps import broadline
    from hubersed.sps.parameter_file import build_obs
    from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs

    params, src, wave, lsf = setup
    flux = np.ones(wave.size)
    good = np.ones(wave.size, bool)
    (obs_p,) = build_obs(flux, flux, good, resolution=lsf, wavelength=wave)
    (obs_r,) = build_resolution_obs(flux, flux, good, gaussian_banded(wave.astype(float), lsf))
    pros = broadline.TwoCompLineModel(params)
    hook = ResolutionMatrixModel(params)
    sp, sr = predict(pros, obs_p, src), predict(hook, obs_r, src)
    w = wave.astype(float)
    inner = (w > 3650) & (w < 9750)

    assert np.array_equal(sr, hook._resolved_continuum + hook._resolved_lines)
    cont_p = sp.copy()
    cont_p[pros._fix_eline_pixelmask] -= pros._fix_eline_spec.sum(axis=1)
    cont_r = hook._resolved_continuum
    rel = np.abs(cont_r / cont_p - 1)[inner]
    print(f"continuum: median {np.median(rel):.2e}, 99% {np.percentile(rel, 99):.2e}")
    assert np.median(rel) < 1e-3 and np.percentile(rel, 99) < 1e-2

    lines_p, lines_r = sp - cont_p, hook._resolved_lines
    for name in LINES:
        mu = model_line(pros, name) * (1 + ZRED)
        win = np.abs(w - mu) < 15.0
        ratio = lines_r[win].sum() / lines_p[win].sum()
        peak = np.max(np.abs(lines_r[win] - lines_p[win])) / lines_p[win].max()
        print(f"{name}: flux ratio R'/prospect {ratio:.5f}, max profile difference {peak:.3f}")
        assert abs(ratio - 1) < 3e-3
        assert peak < 0.08


def model_line(model, name):
    """Return the rest wavelength of a named line of a model's line list."""
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    return float(np.asarray(model.emline_info["wave"])[names == name][0])


def hi_params(hi=None, fb=0.2, fb_forb=0.1):
    """Return parameters with the split widths, a free non-Balmer fraction and ``hi``.

    ``hi`` is passed to ``add_broad_params`` as ``forbidden_hi_broad``.
    """
    from hubersed.sps import broadline

    base = {k: v for k, v in model_params().items() if not k.startswith("eline_")}
    base["eline_sigma"] = model_params()["eline_sigma"]
    p = broadline.add_broad_params(
        base, forbidden_broad="free", sigma_split=45.0, split_forbidden=True, forbidden_hi_broad=hi
    )
    p["eline_fbroad"]["init"] = fb
    p["eline_fbroad_forb"]["init"] = fb_forb
    p["eline_sigma_forb"]["init"] = 30.0
    p["eline_sigma_forb_hi"]["init"] = 18.0
    p["eline_sigma_broad"]["init"] = 200.0
    p["eline_vbroad"]["init"] = -40.0
    return p


def both_obs(wave, lsf):
    """Return a Gaussian-LSF observation and a resolution-matrix observation of flat data."""
    from hubersed.sps.parameter_file import build_obs
    from hubersed.sps.resolution_model import build_resolution_obs

    flux = np.ones(wave.size)
    good = np.ones(wave.size, bool)
    (obs_p,) = build_obs(flux, flux, good, resolution=lsf, wavelength=wave)
    (obs_r,) = build_resolution_obs(flux, flux, good, gaussian_banded(wave.astype(float), lsf))
    return obs_p, obs_r


def both_spectra(params, src, obs_p, obs_r):
    """Return the TwoCompLineModel and ResolutionMatrixModel spectra of ``params``."""
    from hubersed.sps import broadline
    from hubersed.sps.resolution_model import ResolutionMatrixModel

    return (
        predict(broadline.TwoCompLineModel(params), obs_p, src),
        predict(ResolutionMatrixModel(params), obs_r, src),
    )


def test_absent_high_ion_fraction_changes_nothing(setup):
    """Without eline_fbroad_forb_hi the high-ionization lines use eline_fbroad_forb, as before.

    Setting eline_fbroad_forb_hi equal to eline_fbroad_forb gives the same spectra to the last bit,
    in both models.
    """
    from hubersed.sps import broadline

    _, src, wave, lsf = setup
    absent, equal = hi_params(None), hi_params(0.1)
    assert "eline_fbroad_forb_hi" not in absent and not equal["eline_fbroad_forb_hi"]["isfree"]
    model = broadline.TwoCompLineModel(absent)
    expected = np.where(model._balmer_mask(), 0.2, 0.1)
    assert np.array_equal(model._broad_fraction(), expected)
    obs_p, obs_r = both_obs(wave, lsf)
    for a, b in zip(
        both_spectra(absent, src, obs_p, obs_r), both_spectra(equal, src, obs_p, obs_r), strict=True
    ):
        assert np.array_equal(a, b)


def test_high_ion_fraction_only_on_high_ion_lines(setup):
    """eline_fbroad_forb_hi sets the high-ionization lines only, eline_fbroad_forb the others.

    The stand-in source has one high-ionization line, [O III] 5007. Changing the high-ionization
    fraction changes both models only near it, and changing the low-ionization fraction changes
    them only away from it.
    """
    from hubersed.sps import broadline

    _, src, wave, lsf = setup
    model = broadline.TwoCompLineModel(hi_params(0.4))
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    balmer, high = broadline.line_groups(names)
    fb = model._broad_fraction()
    assert np.all(fb[balmer] == 0.2) and np.all(fb[high] == 0.4)
    assert np.all(fb[~balmer & ~high] == 0.1)
    assert high[names == "[O III] 5007"][0] and not high[names == "[N II] 6584"].any()

    obs_p, obs_r = both_obs(wave, lsf)
    ref = both_spectra(hi_params(0.0), src, obs_p, obs_r)
    hi_on = both_spectra(hi_params(0.4), src, obs_p, obs_r)
    lo_on = both_spectra(hi_params(0.0, fb_forb=0.4), src, obs_p, obs_r)
    near = np.abs(wave / (1 + ZRED) - model_line(model, "[O III] 5007")) < 30.0
    for r, h, lo in zip(ref, hi_on, lo_on, strict=True):
        assert np.any(h[near] != r[near]) and np.array_equal(h[~near], r[~near])
        assert np.any(lo[~near] != r[~near]) and np.array_equal(lo[near], r[near])


def test_high_ion_fraction_conserves_line_flux(setup):
    """The high-ionization second component moves [O III] flux without adding or losing any."""
    from hubersed.sps.resolution_model import ResolutionMatrixModel

    _, src, wave, lsf = setup
    obs_p, obs_r = both_obs(wave, lsf)
    w = wave.astype(float)
    ref_model = ResolutionMatrixModel(hi_params(0.0))
    predict(ref_model, obs_r, src)
    near = np.abs(w / (1 + ZRED) - model_line(ref_model, "[O III] 5007")) < 30.0
    to_flam = 2.998e18 / w**2  # maggies to flux per Angstrom, up to a constant
    line = np.sum(ref_model._resolved_lines[near] * to_flam[near])
    ref = both_spectra(hi_params(0.0), src, obs_p, obs_r)
    for f in (0.3, 0.9):
        spec = both_spectra(hi_params(f), src, obs_p, obs_r)
        for s, r in zip(spec, ref, strict=True):
            moved = np.sum((s - r)[near] * to_flam[near])
            assert abs(moved) < 1e-3 * line
        model = ResolutionMatrixModel(hi_params(f))
        predict(model, obs_r, src)
        fine = obs_r.resampled()[1]
        lines, added = model.fine_lines(fine)
        step = (fine[-1] - fine[0]) / (fine.size - 1)
        total = np.sum(lines * 2.998e18 / fine**2) * step
        elum = model._eline_lum * model.flux_norm() / (1 + model._zred)
        assert np.isclose(total, elum[added].sum(), rtol=1e-9)


@pytest.mark.parametrize("hi", ["free", 0.3])
def test_self_check_with_high_ion_fraction(setup, hi):
    """self_check's one-component comparison passes with a free or fixed high-ionization fraction.

    The prior draws are skipped, since HyperSpecModel's prior transform needs the SFH
    hyperparameters that the stand-in parameters do not have.
    """
    from hubersed.sps import broadline

    _, src, wave, lsf = setup
    obs_p, _ = both_obs(wave, lsf)
    params = hi_params(hi)
    theta = broadline.TwoCompLineModel(params).theta.copy()
    broadline.self_check(params, obs_p, src, theta, n_prior=0)


TID = 39627770174637084
EXPERIMENT = Path(__file__).parents[1] / "experiments" / "2026-10-09_nautilus_c3k_er_broadline"
HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")


def experiment_module(name):
    """Load a module of the 2026-10-09 experiment folder from its path."""
    path = EXPERIMENT / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{path} is missing")
    spec = importlib.util.spec_from_file_location(f"er_broadline_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:`trapz` is deprecated:DeprecationWarning")  # cuejax
def test_matches_prototype_at_er_max_likelihood():
    """At the ER max-L point of the 10-09 run the model reproduces the prototype ResampledRModel.

    The model is built as map_resolution.py builds its R_rows_ivar run. With the prototype's
    continuum put in, the two agree to 1e-6 once the lines the prototype dropped are taken out.
    The prototype keeps only the lines prospect finds near a fitted pixel, and those dropped
    lines are reported. Then the full model is compared, whose continuum is smoothed on the
    library samples instead of prospect's coarser resampled grid.
    """
    from hubersed.fitting.map_fits import get_sps
    from hubersed.paths import PATHS
    from hubersed.sps.resolution_matrix import load_resolution_npz
    from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs

    proto = experiment_module("resolution_model")
    fit = experiment_module("fit")
    cue = get_sps(zero_library_resolution=False)["cue"]
    if cue.ssp.libraries[1] not in ("c3k_er", b"c3k_er"):
        pytest.skip("needs the C3K_ER build of FSPS first on PYTHONPATH")
    out = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
    d = np.load(out / f"{TID}_er_broad_shared_split45_rest9000_seed0_pred.npz", allow_pickle=True)
    best = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    z, flux, unc, good = fit.load_data(TID, 9000.0)

    def params():
        p = fit.make_params(z, "shared", 45.0)
        for k in HYPERS:
            p[k]["isfree"] = False
            p[k]["init"] = float(best[k])
        return p

    res = load_resolution_npz(out / str(TID) / "resolution_matrix_rows_ivar.npz")
    fine_p = proto.fine_grid(res["wave"], res["res"].shape[0])
    rp_p = proto.resampled_matrix(res["wave"], res["res"], fine_p)
    mp = proto.ResampledRModel(params())
    mp.set_resolution(rp_p, fine_p)
    theta0 = np.array([best[k] for k in mp.theta_labels()])
    sp = mp.predict(theta0, observations=fit.make_obs(flux, unc, good), sps=cue)[0][0]
    kept_p = mp._fix_eline & mp._valid_eline

    (obs,) = build_resolution_obs(flux, unc, good, res["res"], res["wave"])
    rp, fine, _ = obs.resampled()
    assert fine.size == fine_p.size and np.max(np.abs(fine - fine_p)) < 1e-9
    assert abs(rp - rp_p).max() < 1e-12

    class PrototypeContinuum(ResolutionMatrixModel):
        def fine_continuum(self, fine, reach):
            obs_wave = self._wave * (1 + z)
            sel = (obs_wave > fine[0]) & (obs_wave < fine[-1])
            lib = np.median(np.broadcast_to(self._library_resolution, obs_wave.shape)[sel])
            sig = float(np.atleast_1d(self.params["sigma_smooth"])[0])
            cont = proto.continuum(self, np.sqrt(max(sig**2 - lib**2, 0.0)))
            return np.interp(fine, obs_wave, cont)

    mc = PrototypeContinuum(params())
    sc = mc.predict(theta0, observations=[obs], sps=cue)[0][0]
    extra = mc._valid_eline & ~kept_p
    assert not np.any(kept_p & ~mc._valid_eline)
    use = mc._use_eline.copy()
    mc._use_eline = extra
    extra_spec = rp @ mc.fine_lines(fine)[0]
    mc._use_eline = use
    diff = np.max(np.abs(sc - extra_spec - sp)) / np.max(np.abs(sp))
    print(f"with the prototype continuum and lines, max difference {diff:.1e} of the peak")
    assert diff < 1e-6
    names = np.char.strip(np.asarray(mc.emline_info["name"]).astype(str))
    reach = np.abs(extra_spec[good]) / unc[good]
    print(
        f"{extra.sum()} lines the prototype dropped, largest effect on a fitted pixel "
        f"{reach.max():.2f} sigma: {', '.join(names[extra][:12])}"
    )

    mn = ResolutionMatrixModel(params())
    sn = mn.predict(theta0, observations=[obs], sps=cue)[0][0]
    rel = np.abs(sn - sc)[good] / unc[good]
    chi2 = {k: float(np.sum(((flux - s) / unc)[good] ** 2)) for k, s in (("p", sp), ("n", sn))}
    print(
        f"full model against the prototype continuum: median {np.median(rel):.3f} sigma, "
        f"max {rel.max():.3f} sigma; chi2 prototype {chi2['p']:.1f}, new {chi2['n']:.1f}"
    )
    assert np.percentile(np.abs(sn / sc - 1)[good], 99) < 1e-2
