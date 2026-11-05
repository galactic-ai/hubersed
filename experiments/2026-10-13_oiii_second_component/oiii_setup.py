"""Models, start points and line windows of the [O III] second-component MAP fits.

The data, observation, Cue source and SFH hyperparameters are those of the rmatrix mask run of
experiments/2026-10-13_nautilus_resolution_matrix/map.py (imported from there). Only the line
profile parameters differ. ``add_broad_params`` is called again on that run's parameters with
the split forbidden widths, a free low-ionization second-component fraction and the new
high-ionization fraction ``eline_fbroad_forb_hi``. The second call overwrites every key the first
one added, so the only new keys are ``eline_sigma_forb_hi`` and ``eline_fbroad_forb_hi``.

- F0 has ``eline_fbroad_forb_hi`` fixed at 0. [O III], [Ne III], [Ar IV], [Ne IV] and He II are one
  Gaussian of width ``eline_sigma_forb_hi``.
- F1 has ``eline_fbroad_forb_hi`` free. The high-ionization second component shares the width
  ``eline_sigma_broad`` and shift ``eline_vbroad`` of the other lines' second component.
"""

import numpy as np
from map import HYPERS, MASKS, START
from map import OUT as RM_OUT
from model_setup import get_cue, load_data, make_model, make_obs, make_params

from hubersed.paths import PATHS
from hubersed.sps import broadline

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-10-13_oiii_second_component"
SHARED = RM_OUT / "map_rmatrix_mask.npz"
SPLIT = 45.0
FITS = {"F0": 0.0, "F1": "free"}
# EmFit DR1 [O III] 5007 single-Gaussian sigma of this target, in km/s, from
# knowledge/summary_emfit_oiii_treatment.md (UberSED notes). Used only as a start value.
EMFIT_OIII_SIGMA = 17.3
LINE_PROFILE = (
    "eline_sigma",
    "eline_fbroad",
    "eline_sigma_broad",
    "eline_vbroad",
    "eline_sigma_forb",
    "eline_fbroad_forb",
    "eline_sigma_forb_hi",
)
GAS = ("gas_logz", "gas_logu", "gas_lognH", "gas_logno", "gas_logco")
# rest-frame vacuum windows in Angstrom, grouped by the line group of the main line
WINDOWS = {
    "[NeIII]3869": (3862, 3878),
    "[OIII]4363": (4357, 4372),
    "[OIII]4959": (4950, 4970),
    "[OIII]5007": (4995, 5022),
    "[OII]3726,9": (3718, 3740),
    "[OI]6300": (6292, 6310),
    "[NII]6548": (6540, 6556),
    "[NII]6584": (6576, 6592),
    "[SII]6716": (6708, 6724),
    "[SII]6731": (6724, 6740),
    "Hgamma": (4332, 4350),
    "Hbeta": (4845, 4880),
    "Halpha": (6556, 6572),
}
HIGH_WINDOWS = ("[NeIII]3869", "[OIII]4363", "[OIII]4959", "[OIII]5007")
CUE_LINES = {
    "[NeIII]3869": "[Ne III] 3869",
    "[OII]3726": "[O II] 3726",
    "[OII]3729": "[O II] 3729",
    "Hbeta": "Ba-beta 4861",
    "[OIII]4959": "[O III] 4959",
    "[OIII]5007": "[O III] 5007",
    "Halpha": "Ba-alpha 6563",
    "[NII]6584": "[N II] 6584",
    "[SII]6716": "[S II] 6716",
    "[SII]6731": "[S II] 6731",
}
MAGGIE_HZ_TO_1E17 = 3631e-23 / 1e-17


def max_l_hypers():
    """Return the SFH hyperparameters of the 10-09 ER max-L point, as map.py fixes them."""
    d = np.load(START, allow_pickle=True)
    best = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    return {k: float(best[k]) for k in HYPERS}


def make_fit_params(z, fit):
    """Return the parameter specification of fit F0 or F1."""
    params = make_params(z)
    before = set(params)
    params = broadline.add_broad_params(
        params,
        separate_forbidden_width=True,
        forbidden_broad="free",
        sigma_split=SPLIT,
        split_forbidden=True,
        forbidden_hi_broad=FITS[fit],
    )
    assert set(params) - before == {"eline_sigma_forb_hi", "eline_fbroad_forb_hi"}
    assert params["eline_fbroad_forb"]["isfree"]
    assert "depends_on" not in params["eline_fbroad_forb"]
    for k, v in max_l_hypers().items():
        params[k]["isfree"] = False
        params[k]["init"] = v
    return params


def build(fit):
    """Return the model, observation list and Cue source of ``fit``.

    ``fit`` is F0, F1 or shared. The shared model is the rmatrix mask model of map.py.
    """
    z, flux, unc, good = load_data(MASKS["mask"])
    if fit == "shared":
        params = make_params(z)
        for k, v in max_l_hypers().items():
            params[k]["isfree"] = False
            params[k]["init"] = v
    else:
        params = make_fit_params(z, fit)
    return make_model(params, "rmatrix"), make_obs(flux, unc, good, "rmatrix"), get_cue()


def shared_map():
    """Return the shared rmatrix mask MAP as a dict of parameter values, and its -ln p."""
    s = np.load(SHARED)
    return dict(zip([str(x) for x in s["labels"]], s["theta"], strict=True)), float(s["fun"])


def moment_width(p):
    """Return the sigma of one Gaussian with the variance of the shared non-Balmer profile.

    The shared profile is (1 - f) G(sigma_forb) + f G(sigma_broad, shifted by v_broad), all
    intrinsic, with f = eline_fbroad.
    """
    f, s_n = p["eline_fbroad"], p["eline_sigma_forb"]
    s_b, v = p["eline_sigma_broad"], p["eline_vbroad"]
    return float(np.sqrt((1 - f) * s_n**2 + f * s_b**2 + f * (1 - f) * v**2))


def starts(fit, model):
    """Return the named start points of ``fit`` as parameter vectors of ``model``.

    Every start is the shared MAP mapped onto the new parameters, with ``eline_fbroad_forb``
    equal to ``eline_fbroad`` and ``eline_sigma_forb_hi`` equal to ``eline_sigma_forb``, so the
    low-ionization lines start with the shared profile. Then

    - F0 A: as mapped.
    - F0 B: ``eline_sigma_forb_hi`` set to ``moment_width``, the single Gaussian with the variance
      of the shared [O III] profile, capped below the prior edge.
    - F1 A0: high-ionization fraction 0.
    - F1 A1: high-ionization fraction equal to ``eline_fbroad``. This is the shared MAP itself.
    - F1 B: high-ionization fraction 0.35 with ``eline_sigma_forb_hi`` at EmFit's 17.3 km/s, a
      narrow core with a large second component, on the other side of the width and fraction
      degeneracy from F0 B.
    """
    p, _ = shared_map()
    p = dict(p, eline_fbroad_forb=p["eline_fbroad"], eline_sigma_forb_hi=p["eline_sigma_forb"])
    labels = model.theta_labels()
    hi_max = model.config_dict["eline_sigma_forb_hi"]["prior"].params["maxi"]

    def vec(**kw):
        q = dict(p, **kw)
        return np.array([q[k] for k in labels], dtype=float)

    if fit == "F0":
        assert "eline_fbroad_forb_hi" not in labels
        return {"A": vec(), "B": vec(eline_sigma_forb_hi=min(moment_width(p), hi_max - 0.5))}
    return {
        "A0": vec(eline_fbroad_forb_hi=0.0),
        "A1": vec(eline_fbroad_forb_hi=p["eline_fbroad"]),
        "B": vec(eline_fbroad_forb_hi=0.35, eline_sigma_forb_hi=EMFIT_OIII_SIGMA),
    }


def chi2_parts(model, obs, cue, theta):
    """Return the spectrum, the chi2 in total and per window, and the fitted pixel counts.

    Returns
    -------
    spec : np.ndarray
        Model spectrum at ``theta``.
    parts : dict
        chi2 over all fitted pixels ("all"), in each of ``WINDOWS``, and outside every window.
    npix : dict
        Fitted and total pixels of each window.
    """
    spec = model.predict(theta, observations=obs, sps=cue)[0][0]
    o = obs[0]
    rest = o.wavelength / (1 + float(np.atleast_1d(model.params["zred"])[0]))
    r2 = np.where(o.mask, ((o.flux - spec) / o.uncertainty) ** 2, 0.0)
    parts, npix = {"all": float(r2.sum())}, {"all": (int(o.mask.sum()), o.mask.size)}
    inside = np.zeros(rest.size, bool)
    for k, (a, b) in WINDOWS.items():
        w = (rest >= a) & (rest < b)
        inside |= w
        parts[k] = float(r2[w].sum())
        npix[k] = (int((w & o.mask).sum()), int(w.sum()))
    parts["outside windows"] = float(r2[~inside].sum())
    npix["outside windows"] = (int((~inside & o.mask).sum()), int((~inside).sum()))
    return spec, parts, npix


def cue_fluxes(model):
    """Return the Cue line fluxes of the last prediction in 1e-17 erg/s/cm2."""
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    elum = model._eline_lum * model.flux_norm() / (1 + model._zred) * MAGGIE_HZ_TO_1E17
    return {k: float(elum[np.where(names == v)[0][0]]) for k, v in CUE_LINES.items()}


def at_prior_edge(model, theta, tol=0.01):
    """Return the free parameters within ``tol`` of the prior range of a TopHat edge."""
    out = []
    for k in model.free_params:
        prior = model.config_dict[k].get("prior")
        par = getattr(prior, "params", {})
        if (
            "mini" not in par
            or "maxi" not in par
            or model.theta_index[k].stop - model.theta_index[k].start != 1
        ):
            continue
        lo, hi = float(np.min(par["mini"])), float(np.max(par["maxi"]))
        v = float(theta[model.theta_index[k]][0])
        if v - lo < tol * (hi - lo) or hi - v < tol * (hi - lo):
            out.append(f"{k}={v:.4g} in [{lo:.4g}, {hi:.4g}]")
    return out
