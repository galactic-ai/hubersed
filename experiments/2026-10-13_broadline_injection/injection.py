"""Mock spectra with injected line profiles, and the fit models, for the broad-line injection test.

Scenarios S0 (our model), S1 (EmFit Halpha profile), S2 (per-group broad fractions) and S3 (broad
ratio offsets) at the rmatrix mask MAP truth point. S2 and S3 numbers are from Hogarth+20 tab2,
CLASSY VIII and EmFit for this target.
"""

import csv
from pathlib import Path

import numpy as np
from model_setup import load_data, make_obs
from prospect.models import priors

from hubersed.paths import PATHS
from hubersed.sps import broadline
from hubersed.sps.broadline import C_KMS
from hubersed.sps.resolution_matrix import line_profile
from hubersed.sps.resolution_model import ResolutionMatrixModel

HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
MASK_LINES = (
    "H12",
    "H11",
    "He I 5016",
    "He I 7281",
    "Pa16",
    "Pa15",
    "Pa14",
    "Pa13",
    "Pa12",
    "Pa11",
)
OUT = PATHS["RESULTS"] / "2026-10-13_broadline_injection"
TRUTH_NPZ = PATHS["RESULTS"] / "2026-10-13_nautilus_resolution_matrix" / "map_rmatrix_mask.npz"
POST_NPZ = (
    PATHS["RESULTS"]
    / "2026-10-09_nautilus_c3k_er_broadline"
    / "39627770174637084_er_broad_shared_split45_rest9000_seed0_pred.npz"
)
EMFIT_CSV = (
    Path.home() / "Documents/UberSED/knowledge/_evidence/2026-09-30_kin_vs_ion/emfit_rows_2obj.csv"
)

# EmFit Halpha components of 39627770174637084: (flux fraction, sigma km/s, shift km/s)
EMFIT_HA = {
    "narrow": (1463.625439433989, 10.21799562928374, 6564.572366177524),
    "outflow": (635.9957801084389, 58.90456306926708, 6564.553318876339),
    "broad": (95.36140595676251, 154.9188957634761, 6564.509123638349),
}
S2_FACTOR = {"recombination": 1.0, "low": 1.2, "high": 0.7}
S3_DEX = {"[N II]": 0.136, "[S II]": 0.091, "[O I]": 0.5, "[O III]": -0.210}
RECOMB_PREFIXES = ("Ba-", "Pa-", "He I ")
SCENARIOS = ("S0", "S1", "S2", "S3")
FIT_MODELS = ("shared", "free", "none")
# F-none widths take the prior the Cue model had before add_broad_params narrowed it
NONE_WIDTH_PRIOR = (10.0, 250.0)


def emfit_components():
    """Return EmFit's Halpha components as (weight, sigma, shift) with shifts from the narrow."""
    total = sum(v[0] for v in EMFIT_HA.values())
    mu_n = EMFIT_HA["narrow"][2]
    return [(f / total, s, C_KMS * (mu / mu_n - 1)) for f, s, mu in EMFIT_HA.values()]


def check_emfit(path=EMFIT_CSV):
    """Re-read the EmFit csv row and check it matches ``EMFIT_HA``."""
    if not Path(path).exists():
        return f"EmFit csv not found at {path}; EMFIT_HA not re-checked"
    rows = [r for r in csv.DictReader(open(path)) if r["targetid"] == "39627770174637084"]
    (r,) = rows
    for key, col in (("narrow", "ha_n"), ("outflow", "ha_out"), ("broad", "ha_b")):
        f, s, mu = EMFIT_HA[key]
        assert np.isclose(float(r[col + "_flux"]), f, rtol=1e-12)
        assert np.isclose(float(r[col + "_sigma"]), s, rtol=1e-12)
        assert np.isclose(float(r[col + "_mean"]), mu, rtol=1e-12)
    comps = emfit_components()
    return "EmFit csv matches: " + ", ".join(
        f"w={w:.4f} sigma={s:.2f} v={v:+.3f}" for w, s, v in comps
    )


def line_group(name):
    """Return the S2 group (recombination, high or low) of a Cue line name."""
    name = name.strip()
    if name.startswith(RECOMB_PREFIXES):
        return "recombination"
    if name.startswith(broadline.HIGH_ION_PREFIXES):
        return "high"
    return "low"


def s3_offset(name):
    """Return the S3 broad-component offset in dex of a Cue line name (0 for other ions)."""
    name = name.strip()
    for ion, dex in S3_DEX.items():
        # "[O I]" must not match "[O II]" or "[O III]", so require the closing bracket
        if name.startswith(ion):
            return dex
    return 0.0


class InjectedLineModel(ResolutionMatrixModel):
    """ResolutionMatrixModel whose lines are built from given Gaussian components.

    Parameters
    ----------
    scenario : {"S0", "S1", "S2", "S3"}
        Which line components ``line_components`` returns.
    """

    def __init__(self, params, scenario="S0"):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, not {scenario!r}")
        self.scenario = scenario
        super().__init__(params)

    def line_components(self):
        """Return the components of every line at the current parameters.

        Returns
        -------
        list of list of tuple
            For each line, (weight, intrinsic sigma in km/s, shift in km/s). Weights multiply the Cue flux.
        """
        nline = self._ewave_obs.shape[0]
        names = self._line_names()
        if self.scenario == "S1":
            comps = emfit_components()
            return [list(comps) for _ in range(nline)]
        # our own two-component widths, intrinsic as in ResolutionMatrixModel.fine_lines
        self._eline_sigma_kms = np.full(nline, self._p("eline_sigma", 0.0))
        prof, _, _, inst2 = self._profile_widths()
        assert np.all(inst2 == 0)
        fb = self._broad_fraction()
        s_b = self._p("eline_sigma_broad", 150.0)
        v_b = self._p("eline_vbroad", 0.0)
        f_bal = self._p("eline_fbroad", 0.0)
        out = []
        for i in range(nline):
            if self.scenario == "S2":
                f = f_bal * S2_FACTOR[line_group(names[i])]
                comps = [(1 - f, prof[i], 0.0), (f, s_b, v_b)]
            elif self.scenario == "S3":
                boost = 10 ** s3_offset(names[i])
                comps = [(1 - fb[i], prof[i], 0.0), (fb[i] * boost, s_b, v_b)]
            else:
                comps = [(1 - fb[i], prof[i], 0.0), (fb[i], s_b, v_b)]
            # as in the parent, a broad component with zero weight is not built
            out.append([c for k, c in enumerate(comps) if k == 0 or c[0] > 0])
        return out

    def flux_multipliers(self):
        """Return each line's total flux over its Cue flux, the sum of its component weights."""
        return np.array([sum(c[0] for c in comps) for comps in self.line_components()])

    def fine_lines(self, fine):
        """Return the injected line spectrum on the fine grid, with the parent's line selection."""
        nline = self._ewave_obs.shape[0]
        spec = np.zeros(fine.size)
        added = np.zeros(nline, dtype=bool)
        if not (self._want_lines and self._need_lines):
            return spec, added
        if np.any(self._fit_eline & self._use_eline):
            raise NotImplementedError("line marginalisation is not supported with R'")
        comps = self.line_components()
        elum = self._eline_lum * self.flux_norm() / (1 + self._zred)
        for i in np.where(self._use_eline & self._fix_eline)[0]:
            mu = self._ewave_obs[i]
            for weight, sigma, shift in comps[i]:
                sl, g = line_profile(fine, mu * (1 + shift / C_KMS), sigma)
                if g.size:
                    spec[sl] += weight * elum[i] * g
                    added[i] = True
        return spec, added


def make_fit_params(z, fit_model, hyper_values):
    """Return the parameter specification of one fit model.

    Parameters
    ----------
    fit_model : {"shared", "free", "none"}
        ``shared`` is the 10-13 model, ``free`` adds a free forbidden broad fraction and ``none``
        fixes eline_fbroad at 0 with the TopHat(10, 250) width prior.
    hyper_values : dict
        Values of the SFH hyperparameters ``HYPERS``, fixed in every model.
    """
    from model_setup import make_params

    params = make_params(z)
    if fit_model == "free":
        free = broadline.add_broad_params(
            {"eline_sigma": params["eline_sigma"]}, forbidden_broad="free"
        )
        params["eline_fbroad_forb"] = free["eline_fbroad_forb"]
    elif fit_model == "none":
        params["eline_fbroad"] = dict(params["eline_fbroad"], isfree=False, init=0.0)
        for k in ("eline_sigma_broad", "eline_vbroad"):
            params[k] = dict(params[k], isfree=False)
        for k in ("eline_sigma", "eline_sigma_forb"):
            params[k] = dict(
                params[k], prior=priors.TopHat(mini=NONE_WIDTH_PRIOR[0], maxi=NONE_WIDTH_PRIOR[1])
            )
    elif fit_model != "shared":
        raise ValueError(f"fit_model must be one of {FIT_MODELS}, not {fit_model!r}")
    for k in HYPERS:
        params[k] = dict(params[k], isfree=False, init=float(hyper_values[k]))
    return params


def load_truth():
    """Return the truth point and the 10-09 max-L point as dicts by parameter name."""
    r = np.load(TRUTH_NPZ)
    labels = [str(x) for x in r["labels"]]
    truth = dict(zip(labels, np.asarray(r["theta"], float), strict=True))
    maxl = dict(zip(labels, np.asarray(r["theta0"], float), strict=True))
    return truth, maxl


def load_hypers():
    """Return the SFH hyperparameters of the 10-09 max-L point, as map.py fixes them."""
    d = np.load(POST_NPZ, allow_pickle=True)
    best = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    return {k: float(best[k]) for k in HYPERS}, float(d["z"])


def setup(unmasked_lines=()):
    """Return the data, observation and truth used by every mock and fit.

    Returns
    -------
    dict
        ``z``, ``flux``, ``unc``, ``good``, ``obs``, ``truth``, ``maxl`` and ``hypers``.
    """
    hypers, z_post = load_hypers()
    z, flux, unc, good = load_data(MASK_LINES, unmasked_lines)
    assert np.isclose(z, z_post)
    obs = make_obs(flux, unc, good, "rmatrix")
    truth, maxl = load_truth()
    return dict(z=z, flux=flux, unc=unc, good=good, obs=obs, truth=truth, maxl=maxl, hypers=hypers)


def theta_for(model, values, fill=None):
    """Return a theta vector of ``model`` from a dict by name, clipped into the prior."""
    fill = fill or {}
    th = []
    for k in model.theta_labels():
        v = values[k] if k in values else fill[k]
        name = k.split("_")[0] if k.startswith("logsfr_ratios") else k
        if not k.startswith("logsfr_ratios"):
            lo, hi = model.config_dict[name]["prior"].range
            v = float(np.clip(v, float(np.atleast_1d(lo)[0]), float(np.atleast_1d(hi)[0])))
        th.append(v)
    return np.array(th, float)


def prior_bounds(model):
    """Return the lower and upper prior bounds of each theta entry (inf for logsfr_ratios)."""
    lo, hi = [], []
    for k in model.theta_labels():
        if k.startswith("logsfr_ratios"):
            lo.append(-np.inf)
            hi.append(np.inf)
            continue
        a, b = model.config_dict[k]["prior"].range
        lo.append(float(np.atleast_1d(a)[0]))
        hi.append(float(np.atleast_1d(b)[0]))
    return np.array(lo), np.array(hi)


def jitter_start(model, theta, seed, frac=0.05):
    """Return a start jittered about ``theta``, drawn until it lies inside the prior.

    Parameters
    ----------
    frac : float
        Jitter width as a fraction of the prior range.
    """
    rng = np.random.default_rng(seed)
    lo, hi = prior_bounds(model)
    scale = np.where(np.isfinite(hi - lo), frac * (hi - lo), 0.0)
    if "logsfr_ratios" in model.theta_index:
        sl = model.theta_index["logsfr_ratios"]
        sd = np.sqrt(np.diag(model.config_dict["logsfr_ratios"]["prior"].params["Sigma"]))
        scale[sl] = sd
    for _ in range(1000):
        st = theta + rng.normal(0.0, scale)
        if np.all((st > lo) & (st < hi)):
            return st
    raise RuntimeError("no jittered start inside the prior in 1000 draws")


def noise(unc, good, seed):
    """Return the Gaussian noise of one seed, zero on pixels that are not fitted."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=unc.size)
    return np.where(good, e * np.where(good, unc, 0.0), 0.0)
