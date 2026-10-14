"""Mock spectra with injected line profiles, and the fit models, for the broad-line injection test.

The mocks go through the same pipeline as the 10-13 resolution-matrix fit (ResolutionMatrixModel,
the target's own R, the same pixels, the missing-line mask, maggies). Only the emission lines
differ. ``InjectedLineModel`` is a ResolutionMatrixModel whose ``fine_lines`` builds each line
from a list of Gaussian components, each with a weight (a multiple of the line's Cue flux), an
intrinsic width and a velocity shift. Everything else, the continuum, R' and the response, is the
parent's code.

Scenarios. All use the truth point, the rmatrix mask MAP of 2026-10-13_nautilus_resolution_matrix.

- ``model`` (S0): the components of our own model, (1 - f_b) G(sigma) + f_b G(sigma_b, v_b) with
  the widths of TwoCompLineModel. It must reproduce ResolutionMatrixModel.
- ``S1``: every line has EmFit's three-component Halpha profile for this target.
- ``S2``: our two-component shape, with broad fractions per line group.
- ``S3``: our two-component shape, with the broad component of [N II], [S II], [O I] and [O III]
  scaled by literature broad-to-narrow ratio offsets. Narrow components keep the Cue ratios.

Sources of the injected numbers. Line numbers refer to files in ~/Documents/UberSED/knowledge/.

- EmFit row of 39627770174637084 in ``_evidence/2026-09-30_kin_vs_ion/emfit_rows_2obj.csv``,
  columns ha_n_*, ha_out_*, ha_b_*. Widths ha_n_sigma 10.218, ha_out_sigma 58.905,
  ha_b_sigma 154.919 km/s. Fluxes 1463.63, 636.00, 95.36, so fractions 0.6668, 0.2897, 0.0434.
  Shifts relative to the narrow centre from ha_*_mean, -0.870 and -2.888 km/s. The same numbers
  are in lit_broad_emission_components_2026-10-13.md lines 23-24. ``check_emfit`` re-reads the csv.
- Per-group broad fractions (S2): broad fraction of low-ionization forbidden lines 1.2 times the
  Balmer one and of high-ionization lines 0.7 times, lit_broad_emission_components_2026-10-13.md
  line 393, from Hogarth+20 tab2 (lines 338-341, Ha 0.32, [NII] 0.38, [OIII]5007 0.22) and
  EmFit for this target (line 344, [NII] 0.373 and [SII] 0.349 vs Ha 0.303).
- Broad ratio offsets (S3), each in dex on the broad component's line over Halpha (or Hbeta):
  [N II] +0.136 and [S II] +0.091 from EmFit for this target, log N2Ha -1.005 to -0.869 and log
  S2Ha -0.890 to -0.799 (lit_broad_emission_components_2026-10-13.md line 26). [O I] +0.5 from
  CLASSY VIII (line 116). [O III] -0.210 = log10(4.41 / 7.16) from Hogarth+20 tab2 (line 321,
  marked there as the note's own arithmetic on a checked table).

Assumptions that no source fixes, stated in the report too.

- S2 groups. Recombination lines (Balmer, Paschen, He I) keep the Balmer fraction. High
  ionization is ``broadline.HIGH_ION_PREFIXES`` ([O III], [Ne III], [Ar IV], [Ne IV], He II).
  Every other line is low ionization, including the intermediate ions [S III], [Ar III] and
  [Cl III], as in line_groups_ionization_zones_2026-10-10.md line 22. No paper read gives broad
  fractions for the intermediate ions.
- S3 offsets apply to every line of the ion, auroral lines included ([N II] 5755, [S II] 4070,
  [O I] 5577, [O III] 4363), and both members of each doublet. Other ions are unchanged.
- S1 gives all lines, [O III] included, the Halpha profile. EmFit fits [O III] with one 17.3 km/s
  Gaussian and finds no second component for it (oiii5007_out_sigma_flag = -1).
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
    """Re-read the EmFit csv row and check it matches ``EMFIT_HA``.

    Returns
    -------
    str
        A one-line summary, or a note that the file is missing.
    """
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
    """Return the S2 group of a Cue line name: recombination, high or low."""
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
    params : dict
        Model parameter specification, as for ResolutionMatrixModel.
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
            For each line, a list of (weight, intrinsic sigma in km/s, shift in km/s). The weight
            multiplies the line's Cue flux, so the weights of a line sum to its flux multiplier.
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
        """Return the injected line spectrum on the fine grid, with the parent's line selection.

        Parameters
        ----------
        fine : np.ndarray
            The fine grid in observed Angstrom.

        Returns
        -------
        spec : np.ndarray
            The line spectrum in maggies at ``fine``.
        added : np.ndarray of bool
            The lines that were added.
        """
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
    z : float
        Redshift.
    fit_model : {"shared", "free", "none"}
        ``shared`` is the 10-13 model (make_params of model_setup). ``free`` adds a free forbidden
        broad fraction, as add_broad_params(forbidden_broad="free"). ``none`` fixes eline_fbroad
        at 0, so no line has a broad component, fixes eline_sigma_broad and eline_vbroad (they
        then do nothing) and gives eline_sigma and eline_sigma_forb the TopHat(10, 250) prior
        of the Cue model before add_broad_params.
    hyper_values : dict
        Values of the SFH hyperparameters ``HYPERS``, fixed in every model.

    Returns
    -------
    dict
        The parameter specification.
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


def setup():
    """Return the data, observation and truth used by every mock and fit.

    Returns
    -------
    dict
        ``z``, ``flux`` (the real data), ``unc``, ``good``, ``obs`` (built on the real flux),
        ``truth`` and ``maxl`` (dicts by name) and ``hypers``.
    """
    hypers, z_post = load_hypers()
    z, flux, unc, good = load_data(MASK_LINES)
    assert np.isclose(z, z_post)
    obs = make_obs(flux, unc, good, "rmatrix")
    truth, maxl = load_truth()
    return dict(z=z, flux=flux, unc=unc, good=good, obs=obs, truth=truth, maxl=maxl, hypers=hypers)


def theta_for(model, values, fill=None):
    """Return a theta vector of ``model`` from a dict by name, clipped into the prior.

    Parameters
    ----------
    model : prospect.models.ProspectorParams
        The model.
    values : dict
        Parameter values by theta label.
    fill : dict, optional
        Values for labels missing from ``values``.

    Returns
    -------
    np.ndarray
        The theta vector.
    """
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

    Bounded parameters get a Gaussian of ``frac`` times their prior range. The logsfr_ratios get a
    Gaussian of their prior standard deviation (the diagonal of the MultiVariateNormal prior).

    Parameters
    ----------
    model : prospect.models.ProspectorParams
        The model.
    theta : np.ndarray
        The centre.
    seed : int
        Seed of the draw.
    frac : float
        Jitter width as a fraction of the prior range.

    Returns
    -------
    np.ndarray
        The jittered start.
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
    """Return the Gaussian noise of one seed, zero on pixels that are not fitted.

    The same seed gives the same noise in every scenario, so scenarios can be compared seed by
    seed.
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(size=unc.size)
    return np.where(good, e * np.where(good, unc, 0.0), 0.0)
