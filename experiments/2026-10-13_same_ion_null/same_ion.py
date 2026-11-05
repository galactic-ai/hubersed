"""Model, start point and windows of the same-ion null test of the second-component fractions.

Builds on F1 of 2026-10-13_oiii_second_component: split forbidden widths, a free low-ionization
fraction ``eline_fbroad_forb`` and a free high-ionization fraction ``eline_fbroad_forb_hi``. Here
each line of two doublets gets its own second-component fraction: [O III] 4959 and 5007, and
[N II] 6548 and 6584. Each pair comes from one upper level of one ion, so both lines are emitted by
the same gas in a fixed ratio and must have the same profile. A difference between the fitted
fractions of a pair measures the systematics of the fit (noise, continuum, blends, the line-spread
function), not physics. Every other line keeps its group fraction, and the widths and the second
component's width and shift stay shared as in F1.
"""

import numpy as np
from map import MASKS
from model_setup import get_cue, load_data, make_obs
from oiii_setup import OUT as OIII_OUT
from oiii_setup import make_fit_params
from prospect.models import priors

from hubersed.paths import PATHS
from hubersed.sps.resolution_model import ResolutionMatrixModel

OUT = PATHS["RESULTS"] / "2026-10-13_same_ion_null"
# per-line fraction parameter, the Cue line it applies to, and the group fraction it replaces
PER_LINE = {
    "eline_fbroad_oiii4959": ("[O III] 4959", "eline_fbroad_forb_hi"),
    "eline_fbroad_oiii5007": ("[O III] 5007", "eline_fbroad_forb_hi"),
    "eline_fbroad_nii6548": ("[N II] 6548", "eline_fbroad_forb"),
    "eline_fbroad_nii6584": ("[N II] 6584", "eline_fbroad_forb"),
}
PAIRS = {
    "[O III] 5007 - 4959": ("eline_fbroad_oiii5007", "eline_fbroad_oiii4959"),
    "[N II] 6584 - 6548": ("eline_fbroad_nii6584", "eline_fbroad_nii6548"),
}
# parameters re-optimized at each profile point, besides the other per-line fractions
LINE_PROFILE = (
    "eline_sigma",
    "eline_fbroad",
    "eline_sigma_broad",
    "eline_vbroad",
    "eline_sigma_forb",
    "eline_fbroad_forb",
    "eline_sigma_forb_hi",
    "eline_fbroad_forb_hi",
)
GAS = ("gas_logz", "gas_logu", "gas_lognH", "gas_logno", "gas_logco")
# rest-frame vacuum windows in Angstrom
WINDOWS = {
    "[OIII]4959": (4950, 4970),
    "[OIII]5007": (4995, 5022),
    "[NII]6548": (6540, 6556),
    "Halpha": (6556, 6572),
    "[NII]6584": (6576, 6592),
    "[SII]6716": (6708, 6724),
    "Hbeta": (4845, 4880),
}


class PerLineModel(ResolutionMatrixModel):
    """ResolutionMatrixModel with its own second-component fraction for each line in PER_LINE.

    A line takes its per-line fraction only when that parameter is in the model; otherwise it keeps
    the fraction of its group, so with no per-line parameters this is ResolutionMatrixModel.
    """

    def _broad_fraction(self):
        """Return the per-line fractions, with the PER_LINE overrides applied."""
        fb = super()._broad_fraction()
        names = np.char.strip(np.asarray(self.emline_info["name"]).astype(str))
        for k, (line, _) in PER_LINE.items():
            if k in self.params:
                hit = names == line
                assert hit.sum() == 1, (line, int(hit.sum()))
                fb = np.where(hit, self._p(k, 0.0), fb)
        return fb


def make_params(z):
    """Return the F1 parameters with the four per-line fractions added as free parameters."""
    params = make_fit_params(z, "F1")
    for k in PER_LINE:
        params[k] = dict(N=1, isfree=True, init=0.2, prior=priors.TopHat(mini=0.0, maxi=0.9))
    return params


def build():
    """Return the per-line model, the observation list and the Cue source."""
    z, flux, unc, good = load_data(MASKS["mask"])
    model = PerLineModel(make_params(z))
    return model, make_obs(flux, unc, good, "rmatrix"), get_cue()


def f1_best():
    """Return the best saved F1 point of the [O III] experiment as a dict, its name and -ln p."""
    runs = {p.stem: np.load(p) for p in sorted(OIII_OUT.glob("fit_F1_*.npz"))}
    name = min(runs, key=lambda k: float(runs[k]["fun"]))
    r = runs[name]
    return dict(zip([str(x) for x in r["labels"]], r["theta"], strict=True)), name, float(r["fun"])


def start(model):
    """Return the F1 best point with each per-line fraction at its group value."""
    p, _, _ = f1_best()
    for k, (_, group) in PER_LINE.items():
        p[k] = p[group]
    return np.array([p[k] for k in model.theta_labels()], dtype=float)


def chi2_parts(model, obs, cue, theta):
    """Return the chi2 over all fitted pixels and in each of WINDOWS."""
    spec = model.predict(theta, observations=obs, sps=cue)[0][0]
    o = obs[0]
    rest = o.wavelength / (1 + float(np.atleast_1d(model.params["zred"])[0]))
    r2 = np.where(o.mask, ((o.flux - spec) / o.uncertainty) ** 2, 0.0)
    parts = {"all": float(r2.sum())}
    for k, (a, b) in WINDOWS.items():
        parts[k] = float(r2[(rest >= a) & (rest < b)].sum())
    return spec, parts
