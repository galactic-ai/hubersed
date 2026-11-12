"""Bounded-EW free line columns of the prospector fork on 39627770174637084 at one parameter point.

QUESTION: on real data, does the fork's box probability P_B of each block of free columns agree
with an independent value, how large are the blocks, and what does a prediction cost?
HYPOTHESIS: |ln P_B - reference| < 0.01 for blocks of 1-2 (quadrature) and within 3 Monte Carlo
errors for larger blocks (10^7 draws); the broad H column joins most columns into one block.
INPUTS: the 10-13 max-L point (_pred.npz `best`) with its data, mask and R (model_setup.COADD);
broad widths and offset from the 10-15 free-line MAP (map_free.npz): eline_sigma_broad,
eline_sigma_broad_forb, eline_vbroad. Columns: weak ions (narrow, replace Cue) and broad columns
(add to Cue) of hubersed.sps.free_lines.IONS, W from results/2026-11-06_line_ew_prior/
ion_limits.json. C3K_ER Cue, afe 0.
SEED: numpy default_rng 0 for the Monte Carlo reference.
COMMAND: from the repository root on the Mac, with the fork working copy first on the path,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:$HOME/Astronomy_Research/prospector:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-11-07_free_line_084_check/analysis.py``.
RESULT: 2026-11-07, check.txt. All 8 blocks pass: blocks of 1-2 agree with erf/quadrature to
3e-11, blocks of 3 within 1.2 MC sigma. 13 of 15 columns have fitted pixels ([NII]5755 and
[SIII]9069 none), in blocks [3, 3, 2, 1 x 5]; broad H joins only [NeIII] and [NII] broad. 45 ms per
prediction against 39 ms narrow only. [ArIII]7135 has 1 fitted pixel near its centre, so its sd is
5e5 times its bound and its net term is about -0.04: it drops out, as derived. Assumptions: one
parameter point; broad widths 86-88 km/s (wider broad lines make larger blocks); fork working copy.
FIGURES: results/2026-11-07_free_line_084_check/check.txt.
"""

# %%
import json
import time

import model_setup as ms
import numpy as np
from prospect.models import line_marginalisation as lm
from prospect.models.line_marginalisation import LineColumn
from prospect.models.sedmodel import HyperSpecModel
from prospect.observation import ResolutionMatrixSpectrum
from scipy import integrate, special

from hubersed.paths import PATHS
from hubersed.sps.free_lines import IONS, ion_lines
from hubersed.sps.resolution_matrix import load_resolution_npz

OUT = PATHS["RESULTS"] / "2026-11-07_free_line_084_check"
PRED = (
    PATHS["RESULTS"]
    / "2026-10-13_nautilus_resolution_matrix"
    / "39627770174637084_rmatrix_mask_er_broad_shared_split45_rest9000_seed0_pred.npz"
)
MAP = PATHS["RESULTS"] / "2026-10-15_broadline_free_amplitude" / "map_free.npz"
LIMITS = PATHS["RESULTS"] / "2026-11-06_line_ew_prior" / "ion_limits.json"
WEAK = ("[SII]", "[OI]", "[ArIII]7135", "[OIII]4363", "[OII]7325", "[NII]5755", "[NeIII]",
        "[SIII]6312", "[SIII]9069")  # fmt: skip
OLD_BROAD = ("eline_fbroad", "eline_fbroad_forb", "eline_sigma_forb")
BROAD = ("eline_sigma_broad", "eline_sigma_broad_forb", "eline_vbroad")
NDRAW = 10**7

# %% data, R, the parameter point and the columns
p = np.load(PRED, allow_pickle=True)
z, flux, unc, good = float(p["z"]), p["flux"], p["unc"], p["good"].astype(bool)
best = dict(zip([str(x) for x in p["labels"]], p["best"], strict=True))
mp = np.load(MAP, allow_pickle=True)
best.update({k: v for k, v in zip(mp["labels"], mp["theta"], strict=True) if k in BROAD})
r = load_resolution_npz(ms.COADD)
cue = ms.get_cue()
limits = json.loads(LIMITS.read_text())

params = ms.make_params(z)
for k in OLD_BROAD:
    params.pop(k, None)
for k in BROAD:
    params[k] = dict(N=1, isfree=False, init=float(best[k]))
probe = HyperSpecModel(params)
names = np.char.strip(np.asarray(probe.emline_info["name"]).astype(str))
index = ion_lines(names)
columns = [LineColumn(ion, tuple(names[index[ion]]), limits[ion]) for ion in WEAK]
for ion, spec in IONS.items():
    if spec["broad"] is not None:
        width = "eline_sigma_broad" if spec["broad"] == "balmer" else "eline_sigma_broad_forb"
        columns.append(
            LineColumn(f"{ion} broad", tuple(names[index[ion]]), limits[f"{ion} broad"], width,
                       "eline_vbroad", replace=False)
        )  # fmt: skip

model = HyperSpecModel(params, line_columns=columns)
obs = ResolutionMatrixSpectrum(
    wavelength=r["wave"], resolution_matrix=r["res"], flux=flux, uncertainty=unc, mask=good
)
obs.rectify()
theta = np.array([best[k] for k in model.theta_labels()])

# %% prediction, timing against the narrow-only model
pred = model.predict(theta, observations=[obs], sps=cue)[0][0]
res, used, upper = model._line_marginal, model._line_column_index, model._line_column_upper
plain = HyperSpecModel(params)
times = {}
for label, m in (("columns", model), ("narrow only", plain)):
    t0 = time.perf_counter()
    for _ in range(5):
        m.predict(theta, observations=[obs], sps=cue)
    times[label] = (time.perf_counter() - t0) / 5
model.predict(theta, observations=[obs], sps=cue)
lnlike = -0.5 * np.sum(((pred - flux) / unc)[good] ** 2) - 0.5 * np.sum(
    np.log(2 * np.pi * unc[good] ** 2)
)


# %% independent ln P_B of each block
def ref_1d(mu, s, b):
    """Return ln(Phi((b - mu)/s) - Phi(-mu/s)) with scipy's log_ndtr."""
    lo, hi = -mu / s, (b - mu) / s
    if lo + hi > 0:
        lo, hi = -hi, -lo
    return special.log_ndtr(hi) + np.log1p(-np.exp(special.log_ndtr(lo) - special.log_ndtr(hi)))


def ref_2d(mu, cov, b):
    """Return ln P of the box by quadrature over the first amplitude of its conditional."""
    s1, s2 = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (s1 * s2)
    sc = s2 * np.sqrt(1 - rho**2)

    def f(x):
        m2 = mu[1] + rho * s2 * x
        return np.exp(-0.5 * x * x - 0.5 * np.log(2 * np.pi) + ref_1d(m2, sc, b[1]))

    val, _ = integrate.quad(f, -mu[0] / s1, (b[0] - mu[0]) / s1, epsabs=0, epsrel=1e-10, limit=500)
    return np.log(val), 0.0


def ref_mc(mu, cov, b, rng):
    """Return ln P of the box from NDRAW normal draws and its 1 sigma error in ln P.

    The error is the binomial one, at least 1/NDRAW, which is the resolution when no draw falls
    outside the box.
    """
    inside = 0
    for _ in range(10):
        a = rng.multivariate_normal(mu, cov, size=NDRAW // 10)
        inside += np.count_nonzero(np.all((a >= 0) & (a <= b), axis=1))
    frac = inside / NDRAW
    return np.log(frac), max(np.sqrt((1 - frac) / (frac * NDRAW)), 1 / NDRAW) if inside else np.inf


rng = np.random.default_rng(0)
cols = [model.line_columns[j] for j in used]
lines = [
    "39627770174637084, 10-13 max-L point; broad widths and offset from the 10-15 free-line MAP:",
    ", ".join(f"{k} {best[k]:.2f}" for k in BROAD) + " km/s.",
    f"{len(columns)} columns given, {len(used)} with flux on the fine grid, "
    f"{sum(len(b) for b in res.blocks)} with fitted pixels, in {len(res.blocks)} blocks of sizes "
    f"{sorted((len(b) for b in res.blocks), reverse=True)}.",
    f"time per prediction: {times['columns'] * 1e3:.0f} ms with columns, "
    f"{times['narrow only'] * 1e3:.0f} ms narrow only.",
    f"ln L(theta, a_hat) {lnlike:.2f}, penalty {model._ln_eline_penalty:.2f}.",
    "",
    "block: columns | ln P_B fork | reference (method) | difference | pass",
]
ok_all = True
for blk in res.blocks:
    mu, cov, b = res.amplitude[blk], res.covariance[np.ix_(blk, blk)], upper[blk]
    got = lm.box_lnprob(mu, cov, b)
    if len(blk) == 1:
        ref, err, how = ref_1d(mu[0], np.sqrt(cov[0, 0]), b[0]), 0.0, "erf"
    elif len(blk) == 2:
        (ref, err), how = ref_2d(mu, cov, b), "quadrature"
    else:
        ref, err = ref_mc(mu, cov, b, rng)
        how = f"MC {NDRAW:.0e}, 1 sigma {err:.1e}"
    ok = abs(got - ref) < (0.01 if len(blk) <= 2 else 3 * err)
    ok_all &= bool(ok)
    names_b = ", ".join(cols[i].name for i in blk)
    lines.append(f"{names_b} | {got:.5f} | {ref:.5f} ({how}) | {got - ref:+.1e} | {ok}")
lines += ["", "column | a_hat / sd | bound b / sd"]
sd = np.sqrt(np.diag(res.covariance))
for i, c in enumerate(cols):
    if sd[i] > 0:
        lines.append(f"{c.name} | {res.amplitude[i] / sd[i]:+.2f} | {upper[i] / sd[i]:.2f}")
    else:
        lines.append(f"{c.name} | no fitted pixels | -")
lines.append(f"\nall blocks pass: {ok_all}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
