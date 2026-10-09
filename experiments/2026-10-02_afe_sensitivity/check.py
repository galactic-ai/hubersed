"""QUESTION: can the DESI spectra of 37084 and 26597 tell [alpha/Fe] = -0.2 ... +0.6 apart, given the
rest of the split-45 model?
HYPOTHESIS: weakly at best. Both are young-light dominated, and the alpha-sensitive absorption (Mg b,
Fe5270/5335) comes mostly from old stars. If chi2 at fixed theta barely changes, alpha is not worth a
free parameter; if it changes a lot, the next step is a profile (re-optimize the other parameters).
METHOD: take the max-L point of runs A (37084) and C (26597), both forbidden_broad="shared", from the
2026-09-30 split-45 experiment. At each afe on the FSPS grid, predict the spectrum with everything else
fixed and refit only the overall amplitude, analytically. The model spectrum is proportional to stellar
mass, so this is the exact best logmass. Report chi2(afe) - chi2(afe=0) on all fitted pixels, on
continuum pixels only (500 km/s around lines removed), and per rest-frame band. For scale, the same
number for logzsol +-0.1 dex at afe=0. Fixed theta ignores degeneracies, so a large value here does
not mean alpha is measurable.
Second test (noise free, "Asimov"): take model(afe) at the max-L point as fake data with the real sigma,
continuum pixels only, and fit it with afe = 0 models on a logzsol grid, each times a Legendre
polynomial in wavelength of order 0 (amplitude only) or 5 (a smooth stand-in for dust and other
broadband freedom; the real model has no calibration polynomial). The minimum chi2 over the grid is
the Delta chi2 a real fit would have to detect that afe. Age, SFH and gas stay fixed, so it is still an
upper bound. Line pixels are left out because their fluxes follow the gas parameters and recent SFR.
INPUTS: results/2026-09-30_nautilus_c3k_broadline_split45/<tid>_broad_shared_split45_unmask_rest9000_seed0.{h5,json}
ENVIRONMENT: ls6 only. FSPS built with -DAFE_FLAG=1 -mcmodel=medium (knowledge/_evidence/2026-10-01_afe_build),
put first on PYTHONPATH, and SPS_HOME with the alpha C3K_HR spectra. See run.sh.
SEED: none needed (no random draws).
COMMAND: bash experiments/2026-10-02_afe_sensitivity/run.sh (inside an srun step on a compute node)
RESULT: results/2026-10-02_afe_sensitivity/afe_sensitivity.txt
FIGURES: results/2026-10-02_afe_sensitivity/afe_sensitivity_<tid>.png (spectra in afe_spectra_<tid>.npz)
"""

# %%
import importlib.util
import json

import fsps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from nautilus import Sampler  # noqa: E402

from hubersed.fitting.chi2 import WAVE_OBS  # noqa: E402
from hubersed.paths import PATHS  # noqa: E402
from hubersed.sps import broadline  # noqa: E402
from hubersed.sps.parameter_file import build_cue_sps, mask_spectral_lines  # noqa: E402

SRC = PATHS["ROOT"] / "experiments/2026-09-30_nautilus_c3k_broadline_split45/fit.py"
RES_IN = PATHS["RESULTS"] / "2026-09-30_nautilus_c3k_broadline_split45"
OUT = PATHS["RESULTS"] / "2026-10-02_afe_sensitivity"
RUNS = {"A": 39627770174637084, "C": 39628357133926597}
AFE = [-0.2, 0.0, 0.2, 0.4, 0.6]
DZ = 0.1
# logzsol offsets for the noise-free degeneracy fit, and Legendre orders of the multiplicative polynomial
ZGRID = [round(v, 2) for v in np.arange(-0.3, 0.301, 0.05)]
POLY = (0, 5)
# rest-frame bands in Angstrom for the breakdown of the chi2 change
BANDS = {
    "<4500": (0, 4500),
    "4500-5500 (Mg b, Fe)": (4500, 5500),
    "5500-7000": (5500, 7000),
    ">7000": (7000, 1e9),
}


def load_module(path, name):
    """Import a fit.py from another experiment under its own module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def best_point(path, model):
    """Return the max-likelihood point and its log likelihood from a nautilus checkpoint."""
    s = Sampler(
        lambda x: x, lambda x: 0.0, n_dim=model.ndim, n_live=1000, filepath=str(path), resume=True
    )
    cube, _, log_l = s.posterior()
    return model.prior_transform(cube[np.argmax(log_l)]), float(log_l.max())


def chi2_scaled(flux, unc, model, sel):
    """Return the chi2 over ``sel`` after the best overall amplitude, and that amplitude."""
    d, m, iv = flux[sel], model[sel], unc[sel] ** -2.0
    a = np.sum(d * m * iv) / np.sum(m * m * iv)
    return float(np.sum((d - a * m) ** 2 * iv)), float(a)


def chi2_poly(data, model, unc, sel, x, order):
    """Return the chi2 of ``data`` against ``model`` times the best Legendre polynomial of ``order``.

    Order 0 is a single amplitude. ``x`` is wavelength scaled to [-1, 1] over the selected pixels.
    """
    d, s = data[sel], unc[sel]
    design = np.polynomial.legendre.legvander(x[sel], order) * model[sel][:, None]
    coef, *_ = np.linalg.lstsq(design / s[:, None], d / s, rcond=None)
    return float(np.sum(((d - design @ coef) / s) ** 2))


def predict(model, theta, obs, sps, afe, dlogz=0.0):
    """Return the model spectrum at ``theta`` with [alpha/Fe] = ``afe`` and logzsol shifted by ``dlogz``."""
    model.params["afe"] = np.array([afe])
    t = theta.copy()
    t[model.theta_index["logzsol"]] += dlogz
    return model.predict(t, observations=obs, sps=sps)[0][0]


f30 = load_module(SRC, "fit0930")
cue = build_cue_sps()
print("FSPS", fsps.__file__, cue.ssp.libraries)
OUT.mkdir(parents=True, exist_ok=True)
lines, summary, asimov = [], {}, []

# %% per object: chi2 change with afe at the max-L point, amplitude refitted
for tag, tid in RUNS.items():
    stem = f"{tid}_broad_shared_split45_unmask_rest9000_seed0"
    cfg = json.loads((RES_IN / f"{stem}.json").read_text())
    z, flux, unc, good, _ = f30.load_data(tid, cfg["rest_max"], unmask=True)
    params = f30.make_params(z, cfg["forbidden_broad"], cfg["sigma_split"])
    params["afe"] = dict(N=1, isfree=False, init=0.0)
    model = broadline.TwoCompLineModel(params)
    assert list(model.theta_labels()) == cfg["labels"]
    obs = f30.make_obs(flux, unc, good)
    theta, max_lnl = best_point(RES_IN / f"{stem}.h5", model)
    rest = WAVE_OBS / (1 + z)
    cont = mask_spectral_lines(WAVE_OBS, good, z)
    sels = {"all": good, "continuum": cont}
    sels.update({b: cont & (rest >= lo) & (rest < hi) for b, (lo, hi) in BANDS.items()})

    spec = {a: predict(model, theta, obs, cue, a) for a in AFE}
    assert not np.allclose(spec[0.0], spec[0.4]), (
        "afe has no effect: FSPS was not built with AFE_FLAG=1"
    )
    spec_grid = {dz: predict(model, theta, obs, cue, 0.0, dz) for dz in ZGRID if dz != 0.0}
    spec_grid[0.0] = spec[0.0]
    spec_z = {dz: spec_grid[dz] for dz in (-DZ, DZ)}
    base = {k: chi2_scaled(flux, unc, spec[0.0], s)[0] for k, s in sels.items()}
    rows = {}
    for a in AFE:
        rows[f"afe {a:+.1f}"] = {k: chi2_scaled(flux, unc, spec[a], s) for k, s in sels.items()}
    for dz in (-DZ, DZ):
        rows[f"logzsol {dz:+.1f}"] = {
            k: chi2_scaled(flux, unc, spec_z[dz], s) for k, s in sels.items()
        }
    lines.append(
        f"\n{tag} ({tid}), z {z:.5f}, max lnL {max_lnl:.2f}; pixels: all {good.sum()}, "
        f"continuum {cont.sum()}; chi2 at afe=0: all {base['all']:.1f}, "
        f"continuum {base['continuum']:.1f}"
    )
    lines.append(f"{'point':14s}{'amp':>7s}" + "".join(f"{k:>22s}" for k in sels))
    for name, r in rows.items():
        lines.append(
            f"{name:14s}{r['all'][1]:7.3f}" + "".join(f"{r[k][0] - base[k]:22.1f}" for k in sels)
        )
    lo, hi = rest[cont].min(), rest[cont].max()
    x = 2 * (rest - lo) / (hi - lo) - 1
    assert chi2_poly(spec[0.0], spec[0.0], unc, cont, x, POLY[-1]) < 1e-6, "self fit is not exact"
    asimov.append(f"\n{tag} ({tid}): min over logzsol offset {ZGRID[0]:+.2f}..{ZGRID[-1]:+.2f}")
    asimov.append(f"{'fake data':12s}" + "".join(f"{'order ' + str(o):>26s}" for o in POLY))
    best = {o: [] for o in POLY}
    for a in AFE:
        if a == 0.0:
            for o in POLY:
                best[o].append(0.0)
            continue
        cells = []
        for o in POLY:
            c2 = [chi2_poly(spec[a], spec_grid[dz], unc, cont, x, o) for dz in ZGRID]
            i = int(np.argmin(c2))
            best[o].append(c2[i])
            edge = " EDGE" if i in (0, len(ZGRID) - 1) else ""
            cells.append(f"{c2[i]:10.1f} at dlogz {ZGRID[i]:+.2f}{edge:5s}")
        asimov.append(f"{'afe ' + format(a, '+.1f'):12s}" + "".join(f"{c:>26s}" for c in cells))
    summary[tag] = dict(
        rest=rest,
        good=good,
        cont=cont,
        unc=unc,
        spec={a: spec[a] * rows[f"afe {a:+.1f}"]["all"][1] for a in AFE},
        best=best,
    )
    np.savez(
        OUT / f"afe_spectra_{tid}.npz",
        rest=rest,
        flux=flux,
        unc=unc,
        good=good,
        cont=cont,
        afe=AFE,
        spec=np.array([spec[a] for a in AFE]),
        zgrid=ZGRID,
        spec_zgrid=np.array([spec_grid[dz] for dz in ZGRID]),
        **{f"asimov_order{o}": best[o] for o in POLY},
    )

text = (
    "Delta chi2 relative to afe = 0 at each run's max-L point, overall amplitude refitted "
    "(amp = best scale on all pixels).\n" + "\n".join(lines) + "\n"
    "\nNoise-free fit of model(afe) by afe = 0 models: minimum chi2 on continuum pixels, with the "
    "logzsol offset where it occurs (EDGE = at the grid edge, so the true minimum may be lower).\n"
    + "\n".join(asimov)
    + "\n"
)
print(text)
(OUT / "afe_sensitivity.txt").write_text(text)

# %% figures: noise-free Delta chi2 vs afe, and where on the continuum the alpha signal sits
for tag, tid in RUNS.items():
    s = summary[tag]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw=dict(height_ratios=[1, 1.3]))
    for o, ls in zip(POLY, ("o-", "s--"), strict=True):
        ax[0].plot(
            AFE, s["best"][o], ls, label=f"afe = 0 models, logzsol free, polynomial order {o}"
        )
    for ref in (1, 4, 25):
        ax[0].axhline(ref, color="0.7", lw=0.6, ls=":")
        ax[0].text(AFE[-1] + 0.02, ref, f"{ref}", va="center", fontsize=7, color="0.5")
    ax[0].set_yscale("symlog", linthresh=1)
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel("[alpha/Fe] of the fake data")
    ax[0].set_ylabel("minimum chi2 (noise free)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(
        "Can afe = 0 models with free logzsol mimic the alpha model? (continuum pixels)", fontsize=9
    )
    k = 25
    for a in AFE:
        if a == 0.0:
            continue
        d = np.where(
            s["cont"], (s["spec"][a] - s["spec"][0.0]) / np.where(s["cont"], s["unc"], 1), 0
        )
        n = np.convolve(s["cont"].astype(float), np.ones(k), mode="same")
        sm = np.convolve(d, np.ones(k), mode="same") / np.maximum(n, 1)
        ax[1].plot(s["rest"], np.where(s["cont"], sm, np.nan), lw=0.8, label=f"afe {a:+.1f}")
    ax[1].axhline(0, color="0.5", lw=0.5)
    for lo, _ in BANDS.values():
        if lo > 0:
            ax[1].axvline(lo, color="0.8", lw=0.5)
    ax[1].set_xlim(s["rest"][s["good"]].min(), s["rest"][s["good"]].max())
    ax[1].set_xlabel("rest wavelength (A)")
    ax[1].set_ylabel("(model(afe) - model(0)) / sigma\ncontinuum pixels, 25-pixel mean")
    ax[1].set_title(
        "Fixed max-L point, overall amplitude refitted; emission-line windows removed", fontsize=9
    )
    ax[1].legend(frameon=False, ncol=4, fontsize=8)
    fig.suptitle(f"{tag} ({tid}): sensitivity to [alpha/Fe]")
    fig.tight_layout()
    fig.savefig(OUT / f"afe_sensitivity_{tid}.png", dpi=110)
    plt.close(fig)
