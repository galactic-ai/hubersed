"""
Residual anatomy for high-S/N high-chi2 galaxies.

Question separated: is a high chi2 (a) genuine MODEL misspecification -> structured
residual concentrated at specific features (emission lines, Balmer, 4000A break),
or (b) sigma UNDERESTIMATE -> flat white residual inflated everywhere with no
structure? chi2 magnitude alone can't tell them apart; residual STRUCTURE can.

Re-fits each target with the SAME machinery as map_chi2.py (continuum MAP -> full
Cue MAP), then saves (wave, data, model, sigma, mask) and plots normalized
residuals (data-model)/sigma vs rest wavelength, with line markers. Also reports
how much of chi2 sits IN line windows vs the continuum (line-driven fraction).

Usage (from bin/prospector/):  python residual_anatomy.py
Targets default to /tmp/resid_targets.npy; override with --gidx 135964 46368 ...
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prospect.fitting import lnprobfn

import parameter_file as P
from fit_config import build_continuum_model, build_full_cue_model
from map_chi2 import load_by_index, _map_optimize, _fsps, _cue
from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies
from hubersed.paths import PATHS

WAVE_OBS = P.WAVE_OBS
RESULTS = PATHS["RESULTS"]

# rest-frame lines to mark / test (vacuum A)
LINES = {
    "[OII]": 3728.5,
    "Hd": 4102.9,
    "Hg": 4341.7,
    "Hb": 4862.7,
    "[OIII]4959": 4960.3,
    "[OIII]5007": 5008.2,
    "[OI]6300": 6302.0,
    "Ha": 6564.6,
    "[NII]6584": 6585.3,
    "[SII]6717": 6718.3,
    "[SII]6731": 6732.7,
    "NaD": 5891.6,
    "CaK": 3934.8,
    "CaH": 3969.6,
    "Mgb": 5176.7,
}


def fit_one(gidx, maxfev=20_000, cont_nseeds=3, full_nseeds=3):
    """Rich-budget continuum->full Cue MAP. Returns everything needed to plot."""
    spec, ivar, z, tid = load_by_index(gidx)
    spec_m = flambda_to_maggies(WAVE_OBS, spec)
    ivar_m = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
    sigma = 1 / np.sqrt(np.where(ivar_m > 0, ivar_m, np.inf))
    mask = (sigma > 0) & np.isfinite(sigma) & np.isfinite(spec_m)

    sps = _fsps()
    fw = sps.ssp.emline_wavelengths
    fopt = fw[(fw > 3600) & (fw < 9824)]
    mask_em = P.mask_spectral_lines(
        WAVE_OBS, mask, z, halfwidth_kms=1500.0, line_waves=fopt
    )
    obs_em = P.build_obs(spec=spec_m, unc=sigma, mask=mask_em)
    obs_full = P.build_obs(spec=spec_m, unc=sigma, mask=mask)

    cmodel, ctemplate = build_continuum_model(z)

    def neg_cont(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(
                    th, model=cmodel, observations=obs_em, sps=sps, nested=False
                )
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    bc = _map_optimize(
        neg_cont, cmodel.theta.copy(), n_seeds=cont_nseeds, maxfev=maxfev
    )
    theta_cont = bc.x

    sps = _cue()
    fmodel, ftemplate = build_full_cue_model(ctemplate, theta_cont, cmodel, z)

    def neg_full(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(
                    th, model=fmodel, observations=obs_full, sps=sps, nested=False
                )
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    bf = _map_optimize(
        neg_full, fmodel.theta.copy(), n_seeds=full_nseeds, maxfev=maxfev
    )
    theta_map = bf.x

    preds, _ = fmodel.predict(theta_map, observations=obs_full, sps=sps)
    model = preds[0]
    m = obs_full[0].mask
    data = obs_full[0].flux
    unc = obs_full[0].uncertainty
    resid = np.where(m, (data - model) / unc, np.nan)
    chi2 = float(np.nansum(resid[m] ** 2))
    ndof = int(m.sum()) - len(theta_map)
    return dict(
        gidx=gidx,
        id=tid,
        z=z,
        wave=WAVE_OBS,
        data=data,
        model=model,
        unc=unc,
        mask=m,
        resid=resid,
        chi2_red=chi2 / ndof,
        npix=int(m.sum()),
    )


def line_driven_fraction(r, wave, z, half_A=12.0):
    """Fraction of total chi2 (=sum r^2) that sits within +-half_A (rest) of any
    marked line. High -> line-driven (model fail at lines). Low -> continuum/flat
    (sigma inflation or continuum-shape mismatch)."""
    rest = wave / (1 + z)
    inline = np.zeros(rest.shape, bool)
    for lw in LINES.values():
        inline |= np.abs(rest - lw) < half_A
    r2 = np.where(np.isfinite(r), r**2, 0.0)
    tot = r2.sum()
    frac_line = r2[inline].sum() / tot if tot > 0 else np.nan
    # fraction of masked pixels that are in-line (for context: line pixels are few)
    frac_pix = inline[np.isfinite(r)].mean()
    return frac_line, frac_pix


def main():
    if "--gidx" in sys.argv:
        i = sys.argv.index("--gidx")
        targets = [int(x) for x in sys.argv[i + 1 :] if x.lstrip("-").isdigit()]
    else:
        targets = list(np.load("/tmp/resid_targets.npy").astype(int))
    print("targets:", targets)

    n = len(targets)
    fig, axes = plt.subplots(
        n, 2, figsize=(18, 2.6 * n), gridspec_kw={"width_ratios": [2.4, 1]}
    )
    if n == 1:
        axes = axes[None, :]
    rows = []
    for k, gi in enumerate(targets):
        try:
            R = fit_one(int(gi))
        except Exception as e:
            print(f"{gi}: FAIL {type(e).__name__}: {e}")
            continue
        rest = R["wave"] / (1 + R["z"])
        m = R["mask"]
        fl, fp = line_driven_fraction(R["resid"], R["wave"], R["z"])
        rows.append((R["gidx"], R["id"], R["z"], R["chi2_red"], fl, fp))
        print(
            f"gidx {R['gidx']:>8d} id {R['id']}  z={R['z']:.3f}  chi2_red={R['chi2_red']:.2f}"
            f"  line-driven frac of chi2={fl:.2f}  (line pix frac={fp:.3f})"
        )

        # left: data vs model
        ax = axes[k, 0]
        ax.plot(rest[m], R["data"][m], lw=0.5, color="k", label="data")
        ax.plot(
            rest[m], R["model"][m], lw=0.5, color="C1", alpha=0.8, label="MAP model"
        )
        for nm, lw in LINES.items():
            ax.axvline(lw, color="r", ls=":", lw=0.4, alpha=0.4)
        ax.set_xlim(3700, 7000)
        ax.set_title(
            f"gidx {R['gidx']}  z={R['z']:.3f}  chi2_red={R['chi2_red']:.2f}  "
            f"line-driven={fl:.2f}",
            fontsize=9,
        )
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")

        # right: normalized residual
        ax2 = axes[k, 1]
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.axhspan(-1, 1, color="C2", alpha=0.12)  # +-1 sigma band
        ax2.plot(rest[m], R["resid"][m], lw=0.4, color="k")
        for nm, lw in LINES.items():
            ax2.axvline(lw, color="r", ls=":", lw=0.4, alpha=0.4)
        ax2.set_xlim(3700, 7000)
        ax2.set_ylim(-8, 8)
        ax2.set_title("(data-model)/sigma", fontsize=8)

    axes[-1, 0].set_xlabel("rest wavelength [A]")
    axes[-1, 1].set_xlabel("rest wavelength [A]")
    fig.tight_layout()
    out = RESULTS / "residual_anatomy.pdf"
    fig.savefig(out, bbox_inches="tight")
    np.save(
        RESULTS / "residual_anatomy_summary.npy",
        np.array(rows, dtype=float) if rows else np.array([]),
    )
    print(f"\nsaved -> {out}")
    print("summary (gidx, id, z, chi2_red, line_driven_frac, line_pix_frac):")
    for r in rows:
        print("  ", tuple(round(x, 3) if isinstance(x, float) else x for x in r))


if __name__ == "__main__":
    main()
