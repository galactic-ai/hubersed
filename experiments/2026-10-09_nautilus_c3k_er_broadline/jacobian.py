"""Local Jacobian of the converged ER model at its maximum likelihood point.

QUESTION: which pixels and which emission lines set each of the 29 parameters of the #14 ER fit
of 39627770174637084, and how do the Cue line fluxes respond to each parameter?

Two products, both from central differences of the model at the maximum likelihood point
``best`` of predict.py:

- the pixel Jacobian dm/dtheta on the fitted pixels. It gives the Fisher matrix
  F = J^T diag(1/unc^2) J, the Fisher widths of each parameter (marginal and conditional) next to
  the posterior widths, and the share of each parameter's Fisher information that comes from
  each rest-frame window;
- the line Jacobian d log10 L / dtheta of the dust-attenuated Cue line luminosities that
  prospect stores in ``model._eline_lum``, for the lines inside the fitted range.

Each step is half the weighted posterior standard deviation, one-sided where a step would leave
the prior. The Fisher numbers are local and linear. They ignore the prior and the curvature, so
they are a guide to where information comes from, not a replacement for the posterior.

INPUTS: the ``_pred.npz`` of predict.py (copied from ls6) and the same data, model and build as
the fit. SEED: none (deterministic).

Run from the repository root on the Mac with the C3K_ER build (no AFE_FLAG; [alpha/Fe] is 0 in
the fit). At ``best`` it gives lnL 101547.9788 against 101547.9693 on ls6. Command:
``PYTHONPATH=<site with the C3K_ER fsps>:experiments/2026-10-09_nautilus_c3k_er_broadline
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/jacobian.py``.

OUTPUTS: results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/jacobian.npz,
jacobian_fisher.txt and jacobian_lines.txt.
"""

import numpy as np
from fit import loglike, process_state

from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.paths import PATHS

TID = 39627770174637084
FB, SPLIT, REST_MAX = "shared", 45.0, 9000.0
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
STEM = f"{TID}_er_broad_{FB}_split{SPLIT:g}_rest{REST_MAX:g}_seed0"

# rest-frame vacuum windows; pixels outside all of them count as continuum
WINDOWS = {
    "[OII]3727": (3720, 3736),
    "[NeIII]3869": (3862, 3876),
    "Hdelta": (4095, 4110),
    "Hgamma": (4333, 4349),
    "[OIII]4363": (4359, 4370),
    "Hbeta": (4845, 4880),
    "[OIII]4959": (4950, 4970),
    "[OIII]5007": (4995, 5022),
    "HeI5876": (5868, 5885),
    "[OI]6300": (6292, 6312),
    "[NII]6548": (6540, 6556),
    "Halpha": (6556, 6572),
    "[NII]6584": (6576, 6595),
    "[SII]6716": (6708, 6725),
    "[SII]6731": (6725, 6742),
}


def weighted_std(points, log_w):
    """Weighted standard deviation of each column of the posterior points."""
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    mean = w @ points
    return np.sqrt(w @ (points - mean) ** 2)


def evaluate(model, obs, sps, theta):
    """Return the model spectrum and the attenuated Cue line luminosities at theta."""
    spec = np.asarray(model.predict(theta, observations=obs, sps=sps)[0][0], float)
    return spec, np.array(model._eline_lum, float)


def main():
    d = np.load(OUT / f"{STEM}_pred.npz")
    best, labels = d["best"], [str(x) for x in d["labels"]]
    model, obs, sps = process_state(TID, FB, SPLIT, REST_MAX)
    assert list(model.theta_labels()) == labels
    good, unc = d["good"], d["unc"]
    lo, hi = (np.array(b, float) for b in zip(*model.theta_bounds(), strict=True))

    lnl = loglike(best, TID, FB, SPLIT, REST_MAX)
    spec0, lum0 = evaluate(model, obs, sps, best)
    ewave = np.array(model._eline_wave, float)
    print(f"lnL at best {lnl:.4f} (ls6 {float(d['max_lnl']):.4f})")
    print(
        f"max |spec - sp_best| / unc on good pixels: "
        f"{np.max(np.abs(spec0 - d['sp_best'])[good] / unc[good]):.2e}"
    )

    sd = weighted_std(d["points"], d["log_w"])
    n = len(best)
    jac = np.zeros((n, good.sum()))
    ljac = np.zeros((n, len(lum0)))
    steps = np.zeros(n)
    for i in range(n):
        h = 0.5 * sd[i]
        up, dn = best.copy(), best.copy()
        up[i] = min(best[i] + h, hi[i])
        dn[i] = max(best[i] - h, lo[i])
        steps[i] = up[i] - dn[i]
        s_up, l_up = evaluate(model, obs, sps, up)
        s_dn, l_dn = evaluate(model, obs, sps, dn)
        jac[i] = (s_up - s_dn)[good] / steps[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            ljac[i] = (np.log10(l_up) - np.log10(l_dn)) / steps[i]
        print(f"{labels[i]:>18s} step {steps[i]:.4g}", flush=True)

    # Fisher matrix and its split over windows
    rest = WAVE_OBS[good] / (1 + float(d["z"]))
    region = np.full(rest.size, "continuum", dtype=object)
    for name, (a, b) in WINDOWS.items():
        region[(rest >= a) & (rest < b)] = name
    jw = jac / unc[good]
    fisher = jw @ jw.T
    sig_cond = 1 / np.sqrt(np.diag(fisher))
    sig_marg = np.sqrt(np.diag(np.linalg.pinv(fisher)))
    names = list(WINDOWS) + ["continuum"]
    share = np.array([(jw[:, region == r] ** 2).sum(axis=1) for r in names]).T
    share /= share.sum(axis=1, keepdims=True)
    lines = (rest.min() <= ewave) & (ewave <= rest.max()) & (lum0 > 0)

    np.savez(
        OUT / str(TID) / "jacobian.npz",
        labels=labels,
        best=best,
        steps=steps,
        post_sd=sd,
        jac=jac,
        wave=WAVE_OBS[good],
        unc=unc[good],
        region=region.astype(str),
        fisher=fisher,
        sig_cond=sig_cond,
        sig_marg=sig_marg,
        windows=np.array(names),
        share=share,
        eline_wave=ewave,
        eline_lum=lum0,
        eline_jac=ljac,
    )

    with open(OUT / str(TID) / "jacobian_fisher.txt", "w") as f:
        f.write(
            "Fisher widths at best (local, no prior) and posterior widths. Share columns: "
            "percent of each parameter's Fisher information from each window.\n"
        )
        top = [
            f"{'parameter':>18s} {'best':>9s} {'post_sd':>9s} {'fisher_marg':>11s} "
            f"{'fisher_cond':>11s}  largest shares"
        ]
        for i, lab in enumerate(labels):
            o = np.argsort(share[i])[::-1][:4]
            tops = ", ".join(f"{names[k]} {100 * share[i, k]:.0f}" for k in o)
            top.append(
                f"{lab:>18s} {best[i]:9.4f} {sd[i]:9.4f} {sig_marg[i]:11.4g} "
                f"{sig_cond[i]:11.4g}  {tops}"
            )
        f.write("\n".join(top) + "\n\nFull share table (percent):\n")
        f.write(f"{'parameter':>18s} " + " ".join(f"{w[:11]:>11s}" for w in names) + "\n")
        for i, lab in enumerate(labels):
            f.write(f"{lab:>18s} " + " ".join(f"{100 * v:11.1f}" for v in share[i]) + "\n")

    with open(OUT / str(TID) / "jacobian_lines.txt", "w") as f:
        f.write(
            "d log10 L / d theta of the attenuated Cue line luminosities at best, for lines in "
            "the fitted range with L > 1e-3 L(Hbeta). Units: dex per unit of the parameter.\n"
        )
        hb = lum0[np.argmin(np.abs(ewave - 4862.7))]
        keep = np.where(lines & (lum0 > 1e-3 * hb))[0]
        cols = [i for i, lab in enumerate(labels) if np.any(np.abs(ljac[i, keep]) > 1e-6)]
        f.write(
            f"{'wave':>9s} {'L/L(Hb)':>8s} "
            + " ".join(f"{labels[i][:12]:>12s}" for i in cols)
            + "\n"
        )
        for k in keep:
            f.write(
                f"{ewave[k]:9.2f} {lum0[k] / hb:8.4f} "
                + " ".join(f"{ljac[i, k]:12.4f}" for i in cols)
                + "\n"
            )
    print("wrote jacobian.npz, jacobian_fisher.txt, jacobian_lines.txt")


if __name__ == "__main__":
    main()
