"""Is the 4600-4800 A cost of dust_index = 0 in the fit of 39627770174637084 a feature?

QUESTION: region_chi2.py puts the largest single-bin cost of n = 0 (no mask) at rest 4600-4800 A
(+9.9), where the model has no line. Is it a missing feature (He II 4686, the Wolf-Rayet blue bump,
[Fe III] 4658, [Ar IV] 4711/4740) or the smooth continuum bow?

Repeats region_chi2's no-mask refits (n free and n = 0), lists the Cue lines in 4550-4800 A, and
plots rest 4400-5000 A with the residuals and the running chi2 difference. Gives the excess
equivalent width in rest 4640-4700 A. SEED: none.

RESULT: 2026-10-17, he2_window.txt and .png. The data sit 2-8% above both models over rest
~4590-4700 A. The excess EW over 4640-4700 A is 4.3 +- 0.6 A (n free) and 4.6 +- 0.6 A (n = 0).
Among 60 A windows over rest 4000-7400 A these are the two largest excesses (4.7 and 5.8 sigma).
Other windows reach +-3.4 sigma, so the noise is above the pipeline errors. 4600-4700 A carries
+10.0 of the +21.7 no-mask cost of n = 0. The shape matches a Wolf-Rayet bump. A stellar absorption
mismatch or a calibration wiggle is not ruled out.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude:~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/he2_window.py --state <path>``.

OUTPUT: he2_window.png, he2_window.txt and he2_window_theta.npz next to the state file.
"""

import argparse
from pathlib import Path

import fit
import matplotlib.pyplot as plt
import numpy as np
from mask_refit import FREE, best_point, refit
from poco_plots import desi
from region_chi2 import components

from hubersed.plotting.style import use_apj_style

WINDOW = (4400.0, 5000.0)
BUMP = (4640.0, 4700.0)
BIN = 10.0
VAC = 1.00028  # air to vacuum near 4700 A
MARKS = {"He I 4471": 4471.5, "N III 4640": 4640.6, "C III 4650": 4650.3, "[Fe III] 4658": 4658.1,
         "He II 4686": 4685.7, "[Ar IV] 4711": 4711.4, "[Ar IV] 4740": 4740.2, "Hbeta": 4861.3,
         "[O III] 4959": 4958.9}  # fmt: skip


def binned(x, y, a, b, width):
    """Return bin centres and means of ``y`` in ``width`` A bins of ``x`` over [a, b)."""
    edges = np.arange(a, b + width / 2, width)
    idx = np.digitize(x, edges) - 1
    mean = [np.mean(y[idx == i]) if np.any(idx == i) else np.nan for i in range(len(edges) - 1)]
    return 0.5 * (edges[1:] + edges[:-1]), np.array(mean)


def excess_ew(rest, flux, model, cont, unc, sel):
    """Return the equivalent width of ``flux - model`` over ``sel`` in A and its error."""
    dl = np.gradient(rest)[sel]
    ew = np.sum((flux - model)[sel] / cont[sel] * dl)
    err = np.sqrt(np.sum((unc[sel] / cont[sel] * dl) ** 2))
    return ew, err


def main():
    """Refit, write the table and the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    model, obs, cue = fit.process_state("free")
    labels = list(model.theta_labels())
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)
    i_n = labels.index("dust_index")
    out = args.state.with_name("he2_window_theta.npz")
    if out.exists():
        th_free, th_0 = np.load(out)["free"], np.load(out)["n0"]
    else:
        theta0 = best_point(args.state, model)
        th_free, _ = refit(model, theta0.copy(), FREE + ["dust_index"])
        start = theta0.copy()
        start[i_n] = 0.0
        th_0, _ = refit(model, start, FREE)
        np.savez(out, free=th_free, n0=th_0, labels=labels)
    s1, c1, l1 = components(model, obs, cue, th_free)
    s0, c0, _ = components(model, obs, cue, th_0)

    wave = np.asarray(model.emline_info["wave"])
    near = (wave > 4550) & (wave < 4800)
    lines = [f"n free {th_free[i_n]:.3f}, n 0 {th_0[i_n]:.3f}", "Cue lines in rest 4550-4800 A:"]
    lines += [f"  {model.emline_info['name'][i]} {wave[i]:.1f}" for i in np.flatnonzero(near)]
    win = o.mask & (rest >= WINDOW[0]) & (rest < WINDOW[1])
    peak = np.max(
        np.abs(l1[win & (rest > 4550) & (rest < 4800)]) / c1[win & (rest > 4550) & (rest < 4800)]
    )
    lines.append(f"largest |model line| / continuum in 4550-4800 A: {peak:.4f}")
    bump = o.mask & (rest >= BUMP[0]) & (rest < BUMP[1])
    for name, s in (("n free", s1), ("n 0", s0)):
        ew, err = excess_ew(rest, o.flux, s, c1, o.uncertainty, bump)
        lines.append(f"excess EW of data over {name} model, rest {BUMP[0]:.0f}-{BUMP[1]:.0f} A: "
                     f"{ew:.2f} +- {err:.2f} A")  # fmt: skip
    chi_free = ((o.flux - s1) / o.uncertainty) ** 2
    chi_0 = ((o.flux - s0) / o.uncertainty) ** 2
    for a, b in ((4400, 4600), (4600, 4640), (4640, 4700), (4700, 4800), (4800, 5000)):
        sel = o.mask & (rest >= a) & (rest < b)
        lines.append(f"d chi2 (n 0 - n free) rest {a}-{b} A: {np.sum(chi_0[sel] - chi_free[sel]):6.1f} "
                     f"({int(sel.sum())} pixels)")  # fmt: skip
    text = "\n".join(lines)
    print(text)
    args.state.with_name("he2_window.txt").write_text(text + "\n")

    use_apj_style()
    fig, ax = plt.subplots(3, 1, figsize=(7.0, 6.5), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1.4, 1.0], "hspace": 0.05})  # fmt: skip
    x = rest[win]
    ax[0].plot(x, desi(o.wavelength[win], o.flux[win]), color="0.2", lw=0.7, label="DESI")
    ax[0].plot(x, desi(o.wavelength[win], s1[win]), color="#1b7837", lw=0.9,
               label=f"n free ({th_free[i_n]:.2f})")  # fmt: skip
    ax[0].plot(x, desi(o.wavelength[win], s0[win]), color="#b2182b", lw=0.9, label="n = 0")
    ax[0].plot(x, desi(o.wavelength[win], c1[win]), color="#1b7837", lw=0.7, ls=":",
               label="n free continuum")  # fmt: skip
    ax[0].set_ylim(np.nanpercentile(desi(o.wavelength[win], o.flux[win]), [0.5, 97]) * [0.8, 1.15])
    ax[0].legend(frameon=False, fontsize=7, ncol=2)
    ax[0].set_ylabel(r"flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    for name, s, col in (("n free", s1, "#1b7837"), ("n = 0", s0, "#b2182b")):
        r = 100 * (o.flux - s) / c1
        ax[1].plot(x, r[win], color=col, lw=0.4, alpha=0.35)
        xb, rb = binned(x, r[win], *WINDOW, BIN)
        ax[1].step(xb, rb, where="mid", color=col, lw=1.2, label=f"data - {name}, {BIN:.0f} A bins")
    ax[1].axhline(0, color="0.3", lw=0.6)
    ax[1].set_ylim(-8, 8)
    ax[1].set_ylabel(r"residual [\% of cont.]")
    ax[1].legend(frameon=False, fontsize=7)
    ax[2].plot(x, np.cumsum((chi_0 - chi_free)[win]), color="k", lw=1.0)
    ax[2].set_ylabel(r"running $\Delta\chi^2$")
    ax[2].set_xlabel(r"rest wavelength (vacuum) [$\AA$]")
    for k, lam in MARKS.items():
        for a in ax:
            a.axvline(lam * VAC, color="0.6", lw=0.5, ls="--", zorder=0)
        ax[0].text(lam * VAC, 1.0, k, rotation=90, fontsize=5, va="top", ha="right",
                   transform=ax[0].get_xaxis_transform())  # fmt: skip
    for a in ax:
        a.axvspan(*BUMP, color="0.9", zorder=-1)
    path = args.state.with_name("he2_window.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
