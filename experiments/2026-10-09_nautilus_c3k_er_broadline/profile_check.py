"""Fit simple line profiles to the strong lines of the ER max-L model to find the core misfit.

QUESTION: are the line-core residuals of the ER fit of 39627770174637084 at [OIII]5007,
[OIII]4959, Halpha and Hbeta caused by an instrumental line spread that is too broad, by a line
shape that two Gaussians per line cannot represent, or by velocity offsets of the line centres?

The model adds each line as a Gaussian sampled at the pixel centres, with the intrinsic width
added in quadrature to the DESI instrumental sigma at the line (prospect
SpecModel.cache_eline_parameters, and TwoCompLineModel._profile_widths in
hubersed.sps.broadline). This script builds the same profiles.

The background under each line is the max-L model minus that line's own model contribution. It
holds the stellar and nebular continuum and the wings of the neighbouring lines. The line
contributions come from the fixed-line spectrum prospect caches in predict_spec. On data minus
background, with the pixel uncertainties and the fitted pixels of each rest-frame window, the
script fits

- A, one Gaussian with free flux, velocity and intrinsic sigma;
- B, one Gaussian with intrinsic sigma 10 km/s and a free scale s on the instrumental sigma;
- C1, two Gaussians with one centre and free fluxes and intrinsic sigmas;
- C2, two Gaussians with separate free centres;
- C1s, as C1 but with the narrow intrinsic sigma fixed at 10 km/s and a free scale s on the
  instrumental sigma of both components. This one is B with the broad wings added.

Velocities are relative to the centre of the max-L model's narrow component of the line.

INPUTS: the ``_pred.npz`` of predict.py (data, fitted pixels, max-L point and spectrum). SEED:
none (deterministic least squares from fixed starts).

COMMAND: from the repository root on the Mac with the C3K_ER build (no AFE_FLAG), as jacobian.py,
``PYTHONPATH=<site with the C3K_ER fsps>:experiments/2026-10-09_nautilus_c3k_er_broadline
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/profile_check.py``. Building the Cue source takes
a few minutes.

OUTPUT: profile_check.txt and profile_check.png in
results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/.
"""

import matplotlib.pyplot as plt
import numpy as np
from fit import process_state
from scipy.optimize import least_squares

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39627770174637084
FB, SPLIT, REST_MAX = "shared", 45.0, 9000.0
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
STEM = f"{TID}_er_broad_{FB}_split{SPLIT:g}_rest{REST_MAX:g}_seed0"
C_KMS = 299792.458
SIG_FIX = 10.0  # intrinsic sigma of B and of the C1s narrow component, in km/s
CORE = 150.0  # half width in km/s of the core pixels listed in the text output

# rest-frame vacuum windows, as in jacobian.py, with the Cue name of the line in each
WINDOWS = {
    "[OIII]5007": ((4995, 5022), "[O III] 5007"),
    "[OIII]4959": ((4950, 4970), "[O III] 4959"),
    "Halpha": ((6556, 6572), "Ba-alpha 6563"),
    "Hbeta": ((4845, 4880), "Ba-beta 4861"),
}


def model_lines(best, sp_best):
    """Return the max-L spectrum and each window line's contribution, centre and instrument.

    Parameters
    ----------
    best : np.ndarray
        The max-L parameter vector.
    sp_best : np.ndarray
        The max-L spectrum stored by predict.py, used as a check.

    Returns
    -------
    pred : np.ndarray
        The max-L spectrum in maggies on the full grid.
    lines : dict
        For each window, a tuple of the line's model spectrum on the full grid, the observed
        centre of its narrow component in Angstrom and the instrumental sigma in km/s.
    """
    model, obs, sps = process_state(TID, FB, SPLIT, REST_MAX)
    pred = model.predict(best, observations=obs, sps=sps)[0][0]
    assert np.allclose(pred, sp_best, rtol=1e-3), "the model does not reproduce sp_best"
    # no calibration vector and no fitted lines, so the lines are the cached fixed-line spectrum
    assert np.all(np.asarray(model._speccal) == 1.0)
    assert not np.any(model._fit_eline & model._valid_eline)
    inds = np.where(model._fix_eline & model._valid_eline)[0]
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))[inds]
    inst = np.sqrt(model._profile_widths()[3])[inds]
    emask = model._fix_eline_pixelmask
    lines = {}
    for key, (_, lname) in WINDOWS.items():
        j = int(np.where(names == lname)[0][0])
        spec = np.zeros_like(pred)
        spec[emask] = model._fix_eline_spec[:, j]
        lines[key] = (spec, float(model._ewave_obs[inds[j]]), float(inst[j]))
    return pred, lines


def profile(wave, grid, mu, sigma):
    """Return a unit-flux Gaussian line sampled at ``wave``, as prospect builds it.

    The Gaussian is in velocity with the d(v)/d(nu) factor of prospect's get_eline_gaussians,
    and it is normalized by its trapezoid integral on the full pixel grid ``grid``.
    """

    def g(w):
        return np.exp(-0.5 * (C_KMS * (w / mu - 1) / sigma) ** 2) * (w / mu) ** 2

    return g(wave) / np.trapezoid(g(grid), grid)


def fit_kinds(w, grid, mu0, inst, f0):
    """Return the profile function, start points and bounds of each fit.

    Parameters are fluxes in maggies Angstrom, velocities in km/s, intrinsic sigmas in km/s and
    scales of the instrumental sigma.
    """

    def p(f, v, sigma):
        return f * profile(w, grid, mu0 * (1 + v / C_KMS), sigma)

    inf = np.inf
    return {
        "A": (
            lambda t: p(t[0], t[1], np.hypot(t[2], inst)),
            [[f0, 0, s] for s in (1, 10, 30, 60)],
            ([-inf, -100, 0], [inf, 100, 500]),
        ),
        "B": (
            lambda t: p(t[0], t[1], np.hypot(SIG_FIX, t[2] * inst)),
            [[f0, 0, s] for s in (0.5, 0.8, 1.0, 1.3)],
            ([-inf, -100, 0.05], [inf, 100, 5]),
        ),
        "C1": (
            lambda t: p(t[0], t[2], np.hypot(t[3], inst)) + p(t[1], t[2], np.hypot(t[4], inst)),
            [[0.8 * f0, 0.2 * f0, 0, a, b] for a in (1, 15) for b in (50, 100, 200)],
            ([-inf, -inf, -100, 0, 0], [inf, inf, 100, 1000, 1000]),
        ),
        "C2": (
            lambda t: p(t[0], t[2], np.hypot(t[4], inst)) + p(t[1], t[3], np.hypot(t[5], inst)),
            [
                [0.8 * f0, 0.2 * f0, 0, vb, a, b]
                for vb in (-50, 0, 50)
                for a in (1, 15)
                for b in (50, 150)
            ],
            ([-inf, -inf, -150, -400, 0, 0], [inf, inf, 150, 400, 1000, 1000]),
        ),
        "C1s": (
            lambda t: (
                p(t[0], t[2], np.hypot(SIG_FIX, t[4] * inst))
                + p(t[1], t[2], np.hypot(t[3], t[4] * inst))
            ),
            [[0.8 * f0, 0.2 * f0, 0, b, s] for b in (50, 100) for s in (0.7, 1.0)],
            ([-inf, -inf, -100, 0, 0.05], [inf, inf, 100, 1000, 5]),
        ),
    }


def fit_window(w, y, e, grid, mu0, inst, f0):
    """Fit every profile of ``fit_kinds`` to one window and keep the best start of each.

    Returns
    -------
    dict
        For each fit, the best parameters, their 1 sigma errors from the column-scaled Jacobian
        (a parameter at a bound gets a meaningless error), the chi^2 and the fitted profile.
    """
    out = {}
    for kind, (f, starts, bounds) in fit_kinds(w, grid, mu0, inst, f0).items():
        best = None
        for s0 in starts:
            r = least_squares(lambda t, f=f: (y - f(t)) / e, s0, bounds=bounds, x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
        # scale the Jacobian columns first, since fluxes (1e-6) and velocities (1) differ a lot
        norm = np.linalg.norm(best.jac, axis=0)
        norm[norm == 0] = 1.0
        js = best.jac / norm
        err = np.sqrt(np.abs(np.diag(np.linalg.pinv(js.T @ js)))) / norm
        out[kind] = (best.x, err, 2 * best.cost, f(best.x))
    return out


def describe(d, pred, lines):
    """Run the fits in every window and return the results and the text report."""
    z = float(d["z"])
    wave, flux, unc, good = d["wave"], d["flux"], d["unc"], d["good"]
    rest = wave / (1 + z)
    res = {}
    rows, pattern = [], []
    for key, ((a, b), _) in WINDOWS.items():
        line, mu0, inst = lines[key]
        m = good & (rest >= a) & (rest < b)
        w, e = wave[m], unc[m]
        y = (flux - (pred - line))[m]
        f0 = np.trapezoid(line, wave)
        fits = fit_window(w, y, e, wave, mu0, inst, f0)
        chi2_ml = float(np.sum(((flux - pred)[m] / e) ** 2))
        res[key] = dict(w=w, y=y, e=e, mu0=mu0, ml=line[m], chi2_ml=chi2_ml, fits=fits)

        xa, ea, c2a, _ = fits["A"]
        xb, eb, c2b, _ = fits["B"]
        x1, _, c21, _ = fits["C1"]
        x2, _, c22, _ = fits["C2"]
        xs, es, c2s, _ = fits["C1s"]
        nar_s = np.hypot(SIG_FIX, xs[4] * inst)
        rows.append(
            f"{key:>11s} {m.sum():4d} {inst:6.2f} {chi2_ml:8.1f} {c2a:8.1f} {c2b:8.1f} "
            f"{c21:8.1f} {c22:8.1f} {c2s:8.1f} {xa[2]:5.1f}+-{ea[2]:3.1f} {xb[2]:5.3f}+-{eb[2]:5.3f} "
            f"{xs[4]:5.3f}+-{es[4]:5.3f} {nar_s:6.1f} {xa[1]:+5.2f}+-{ea[1]:4.2f} {x1[2]:+6.2f}"
        )
        fb1 = x1[1] / (x1[0] + x1[1])
        fb2 = x2[1] / (x2[0] + x2[1])
        pattern.append(
            f"{key}: C1 narrow sigma_int {x1[3]:.1f}, broad sigma_int {x1[4]:.1f}, broad flux "
            f"fraction {fb1:.3f}. C2 narrow v {x2[2]:+.2f} sigma_int {x2[4]:.1f}, broad v "
            f"{x2[3]:+.2f} sigma_int {x2[5]:.1f}, broad fraction {fb2:.3f}. C1s broad sigma_int "
            f"{xs[3]:.1f}, broad fraction {xs[1] / (xs[0] + xs[1]):.3f}, v {xs[2]:+.2f}."
        )
        dv = (w / mu0 - 1) * C_KMS
        core = np.abs(dv) < CORE
        pattern.append("  dv [km/s]   " + " ".join(f"{v:+6.0f}" for v in dv[core]))
        pattern.append("  chi max-L   " + " ".join(f"{c:+6.1f}" for c in ((y - line[m]) / e)[core]))
        for kind in ("A", "C1", "C2", "C1s"):
            c = (y - fits[kind][3]) / e
            pattern.append(f"  chi {kind:7s} " + " ".join(f"{v:+6.1f}" for v in c[core]))
    head = (
        "Profile fits to data minus background at the ER max-L point of 39627770174637084.\n"
        "Background is the max-L model minus the line's own contribution. Fitted pixels only.\n"
        "inst is the instrumental sigma the fit uses at the line, in km/s. chi2 columns are for\n"
        "the max-L model and the fits A, B, C1, C2 and C1s (see the script docstring).\n"
        "sig_A is the intrinsic sigma of A. s_B and s_C1s scale the instrumental sigma, with\n"
        f"intrinsic {SIG_FIX:g} km/s on the (narrow) component. nar_C1s is the total narrow sigma\n"
        "of C1s in km/s. v_A and v_C1 are line centres relative to the max-L narrow centre.\n\n"
        f"{'window':>11s} {'npix':>4s} {'inst':>6s} {'max-L':>8s} {'A':>8s} {'B':>8s} "
        f"{'C1':>8s} {'C2':>8s} {'C1s':>8s} {'sig_A':>10s} {'s_B':>13s} {'s_C1s':>13s} "
        f"{'nar_C1s':>6s} {'v_A':>12s} {'v_C1':>6s}"
    )
    text = (
        head
        + "\n"
        + "\n".join(rows)
        + "\n\nComponent fits and the chi of the pixels within "
        + f"{CORE:g} km/s of the line centre\n"
        + "\n".join(pattern)
        + "\n"
    )
    return res, text


def plot(res, path):
    """Plot, per window, the line data, the max-L line and the A, C1, C2 and C1s fits, with chi below."""
    use_apj_style()
    fig, ax = plt.subplots(
        2,
        len(res),
        figsize=(4 * len(res), 5.5),
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05},
        sharex="col",
    )
    styles = {
        "max-L": ("#b2182b", "-"),
        "A": ("#2166ac", "--"),
        "C1": ("#1b7837", ":"),
        "C2": ("#762a83", "-."),
        "C1s": ("#e08214", "-"),
    }
    for i, (key, r) in enumerate(res.items()):
        dv = (r["w"] / r["mu0"] - 1) * C_KMS
        curves = {"max-L": (r["ml"], r["chi2_ml"])}
        curves.update({k: (r["fits"][k][3], r["fits"][k][2]) for k in ("A", "C1", "C2", "C1s")})
        ax[0, i].errorbar(dv, r["y"], r["e"], fmt="o", ms=3, color="0.3", lw=0.8, label="data")
        for k, (prof, c2) in curves.items():
            col, ls = styles[k]
            ax[0, i].plot(dv, prof, color=col, ls=ls, lw=1.0, label=rf"{k}, $\chi^2$ = {c2:.0f}")
            ax[1, i].plot(dv, (r["y"] - prof) / r["e"], color=col, ls=ls, lw=1.0, marker=".")
        ax[0, i].set_title(key)
        ax[0, i].legend(frameon=False, fontsize=7)
        for h in (-2, 2):
            ax[1, i].axhline(h, color="0.5", ls="--", lw=0.6)
        ax[1, i].axhline(0, color="0.3", lw=0.6)
        ax[1, i].set_xlabel(r"velocity from max-L narrow centre [km s$^{-1}$]")
        ax[1, i].set_xlim(-400, 400)
    ax[0, 0].set_ylabel("data minus background [maggies]")
    ax[1, 0].set_ylabel(r"$\chi$")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    d = np.load(OUT / f"{STEM}_pred.npz")
    pred, lines = model_lines(d["best"], d["sp_best"])
    res, text = describe(d, pred, lines)
    (OUT / str(TID) / "profile_check.txt").write_text(text)
    plot(res, OUT / str(TID) / "profile_check.png")
    print(text)


if __name__ == "__main__":
    main()
