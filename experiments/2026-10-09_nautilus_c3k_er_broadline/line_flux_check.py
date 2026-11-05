"""Split the line-window misfit of the converged ER fit into a flux part and a profile part.

QUESTION: is the [OIII]5007 misfit of the #14 ER fit of 39627770174637084 (rms chi 3.0 in the
analysis) a wrong total flux, which C/O and the other Cue parameters can change, or a wrong line
shape, which they cannot?

For each rest-frame window, on the fitted pixels and at the maximum likelihood spectrum:

- R = sum(data - model) dlam, its noise sqrt(sum unc^2) dlam, and R as a fraction of the model
  line flux. The model line flux is the model minus a straight line through the median model in
  two 6 A sidebands. For Balmer lines that baseline does not remove the stellar absorption, so the
  fraction is a lower bound on the size of the line there; R itself does not depend on the baseline;
- the window chi^2, and the chi^2 after the best single rescale a of the model line
  (data - baseline = a (model - baseline)). The drop is the flux part, the rest the profile part;
- the 16-84 percent range of the model line flux over the 100 posterior draws of predict.py.

INPUTS: the ``_pred.npz`` of predict.py. SEED: none. No FSPS needed.

Run from the repository root with
``uv run python experiments/2026-10-09_nautilus_c3k_er_broadline/line_flux_check.py``.

OUTPUT: results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/line_flux_check.txt.
"""

import numpy as np

from hubersed.paths import PATHS

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
STEM = f"{TID}_er_broad_shared_split45_rest9000_seed0"
SIDE = 6.0  # rest-frame sideband width in A

# rest-frame vacuum windows, as in jacobian.py
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


def baseline(rest, spec, good, a, b):
    """Straight line through the median of spec in the sidebands left of a and right of b."""
    left = good & (rest >= a - SIDE) & (rest < a)
    right = good & (rest > b) & (rest <= b + SIDE)
    if left.sum() < 2 or right.sum() < 2:
        return None
    x = np.array([np.median(rest[left]), np.median(rest[right])])
    y = np.array([np.median(spec[left]), np.median(spec[right])])
    return np.interp(rest, x, y)


def main():
    d = np.load(OUT / f"{STEM}_pred.npz")
    z = float(d["z"])
    wave, flux, unc, good = d["wave"], d["flux"], d["unc"], d["good"]
    model, draws = d["sp_best"], d["draw_spec"]
    rest = wave / (1 + z)
    dlam = np.gradient(wave)

    rows = []
    for name, (a, b) in WINDOWS.items():
        m = good & (rest >= a) & (rest < b)
        base = baseline(rest, model, good, a, b)
        if m.sum() < 3 or base is None:
            rows.append(f"{name:>12s}  too few fitted pixels")
            continue
        resid = (flux - model)[m]
        r = np.sum(resid * dlam[m])
        r_err = np.sqrt(np.sum((unc[m] * dlam[m]) ** 2))
        line = (model - base)[m]
        f_line = np.sum(line * dlam[m])
        chi2 = np.sum((resid / unc[m]) ** 2)
        w = 1 / unc[m] ** 2
        scale = np.sum(w * line * (flux - base)[m]) / np.sum(w * line**2)
        chi2_scaled = np.sum((((flux - base)[m] - scale * line) / unc[m]) ** 2)
        f_draws = np.array([np.sum((s - base)[m] * dlam[m]) for s in draws])
        lo, hi = np.percentile(f_draws, [16, 84])
        rows.append(
            f"{name:>12s} {m.sum():4d} {r / r_err:+8.2f} {r / f_line:+8.3f} {scale:7.3f} "
            f"{chi2 / m.sum():8.2f} {chi2_scaled / m.sum():8.2f} "
            f"{100 * (chi2 - chi2_scaled) / chi2:7.1f} {(hi - lo) / 2 / abs(f_line):8.3f}"
        )

    head = (
        "Line-window misfit at the maximum likelihood spectrum of the ER fit.\n"
        "R/err: integrated data - model over its noise. R/F: as a fraction of the model line flux\n"
        "(model minus sideband baseline). scale: best single rescale of the model line.\n"
        "chi2/px before and after that rescale; flux%: share of the window chi2 removed by the\n"
        "rescale (the rest is profile). draw_sd/F: half the 16-84% range of the model line flux\n"
        "over 100 posterior draws, as a fraction of F.\n\n"
        f"{'window':>12s} {'npix':>4s} {'R/err':>8s} {'R/F':>8s} {'scale':>7s} {'chi2/px':>8s} "
        f"{'scaled':>8s} {'flux%':>7s} {'draw_sd/F':>9s}"
    )
    text = head + "\n" + "\n".join(rows) + "\n"
    (OUT / str(TID) / "line_flux_check.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
