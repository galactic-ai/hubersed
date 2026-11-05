"""Export the data and both best fits of 39627770174637084 as npz and txt, for sharing.

One pair of files per fit, in the layout of the earlier outlier exports (emission_line_only_*.txt):
wave_obs, wave_rest, flux, unc, mask, model, medfilt9, chi, plus chi2, ndof, chi2_red, z, targetid
and the best-fit theta.
1. MILES + Cue, 2026-09-24_nautilus_tauin: DR1 coadd in the MILES window.
   - Max-likelihood point from 2026-09-25_nautilus_tauin_c3k_hr/<TID>_MILES_plotdata.npz.
   - Spectrum from miles_0924.npz (miles.py, the 09-24 environment; the current code is 6.6% off).
2. UberSED, the pocoMC free_dust run of 2026-10-15_broadline_free_amplitude.
   - Model: C3K ER, free-line broad components, Cue, resolution matrix, recoadd ivar.
   - Highest-posterior particle and spectrum from analysis.py poco_row(), as in the #14 figures.
Flux, unc, model and medfilt9 are in 1e-6 maggies. Pixels outside a fit's mask are NaN except the
model. chi2_red = chi2 / (fitted pixels - free parameters).
COMMAND: as analysis.py, ``PYTHONPATH=<er_path>:experiments/2026-10-15_broadline_free_amplitude:
experiments/2026-10-13_nautilus_resolution_matrix SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync
python experiments/2026-10-18_issue14_figures/export.py``.
OUTPUT: results/2026-10-18_issue14_figures/export/emission_line_39627770174637084_{miles_cue,ubersed}
.{npz,txt}.
"""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.signal import medfilt

from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
OUT = PATHS["RESULTS"] / HERE.name
TID = 39627770174637084


def load(name, path):
    """Load the file at ``path`` as module ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def theta_dict(labels, best):
    """Group a flat theta by parameter name, joining vector parameters such as logsfr_ratios."""
    out = {}
    for name, v in zip(labels, best, strict=True):
        base = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else name
        out.setdefault(base, []).append(float(v))
    return {k: np.array(v, np.float32) for k, v in out.items()}


def write(tag, fit_line, wave, z, flux, unc, mask, model, n_free, theta):
    """Write the npz and txt of one fit."""
    mask = np.asarray(mask, bool) & np.isfinite(unc) & (unc > 0)
    f, e, m = 1e6 * flux, 1e6 * unc, 1e6 * np.asarray(model, float)
    med = np.full(f.size, np.nan)
    med[mask] = medfilt(f[mask], 9)
    chi = np.where(mask, (f - m) / np.where(mask, e, 1.0), np.nan)
    chi2 = float(np.nansum(chi**2))
    ndof = int(mask.sum() - n_free)
    rest = wave / (1 + z)
    cols = dict(wave_obs=wave.astype(np.float32), wave_rest=rest.astype(np.float32),
                flux=np.where(mask, f, np.nan).astype(np.float32),
                unc=np.where(mask, e, np.nan).astype(np.float32), mask=mask, model=m,
                medfilt9=med.astype(np.float32), chi=chi.astype(np.float32))  # fmt: skip
    path = OUT / "export" / f"emission_line_{TID}_{tag}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), **cols, chi2=chi2, ndof=ndof, chi2_red=chi2 / ndof, z=z,
             targetid=TID, theta_names=np.array(list(theta)),
             **{f"theta_{k}": v for k, v in theta.items()})  # fmt: skip
    th = "; ".join(f"{k}={np.array2string(v, precision=5)}" for k, v in theta.items())
    head = [f"TARGETID {TID}  z {z:.6f}", f"chi2 {chi2:.3f}  ndof {ndof}  chi2_red {chi2 / ndof:.4f}",
            f"fit: {fit_line}", "flux, unc, model, medfilt9 in 1e-6 maggies; masked pixels are nan except model",
            f"best theta: {th}", " ".join(cols)]  # fmt: skip
    rows = np.column_stack([cols[k].astype(float) for k in cols])
    np.savetxt(path.with_suffix(".txt"), rows, fmt="%.6g", header="\n".join(head))
    print(f"{tag}: chi2_red {chi2 / ndof:.4f} on {int(mask.sum())} px, wrote {path}.npz/.txt")


def main():
    """Write both exports."""
    from hubersed.fitting.chi2 import WAVE_OBS

    an = load("issue14_analysis", HERE / "analysis.py")
    m = np.load(OUT / "miles_0924.npz")
    pd = np.load(
        PATHS["RESULTS"] / "2026-09-25_nautilus_tauin_c3k_hr" / f"{TID}_MILES_plotdata.npz"
    )
    labels = [str(x) for x in pd["labels"]]
    write("miles_cue", "MILES + Cue nautilus, 2026-09-24_nautilus_tauin, max likelihood (DR1 coadd, MILES window)",
          np.asarray(WAVE_OBS, float), float(m["z"]), m["flux"], m["unc"], m["mask"], m["model"],
          int(m["n_free"]), theta_dict(labels, pd["best"]))  # fmt: skip
    model, _, _, best, row = an.poco_row()
    write("ubersed", "UberSED pocoMC free_dust (C3K ER + free-line broad + Cue + resolution matrix + recoadd "
          "ivar), 2026-10-15_broadline_free_amplitude, highest-posterior particle",
          row["wave"], row["z"], row["flux"], row["unc"], row["mask"], row["model"], row["n_free"],
          theta_dict(list(model.theta_labels()), best))  # fmt: skip


if __name__ == "__main__":
    main()
