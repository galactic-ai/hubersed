"""Check the Balmer decrement of the free-line fit of 39627770174637084 at the pocoMC best point.

QUESTION: at n = 0 the mask refits put dust_ratio at 0. Is the model's Halpha/Hbeta right?

Prints Cue's intrinsic Halpha/Hbeta, the model's attenuated ratio, the ratio of data minus model
continuum in clean windows (Hbeta 4855-4870 A, Halpha 6555-6572 A), and the EmFit and FastSpecFit
ratios. The narrow Balmer lines are tied to Cue and the broad ones share one ratio, so the model
ratio is Cue's attenuated ratio.

RESULT: 2026-10-17, balmer_check.txt. Cue intrinsic 2.804 (Case B 2.86), model 3.443, spectrum
3.460, EmFit 3.551 +- 0.077, FastSpecFit 3.363 +- 0.028. The fit matches the decrement to 0.5%, so
the grey slope comes from the continuum, not the lines.

INPUTS: pmc_final.state of poco.py; results/2026-10-14_cue_lineflux_calibration/lineflux_table.fits.
SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude:~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/balmer_check.py --state <path>``.

OUTPUT: balmer_check.txt next to the state file.
"""

import argparse
from pathlib import Path

import fit
import numpy as np
from astropy.table import Table
from mask_refit import best_point

from hubersed.paths import PATHS

WINDOWS = {"Hbeta": (4855.0, 4870.0), "Halpha": (6555.0, 6572.0)}
TABLE = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration" / "lineflux_table.fits"


def balmer_index(model, rest):
    """Return the index of the Balmer line nearest ``rest`` A in the Cue line list (Ba-alpha, Ba-beta)."""
    i = int(np.argmin(np.abs(np.asarray(model.emline_info["wave"]) - rest)))
    assert str(model.emline_info["name"][i]).startswith("Ba-"), model.emline_info["name"][i]
    return i


def main():
    """Print and write the decrement comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    model, obs, cue = fit.process_state("free")
    labels = list(model.theta_labels())
    theta = best_point(args.state, model)
    ia, ib = balmer_index(model, 6564.6), balmer_index(model, 4862.7)
    names = [str(n) for n in model.emline_info["name"]]

    spec = model.predict(theta, observations=obs, sps=cue)[0][0]
    lum = model._eline_lum.copy()
    cont = model._resolved_continuum.copy()
    nodust = theta.copy()
    nodust[labels.index("dust2")] = 0.0
    nodust[labels.index("dust_ratio")] = 0.0
    model.predict(nodust, observations=obs, sps=cue)
    lum0 = model._eline_lum.copy()
    model.predict(theta, observations=obs, sps=cue)

    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)
    data_sum, model_sum = {}, {}
    for k, (a, b) in WINDOWS.items():
        w = (rest >= a) & (rest <= b) & o.mask
        data_sum[k] = np.sum((o.flux - cont)[w])
        model_sum[k] = np.sum((spec - cont)[w])
    model_ratio = lum[ia] / lum[ib]
    data_ratio = (
        model_ratio
        * (data_sum["Halpha"] / model_sum["Halpha"])
        / (data_sum["Hbeta"] / model_sum["Hbeta"])
    )
    row = Table.read(TABLE)
    row = row[row["TARGETID"] == fit.TID][0]
    lines = [
        f"lines: {names[ia]}, {names[ib]}",
        f"Cue intrinsic Halpha/Hbeta (no dust)      {lum0[ia] / lum0[ib]:.3f}  (Case B 10^4 K: 2.86)",
        f"model attenuated Halpha/Hbeta             {model_ratio:.3f}",
        f"data/model in window: Halpha {data_sum['Halpha'] / model_sum['Halpha']:.4f}, "
        f"Hbeta {data_sum['Hbeta'] / model_sum['Hbeta']:.4f}",
        f"spectrum Halpha/Hbeta (model continuum)   {data_ratio:.3f}",
        f"EmFit totals Halpha/Hbeta                 {row['F_HA'] / row['F_HB']:.3f} "
        f"+- {row['F_HA'] / row['F_HB'] * np.hypot(row['E_HA'] / row['F_HA'], row['E_HB'] / row['F_HB']):.3f}",
        f"FastSpecFit Halpha/Hbeta                  {row['FSF_F_HA'] / row['FSF_F_HB']:.3f} "
        f"+- {row['FSF_F_HA'] / row['FSF_F_HB'] * np.hypot(row['FSF_E_HA'] / row['FSF_F_HA'], row['FSF_E_HB'] / row['FSF_F_HB']):.3f}",
        "dust at the best point: "
        + ", ".join(
            f"{k} {theta[labels.index(k)]:.3f}" for k in ("dust2", "dust_ratio", "dust_index")
        ),
    ]
    text = "\n".join(lines)
    print(text)
    args.state.with_name("balmer_check.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
