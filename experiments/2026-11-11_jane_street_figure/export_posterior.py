"""Save the free_dust posterior of logmass, logzsol and gas_logz for the Year 2 panel.

The prior transform needs the model of 2026-10-15_broadline_free_amplitude, which imports modules
removed later (hubersed.sps.broadline). So this runs on the source of commit 2938e32, copied out with
git show, not on the current tree.
INPUTS: pmc_final.state of the free_dust pocoMC run of 39627770174637084. SEED: none.
COMMAND: copy src and the 10-13 and 10-15 experiment folders of 2938e32 to <old> (git show per
file), then from the repository root
``PYTHONPATH=<old>/src:<old>/experiments/2026-10-15_broadline_free_amplitude:<old>/experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-11-11_jane_street_figure/export_posterior.py``.
OUTPUT: results/2026-11-11_jane_street_figure/posterior_free_dust.npz (theta, w, names).
"""

from pathlib import Path

import fit
import numpy as np
from compare_poco import load_sampler

TID = 39627770174637084
RES = Path(__file__).resolve().parents[2] / "results"  # the old PATHS points into <old>
OUT = RES / Path(__file__).resolve().parent.name
STATE = (
    RES
    / "2026-10-15_broadline_free_amplitude"
    / f"poco_{TID}_free_dust_neff1024"
    / "pmc_final.state"
)
EXPORT = RES / "2026-10-18_issue14_figures" / "export" / f"emission_line_{TID}_ubersed.npz"
NAMES = ["logmass", "logzsol", "gas_logz"]


def main():
    """Transform the weighted pocoMC samples and save the three parameters."""
    import hubersed

    print("hubersed from", hubersed.__file__)
    s = load_sampler(STATE)
    usamp, w, _, _ = s.posterior()
    model = fit.make_model(float(np.load(EXPORT)["z"]), "free_dust")
    theta = np.array([model.prior_transform(x) for x in usamp])
    labels = list(model.theta_labels())
    assert model.ndim == usamp.shape[1], (model.ndim, usamp.shape)
    idx = [labels.index(p) for p in NAMES]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "posterior_free_dust.npz", theta=theta[:, idx], w=w / w.sum(), names=NAMES)
    q = [np.quantile(theta[:, i], [0.16, 0.5, 0.84]) for i in idx]
    print({n: np.round(v, 3) for n, v in zip(NAMES, q, strict=True)}, len(w))


if __name__ == "__main__":
    main()
