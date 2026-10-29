"""Save the max-L point of the C3K_HR afefree nautilus run of 39633433328091196.

fit.py fixes the SFH hyperparameters there and starts from it. The checkpoint is
results/2026-10-08_continuum_outlier/39633433328091196_c3k_hr_full_neboff_afefree_seed0.h5
(window full, nebular off, degraded data, afe free). It is read as predict.py does: the
likelihood is never called, and the unit-cube point with the largest log L goes through the
prior transform of the 10-08 make_model. The likelihood at that point is not recomputed, since
that needs the C3K_HR AFE build.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-29_continuum_39633433328091196_map/hypers.py``.
OUTPUT: results/2026-10-29_continuum_39633433328091196_map/hr_maxl.npz.
"""

import importlib.util
from pathlib import Path

import numpy as np
from nautilus import Sampler

from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS

TID = 39633433328091196
RUN = PATHS["RESULTS"] / "2026-10-08_continuum_outlier" / f"{TID}_c3k_hr_full_neboff_afefree"
OUT = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map"

# the 10-08 fit.py, loaded by path since this folder has its own fit.py
_spec = importlib.util.spec_from_file_location(
    "fit_1008", Path(__file__).resolve().parents[1] / "2026-10-08_continuum_outlier" / "fit.py"
)
fit_1008 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_1008)


def main():
    """Read the checkpoint and save the max-L point."""
    z = float(load_spectrum(TID).redshift.value)
    model = fit_1008.make_model(z, False, "free")
    s = Sampler(
        lambda u: u,
        lambda x: 0.0,
        n_dim=model.ndim,
        n_live=1000,
        filepath=f"{RUN}_seed0.h5",
        resume=True,
    )
    assert s.explored, RUN
    unit, log_w, log_l = s.posterior()
    best = model.prior_transform(unit[np.argmax(log_l)])
    labels = model.theta_labels()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "hr_maxl.npz", labels=np.array(labels), best=best, max_l=log_l.max(), z=z)
    print(f"{RUN.name}: {len(unit)} points, N_eff {s.n_eff:.0f}, max log L {log_l.max():.2f}")
    for k, v in zip(labels, best, strict=True):
        print(f"  {k:16s} {v:9.4f}")


if __name__ == "__main__":
    main()
