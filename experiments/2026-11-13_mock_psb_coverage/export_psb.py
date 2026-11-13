"""Save the DR1 SV3 bright coadds of the gallery105 PSBs for indices.py on stampede3.

The PSBs are the rows of 2026-11-13_gallery105/review.tsv with review_class psb. Spectra come from
load_by_index (the coadds the flows scored), on the DESI grid.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_mock_psb_coverage/export_psb.py`` (Mac).
OUTPUT: results/2026-11-13_mock_psb_coverage/psb_spectra.npz (nn, targetid, z, wave, flux in
1e-17 erg/s/cm2/A, ivar).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-13_mock_psb_coverage"
REVIEW = Path(__file__).resolve().parents[1] / "2026-11-13_gallery105" / "review.tsv"


def main():
    """Load and save the PSB coadds."""
    OUT.mkdir(parents=True, exist_ok=True)
    rev = pd.read_csv(REVIEW, sep="\t")
    tab = pd.read_csv(PATHS["RESULTS"] / "2026-11-13_gallery105" / "gallery105.tsv", sep="\t")
    psb = rev[rev["review_class"] == "psb"].merge(tab[["nn", "target_id"]], on="nn")
    flux, ivar, zs = [], [], []
    for t in psb["target_id"]:
        f, iv, z, _ = load_by_index(int(tids_to_indices(np.array([t]))[0]))
        flux.append(np.asarray(f, float))
        ivar.append(np.asarray(iv, float))
        zs.append(float(z))
    np.savez(OUT / "psb_spectra.npz", nn=psb["nn"].to_numpy(), targetid=psb["target_id"].to_numpy(),
             z=np.array(zs), wave=DESI._wave_obs.numpy(), flux=np.array(flux), ivar=np.array(ivar))  # fmt: skip
    print(f"saved {len(psb)} PSBs to {OUT / 'psb_spectra.npz'}")


if __name__ == "__main__":
    main()
