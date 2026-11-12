"""Save the SV3 bright rows of the EmFit DR1 VAC v2.3.1 with the line fluxes analysis.py uses.

Replaces fetch_emfit.py when Data Lab is unavailable: same columns and output, read from the VAC
file (data.desi.lbl.gov public/dr1/vac/dr1/emfit/v2.3/emfit-dr1-v2.3.1.fits, sha256 checked),
column names lower-cased to match the Data Lab table.
COMMAND: on stampede3 through extract_emfit.slurm, ``python extract_emfit.py <vac.fits> <out.fits>``.
OUTPUT: <out.fits>, copied to results/2026-11-06_line_ew_prior/emfit_sv3_bright.fits.
"""

import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table

FLUX = (
    "hb_n hb_out hb_b oiii4959 oiii4959_out oiii5007 oiii5007_out nii6548 nii6548_out nii6583 "
    "nii6583_out ha_n ha_out ha_b sii6716 sii6716_out sii6731 sii6731_out"
).split()


def main(vac, out):
    """Select survey sv3, program bright and write the flux columns."""
    cols = ["TARGETID", "Z"] + [f"{c.upper()}_FLUX" for c in FLUX]
    with fits.open(vac, memmap=True) as h:
        d = h["EMFIT"].data
        survey = np.char.strip(np.asarray(d["SURVEY"]).astype(str))
        program = np.char.strip(np.asarray(d["PROGRAM"]).astype(str))
        sel = (survey == "sv3") & (program == "bright")
        tab = Table({c.lower(): np.asarray(d[c])[sel] for c in cols})
    tab.write(out, overwrite=True)
    print(f"{sel.sum()} of {sel.size} rows (sv3 bright) -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
