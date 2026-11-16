"""Keep the DR1 CIGALE VAC rows of the SV3 BGS bright galaxies, for the UVJ diagram.

INPUTS: data/cigale_dr1/IronPhysProp_v1.2.fits and data/fastspec-iron-sv3-bright.fits.
COMMAND: ``uv run --no-sync python experiments/2026-10-30_uvj_sv3_bright/extract.py``.
OUTPUT: data/cigale_dr1/cigale_sv3_bright.npz.
"""

import numpy as np
from astropy.io import fits

from hubersed.paths import get_paths

KEEP = ["TARGETID", "SURVEY", "PROGRAM", "Z", "CHI2", "LOGM", "LOGM_ERR", "LOGSFR", "AGNFRAC",
        "UV", "VJ", "UV_ERR", "VJ_ERR", "FLAG_MASSPDF", "FLAGOPTICAL", "FLAGINFRARED"]  # fmt: skip
PATHS = get_paths()
VAC = PATHS["DATA"] / "cigale_dr1" / "IronPhysProp_v1.2.fits"
OUT = PATHS["DATA"] / "cigale_dr1" / "cigale_sv3_bright.npz"

ids = np.asarray(
    fits.getdata(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", "METADATA")["TARGETID"]
)
with fits.open(VAC, memmap=True) as h:
    data = h[1].data
    sel = np.isin(np.asarray(data["TARGETID"]), ids)
    out = {}
    for k in KEEP:
        col = np.asarray(data[k][sel])
        out[k] = np.char.strip(col.astype(str)) if col.dtype.kind in "SU" else col
_, n = np.unique(out["TARGETID"], return_counts=True)
assert n.max() == 1, "a TARGETID has more than one CIGALE row"
np.savez(OUT, **out)
print(f"wrote {OUT}: {sel.sum()} of {len(ids)} SV3 bright TARGETIDs")
