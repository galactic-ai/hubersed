"""Select the SV3 BGS Bright and Faint parent sample from the DESI DR1 redshift catalog.

Cuts follow the fixed spender loader: COADD_FIBERSTATUS = 0, SV3_DESI_TARGET != 0, ZWARN = 0,
SPECTYPE = GALAXY, Z < 0.8, then the BGS_BRIGHT or BGS_FAINT bit and the Hahn et al. (2023,
arXiv:2208.08512) redshift success and range, ZERR < 0.0005 (1 + z), DELTACHI2 > 40, 0 < z < 0.6.
INPUTS: data/desi_dr1/zpix-sv3-bright.fits, DR1 iron zcatalog/v1, sha256 85e292b8...194ced.
COMMAND: ``uv run python experiments/2026-11-20_bgs_parent_selection/selection.py``.
OUTPUT: results/2026-11-20_bgs_parent_selection/parent_bgs.npz and cutflow.txt.
"""

import numpy as np
from astropy.io import fits
from desitarget.sv3.sv3_targetmask import bgs_mask

from hubersed.paths import PATHS

CATALOG = PATHS["DATA"] / "desi_dr1" / "zpix-sv3-bright.fits"
OUT = PATHS["RESULTS"] / "2026-11-20_bgs_parent_selection"
COLUMNS = (
    "TARGETID", "HEALPIX", "COADD_FIBERSTATUS", "SV3_DESI_TARGET", "SV3_BGS_TARGET", "ZWARN",
    "SPECTYPE", "Z", "ZERR", "DELTACHI2",
)  # fmt: skip


def cuts(c):
    """Return the named cuts in the order they are applied."""
    z = c["Z"]
    bgs = bgs_mask["BGS_BRIGHT"] | bgs_mask["BGS_FAINT"]
    return {
        "COADD_FIBERSTATUS = 0": c["COADD_FIBERSTATUS"] == 0,
        "SV3_DESI_TARGET != 0": c["SV3_DESI_TARGET"] != 0,
        "ZWARN = 0": c["ZWARN"] == 0,
        "SPECTYPE = GALAXY": np.char.strip(c["SPECTYPE"].astype(str)) == "GALAXY",
        "Z < 0.8": z < 0.8,
        "BGS_BRIGHT or BGS_FAINT": (c["SV3_BGS_TARGET"] & bgs) != 0,
        "ZERR < 0.0005 (1 + z)": c["ZERR"] < 0.0005 * (1 + z),
        "DELTACHI2 > 40": c["DELTACHI2"] > 40,
        "0 < z < 0.6": (z > 0) & (z < 0.6),
    }


def main():
    """Apply the cuts, write the cut flow and the selected TARGETIDs."""
    OUT.mkdir(parents=True, exist_ok=True)
    with fits.open(CATALOG, memmap=True) as h:
        d = h["ZCATALOG"].data
        c = {k: np.asarray(d[k]) for k in COLUMNS}
    n = len(c["TARGETID"])
    assert np.unique(c["TARGETID"]).size == n, "TARGETID is not unique in the catalog"

    keep = np.ones(n, bool)
    first_fail = np.full(n, "", dtype=object)
    lines = [f"{'cut':28s} {'alone':>8s} {'cumulative':>11s}", f"{'all rows':28s} {n:8d} {n:11d}"]
    for name, m in cuts(c).items():
        first_fail[keep & ~m] = name
        keep &= m
        lines.append(f"{name:28s} {int(m.sum()):8d} {int(keep.sum()):11d}")
    text = "\n".join(lines)
    (OUT / "cutflow.txt").write_text(text + "\n")
    print(text)

    order = np.argsort(c["TARGETID"][keep])
    np.savez(
        OUT / "parent_bgs.npz",
        targetid=c["TARGETID"][keep][order],
        healpix=c["HEALPIX"][keep][order],
        z=c["Z"][keep][order],
        catalog_targetid=c["TARGETID"],
        catalog_first_fail=first_fail.astype(str),
    )
    print(f"saved {int(keep.sum())} TARGETIDs to {OUT / 'parent_bgs.npz'}")


if __name__ == "__main__":
    main()
