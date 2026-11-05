"""Save Legacy DR9 Tractor fracflux, type and maskbits for every S/N > 3 latent spectrum in the VAC.

Queries ls_dr9.tractor on the Astro Data Lab TAP service by LS_ID (FastSpecFit VAC), 500 per query.
COMMAND: ``uv run --no-sync python -u experiments/2026-10-31_screen_all/fracflux.py``.
OUTPUT: results/2026-10-31_screen_all/fracflux.csv.
"""

import time
import warnings

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astroquery.utils.tap.core import TapPlus

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-31_screen_all"
BATCH = 500
COLS = "ls_id, type, fracflux_g, fracflux_r, fracmasked_g, maskbits"


def main():
    """Query every LS_ID and write one row per target."""
    warnings.filterwarnings("ignore", module="astroquery")
    with h5py.File(PATHS["DATA"] / "latents" / "spender_spec_cont10latent_snr3.h5", "r") as f:
        tids = f["target_ids"][:].astype(np.int64)
    M = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)["METADATA"].data
    vt = np.asarray(M["TARGETID"], np.int64)
    keep = np.isin(vt, tids)
    tid, lsid = vt[keep], np.asarray(M["LS_ID"], np.int64)[keep]
    uniq = np.unique(lsid[lsid > 0])
    tap = TapPlus(url="https://datalab.noirlab.edu/tap", verbose=False)
    parts, t0 = [], time.time()
    for i in range(0, len(uniq), BATCH):
        ids = ",".join(str(int(x)) for x in uniq[i : i + BATCH])
        for attempt in range(4):
            try:
                parts.append(tap.launch_job(f"SELECT {COLS} FROM ls_dr9.tractor WHERE ls_id IN ({ids})")
                             .get_results())  # fmt: skip
                break
            except Exception as exc:  # noqa: BLE001 - retried, then raised
                print(f"batch {i}: attempt {attempt + 1} failed: {exc}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"batch {i} failed")
        if (i // BATCH) % 20 == 0:
            print(f"{i + BATCH} of {len(uniq)} LS_IDs, {time.time() - t0:.0f} s", flush=True)
    tr = vstack(parts)
    row = {int(x): k for k, x in enumerate(tr["ls_id"])}
    has = np.array([int(x) in row for x in lsid])
    idx = np.array([row.get(int(x), 0) for x in lsid])
    out = Table({"targetid": tid, "ls_id": lsid, "in_tractor": has})
    out["type"] = np.where(has, np.asarray(tr["type"]).astype(str)[idx], "")
    for c in ("fracflux_g", "fracflux_r", "fracmasked_g", "maskbits"):
        out[c] = np.where(has, np.asarray(tr[c], float)[idx], np.nan)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write(OUT / "fracflux.csv", format="csv", overwrite=True)
    print(f"{len(out)} targets, {int(has.sum())} with a Tractor row -> fracflux.csv")


if __name__ == "__main__":
    main()
