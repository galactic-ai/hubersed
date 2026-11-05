"""Pick three control sets of 10 galaxies per outlier for the 94 common continuum outliers.

The pool is the S/N > 3 DESI spectra of the latent files that neither seed 0 continuum flow flags,
that are in the FastSpecFit VAC, and that pass the redshift cuts of sample.py (VAC z in
[0.01, 0.6], |z_pipeline - z_VAC| / (1 + z_VAC) <= 0.01). The Gaia, SGA, SIMBAD, PSF, BPT and
shred screens are not applied to the pool.

matched: nearest in (VAC z, r fiber magnitude, SNR_B), each standardised over the pool.
cont10, cont15: nearest in that flow's latent space, scaled by the flow's mock scaler.
Within a set no galaxy is used twice. Outliers take their neighbours in TARGETID order.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-31_spectrum_qa_cont_outliers/controls.py``.
OUTPUT: results/2026-10-31_spectrum_qa_cont_outliers/controls.npz.
"""

import h5py
import numpy as np
import torch
from astropy.io import fits
from scipy.spatial import cKDTree

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_spectrum_qa_cont_outliers"
FLOWS = RES / "noised_cue_meanzero_wide_flow"
N_PER = 10
ZMIN, ZMAX, ZBAD = 0.01, 0.6, 0.01


def pick(tree, queries, n_per):
    """Return n_per pool indices per query point, nearest first, never reusing an index."""
    used, out = set(), []
    for q in queries:
        k = 4 * n_per
        while True:
            _, idx = tree.query(q, k=min(k, tree.n))
            free = [int(i) for i in idx if int(i) not in used][:n_per]
            if len(free) == n_per or k >= tree.n:
                break
            k *= 4
        if len(free) < n_per:
            raise RuntimeError("pool too small")
        used.update(free)
        out.append(free)
    return np.array(out)


def main():
    """Build the pool, pick the three control sets and save them."""
    outliers = np.sort(np.load(OUT / "sample94.npz")["target_ids"])

    flagged = set()
    for tag in ("cont10", "cont15"):
        o = torch.load(
            FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
            weights_only=False,
            map_location="cpu",
        )
        flagged |= {int(t) for t in o["outlier_target_ids"]}

    lat = {}
    for tag in ("cont10", "cont15"):
        with h5py.File(PATHS["DATA"] / "latents" / f"spender_spec_{tag}latent_snr3.h5", "r") as f:
            tids = f["target_ids"][:].astype(np.int64)
            lat[tag] = f["latents"][:].astype(np.float64)
        d = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
        lat[tag] = (lat[tag] - d["scaler_mean"]) / d["scaler_scale"]
    row = {int(t): i for i, t in enumerate(tids)}

    h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
    M, S = h["METADATA"].data, h["FASTSPEC"].data
    vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}
    with h5py.File(RES / "lineEW_flow" / "desi_lines.h5", "r") as f:
        zpipe = dict(
            zip(f["target_ids"][:].astype(np.int64).tolist(), f["zs"][:].tolist(), strict=True)
        )

    def ok(t):
        if t in flagged or t not in vrow or t not in zpipe:
            return False
        z = float(M["Z"][vrow[t]])
        return ZMIN <= z <= ZMAX and abs(zpipe[t] - z) / (1 + z) <= ZBAD

    pool = np.array([t for t in tids.tolist() if ok(t)], np.int64)
    print(f"pool {len(pool)} of {len(tids)} latent spectra")

    def obs(t):
        i = vrow[t]
        ff = float(M["FIBERFLUX_R"][i])
        rfib = 22.5 - 2.5 * np.log10(ff) if ff > 0 else np.nan
        return [float(M["Z"][i]), rfib, float(S["SNR_B"][i])]

    xp = np.array([obs(t) for t in pool])
    good = np.all(np.isfinite(xp), axis=1)
    print(f"  {int((~good).sum())} pool galaxies dropped from matched: no positive FIBERFLUX_R")
    mu, sd = xp[good].mean(0), xp[good].std(0)
    xo = (np.array([obs(t) for t in outliers]) - mu) / sd
    if not np.all(np.isfinite(xo)):
        raise ValueError("an outlier has no positive FIBERFLUX_R")
    sets = {"matched": pool[good][pick(cKDTree((xp[good] - mu) / sd), xo, N_PER)]}
    for tag in ("cont10", "cont15"):
        lp = lat[tag][[row[t] for t in pool]]
        lo = lat[tag][[row[t] for t in outliers]]
        sets[tag] = pool[pick(cKDTree(lp), lo, N_PER)]

    np.savez(OUT / "controls.npz", outliers=outliers, pool_size=len(pool), **sets)
    for k, v in sets.items():
        print(f"{k}: {v.shape}, {len(np.unique(v))} unique")
    print(f"wrote {OUT / 'controls.npz'}")


if __name__ == "__main__":
    main()
