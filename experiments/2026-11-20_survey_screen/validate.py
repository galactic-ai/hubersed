"""Recompute the screening flags of the old scored sample from committed rules and compare.

The 2026-09 screening code was not kept, only its flags. This recomputes each flag for the
113,880 spectra that pass steps 1-4 of 2026-11-10_prescreen_replay and compares it with the saved
flag, then applies the recomputed chain and compares the result with prescreened_sample_v1.npz.
Rules: Gaia as in the replay; SGA-2020 r_ell <= 1 in a 0.25 deg box (sky_screens); weak gas,
fewer than NSPECIES_MIN forbidden species (line_screen.forbidden), counted only for PSF type;
BPT AGN (line_screen.bpt); shred, another target within 10 proper kpc and |dz| < 0.002 (Planck18,
build_cont_outlier_sample). The adopted shred pool is every DR1 SV3 bright catalog target, with the
catalog positions and redshifts; other pools are listed for comparison.
COMMAND: ``uv run python experiments/2026-11-20_survey_screen/validate.py``.
OUTPUT: results/2026-11-20_survey_screen/validate.txt and validate.npz.
RESULT: SGA, weak-gas and BPT flags match exactly. The adopted shred rule differs on 5 pairs at
9.91-10.05 kpc, with no single cutoff matching the saved flags, so the chain keeps 104,991 against
104,990 (2 extra, 1 missing; TARGETIDs in validate.txt).
Assumptions: the 2026-09 shred code is lost; SGA from the SGA-2020 file, not the Data Lab table.
"""

import csv

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree

from hubersed.detect.line_screen import NSPECIES_MIN, bpt, forbidden
from hubersed.detect.sky_screens import SGA_BOX, ellipse_radius
from hubersed.paths import PATHS

REPLAY = PATHS["RESULTS"] / "2026-11-10_prescreen_replay" / "inputs"
OUT = PATHS["RESULTS"] / "2026-11-20_survey_screen"
VAC = PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"
SGA = PATHS["DATA"] / "catalogs" / "SGA-2020.fits"
CATALOG = PATHS["DATA"] / "desi_dr1" / "zpix-sv3-bright.fits"
ADOPTED = "pool DR1 catalog, all z"
SHRED_KPC, SHRED_DZ = 10.0, 0.002


def unit_vectors(ra, dec):
    """Return unit vectors on the sphere for positions in degrees."""
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


def gaia_flags(tids, t=None):
    """Return the Gaia star flag of each TARGETID, the rule of the 2026-11-10 replay.

    ``t`` is the Gaia match table, by default the 2026-09 cross-match of the replay.
    """
    if t is None:
        t = Table.read(REPLAY / "gaia_xmatch.fits")
    order = np.argsort(np.asarray(t["angDist"], float))
    near = {}
    for i in order:
        near.setdefault(int(t["tid"][i]), i)

    def col(i, name):
        try:
            v = float(t[name][i])
        except Exception:
            return np.nan
        return v if np.isfinite(v) else np.nan

    flag = {}
    for tid, i in near.items():
        sep, plx, eplx = col(i, "angDist"), col(i, "Plx"), col(i, "e_Plx")
        plx_snr = plx / eplx if np.isfinite(plx) and np.isfinite(eplx) and eplx > 0 else np.nan
        pm, pmra, pmde = col(i, "PM"), col(i, "pmRA"), col(i, "pmDE")
        epmra, epmde = col(i, "e_pmRA"), col(i, "e_pmDE")
        pme = np.hypot(pmra * epmra, pmde * epmde) / pm if np.isfinite(pm) and pm > 0 else np.nan
        pm_snr = pm / pme if np.isfinite(pme) and pme > 0 else np.nan
        ruwe, g = col(i, "RUWE"), col(i, "Gmag")
        onsrc = (
            sep < 1.0
            and np.isfinite(ruwe)
            and ruwe < 1.4
            and ((np.isfinite(plx_snr) and plx_snr > 5) or (np.isfinite(pm_snr) and pm_snr > 5))
        )
        wing = (
            np.isfinite(g)
            and g < 16.0
            and ((np.isfinite(plx_snr) and plx_snr > 5) or (np.isfinite(pm) and pm > 3.0))
        )
        flag[tid] = bool(onsrc or wing)
    return np.array([flag.get(int(x), False) for x in tids])


def sga_flags(ra, dec):
    """Return whether each position lies inside the D26 ellipse of an SGA-2020 galaxy in its box."""
    with fits.open(SGA, memmap=True) as h:
        g = h["ELLIPSE"].data
        g_ra, g_dec = np.asarray(g["RA"], float), np.asarray(g["DEC"], float)
        d26, pa, ba = (np.asarray(g[c], float) for c in ("D26", "PA", "BA"))
    ok = np.isfinite(d26) & np.isfinite(ba) & (d26 > 0) & (ba > 0)
    pa = np.nan_to_num(pa)
    tree = cKDTree(unit_vectors(g_ra, g_dec))
    # chord of the box half-diagonal, wide enough at every declination we have
    reach = 2 * np.sin(
        np.radians(SGA_BOX * 1.5 / max(np.cos(np.radians(np.abs(dec).max())), 1e-3)) / 2
    )
    out = np.zeros(ra.size, bool)
    for k, idx in enumerate(tree.query_ball_point(unit_vectors(ra, dec), reach)):
        idx = np.asarray(idx, int)
        if idx.size == 0:
            continue
        dra = SGA_BOX / max(np.cos(np.radians(dec[k])), 1e-3)
        dd = (g_ra[idx] - ra[k] + 180.0) % 360.0 - 180.0
        box = (np.abs(dd) <= dra) & (np.abs(g_dec[idx] - dec[k]) <= SGA_BOX) & ok[idx]
        idx = idx[box]
        if idx.size == 0:
            continue
        r, _ = ellipse_radius(ra[k], dec[k], g_ra[idx], g_dec[idx], d26[idx], pa[idx], ba[idx])
        out[k] = bool(np.nanmin(r) <= 1.0)
    return out


def shred_flags(ra, dec, z, pool_ra, pool_dec, pool_z, zmax=None):
    """Return whether another pool target lies within SHRED_KPC proper and |dz| < SHRED_DZ."""
    kpc_per_as = Planck18.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value
    rad = np.radians(SHRED_KPC / kpc_per_as / 3600.0)
    pool = unit_vectors(pool_ra, pool_dec)
    tree = cKDTree(pool)
    xyz = unit_vectors(ra, dec)
    out = np.zeros(ra.size, bool)
    for k, idx in enumerate(tree.query_ball_point(xyz, 2 * np.sin(rad / 2))):
        if zmax is not None and z[k] >= zmax:
            continue
        idx = np.asarray(idx, int)
        sep = 2 * np.arcsin(np.linalg.norm(pool[idx] - xyz[k], axis=1) / 2)
        out[k] = bool(((sep > 0) & (np.abs(pool_z[idx] - z[k]) < SHRED_DZ)).any())
    return out


def compare(name, new, old):
    """Return one line of agreement counts between a recomputed and a saved flag."""
    return (
        f"{name:34s} saved {int(old.sum()):6d}  new {int(new.sum()):6d}  both {int((new & old).sum()):6d}"
        f"  only saved {int((old & ~new).sum()):5d}  only new {int((new & ~old).sum()):5d}"
    )


def main():
    """Recompute the flags, compare them with the saved ones and apply the chain."""
    OUT.mkdir(parents=True, exist_ok=True)
    keep = np.load(REPLAY / "prescreen_state.npz")["keep"].astype(np.int64)
    saved = {
        k: np.load(REPLAY / f"{k}.npy") for k in ("sga_flag_full", "weak_mask", "bpt_agn_full")
    }
    saved["shred_flag_full"] = np.load(REPLAY / "shred_flag_full.npy")
    ref = np.load(PATHS["RESULTS"] / "prescreened_sample_v1.npz")["keep"].astype(np.int64)

    h = fits.open(VAC, memmap=True)
    d, md = h["FASTSPEC"].data, h["METADATA"].data
    vac_tid = np.asarray(md["TARGETID"]).astype(np.int64)
    row = {t: i for i, t in enumerate(vac_tid)}
    ii = np.array([row[t] for t in keep])
    ra, dec = np.asarray(md["RA"], float)[ii], np.asarray(md["DEC"], float)[ii]
    zv = np.asarray(md["Z"], float)[ii]
    morph = {
        int(r["target_id"]): r["type"].strip()
        for r in csv.DictReader(open(REPLAY / "morph_bulk.csv"))
    }

    new = {"gaia": gaia_flags(keep)}
    new["sga_flag_full"] = sga_flags(ra, dec)
    new["weak_mask"] = np.array([forbidden(d, i)[1] < NSPECIES_MIN for i in ii])
    new["bpt_agn_full"] = np.array([bpt(d, i)[0] == "AGN" for i in ii])
    psf = np.array([morph.get(int(t), "") == "PSF" for t in keep])

    with fits.open(CATALOG, memmap=True) as hc:
        c = hc["ZCATALOG"].data
        c_tid = np.asarray(c["TARGETID"]).astype(np.int64)
        c_ra, c_dec, c_z = (np.asarray(c[k], float) for k in ("TARGET_RA", "TARGET_DEC", "Z"))
    cpos = {t: i for i, t in enumerate(c_tid)}
    jj = np.array([cpos[t] for t in keep])
    real = c_tid > 0
    vac = (np.asarray(md["RA"], float), np.asarray(md["DEC"], float), np.asarray(md["Z"], float))
    variants = {
        "pool VAC, all z": ((ra, dec, zv), vac, None),
        "pool VAC, z < 0.02": ((ra, dec, zv), vac, 0.02),
        "pool steps 1-4 set, all z": ((ra, dec, zv), (ra, dec, zv), None),
        ADOPTED: (
            (c_ra[jj], c_dec[jj], c_z[jj]),
            (c_ra[real], c_dec[real], c_z[real]),
            None,
        ),
    }
    lines = ["flag comparison on the 113,880 spectra after steps 1-4"]
    for k in ("sga_flag_full", "weak_mask", "bpt_agn_full"):
        lines.append(compare(k, new[k], saved[k]))
    shred = {}
    for name, (target, pool, zmax) in variants.items():
        shred[name] = shred_flags(*target, *pool, zmax)
        lines.append(compare(f"shred, {name}", shred[name], saved["shred_flag_full"]))
    s, old = shred[ADOPTED], saved["shred_flag_full"]
    lines += [
        f"adopted shred rule: {ADOPTED}",
        f"  only saved {keep[old & ~s].tolist()}, only new {keep[s & ~old].tolist()}",
    ]

    drop = new["gaia"] | new["sga_flag_full"] | (new["weak_mask"] & psf) | new["bpt_agn_full"] | s
    final = np.sort(keep[~drop])
    lines += [
        "",
        f"recomputed chain keeps {final.size}; prescreened_sample_v1 keeps {ref.size}",
        f"only recomputed {np.setdiff1d(final, ref).tolist()}; "
        f"only saved {np.setdiff1d(ref, final).tolist()}",
    ]
    text = "\n".join(lines)
    (OUT / "validate.txt").write_text(text + "\n")
    print(text)
    np.savez(OUT / "validate.npz", keep=keep, psf=psf, final=final, shred=s, **new)


if __name__ == "__main__":
    main()
