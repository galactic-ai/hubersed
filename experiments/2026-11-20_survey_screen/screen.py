"""Screen the BGS Bright and Faint parent into the scored sample with the rules of validate.py.

Steps: median per-pixel S/N > 3 over pixels with w > 0, from desi_sv3_bright_v1; a FastSpecFit
row; VAC z in [0.01, 0.6]; |z - z_VAC| / (1 + z_VAC) <= 0.01; then Gaia star, SGA, PSF with weak
gas, BPT AGN and shred as in validate.py. Targets outside the 2026-09 screened set get new inputs:
Gaia DR3 within 2 arcsec from the ESA archive gaiadr3.gaia_source, renamed to the VizieR columns
of the 2026-09 file, and Legacy Surveys DR9 type from the Legacy Survey viewer catalog, both
cached in results/2026-11-20_survey_screen/inputs.
COMMAND: ``uv run python experiments/2026-11-20_survey_screen/screen.py``.
OUTPUT: results/2026-11-20_survey_screen/scored_bgs.npz, snr_parent.npz and screen.txt.
RESULT: 105,118 scored from the 250,987 parent (114,019 with S/N > 3). Of the old 104,990, 104,903
stay and 87 leave; 215 are new. New inputs: Gaia for 237 targets, morphology for 118.
Assumptions: shred rule of validate.py (5 boundary pairs differ from 2026-09); Gaia from the ESA
archive, not CDS XMatch, for the 237.
"""

import csv
import importlib.util
from pathlib import Path

import astropy.units as u
import numpy as np
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.table import Table, vstack

from hubersed.detect.line_screen import NSPECIES_MIN, bpt, forbidden
from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
OUT = PATHS["RESULTS"] / "2026-11-20_survey_screen"
INPUTS = OUT / "inputs"
DATASET = PATHS["DATA"] / "datasets" / "desi_sv3_bright_v1"
PARENT = PATHS["RESULTS"] / "2026-11-20_bgs_parent_selection" / "parent_bgs.npz"
SNR_MIN = 3.0


def validate_module():
    """Load validate.py of this folder, which holds the flag rules."""
    spec = importlib.util.spec_from_file_location("survey_validate", HERE / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parent_snr(parent):
    """Return the median per-pixel S/N and the stored z of each parent spectrum, cached."""
    cache = OUT / "snr_parent.npz"
    if cache.exists():
        d = np.load(cache)
        assert np.array_equal(d["targetid"], parent), "cached S/N is for another parent list"
        return d["snr"], d["z"]
    want = set(parent.tolist())
    found = {}
    for f in sorted((DATASET / "data").glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["target_id", "spec", "w", "z"])
            tid = t["target_id"].to_numpy()
            sel = np.flatnonzero([x in want for x in tid.tolist()])
            if sel.size == 0:
                continue
            n = t.num_rows
            spec = t["spec"].combine_chunks().values.to_numpy().reshape(n, -1)[sel]
            w = t["w"].combine_chunks().values.to_numpy().reshape(n, -1)[sel]
            snr = np.where(w > 0, spec * np.sqrt(w), np.nan)
            med = np.nanmedian(snr, axis=1)
            for k, j in enumerate(sel):
                found[int(tid[j])] = (med[k], float(t["z"][int(j)].as_py()))
    missing = [t for t in parent if t not in found]
    assert not missing, f"{len(missing)} parent spectra missing from {DATASET}"
    snr = np.array([found[t][0] for t in parent], np.float32)
    z = np.array([found[t][1] for t in parent])
    np.savez(cache, targetid=parent, snr=snr, z=z)
    return snr, z


GAIA_COLS = ["Plx", "e_Plx", "PM", "pmRA", "e_pmRA", "pmDE", "e_pmDE", "RUWE", "Gmag"]
# ESA archive gaiadr3.gaia_source columns, in the order and VizieR names of GAIA_COLS
GAIA_SOURCE = [
    "parallax", "parallax_error", "pm", "pmra", "pmra_error", "pmdec", "pmdec_error", "ruwe",
    "phot_g_mean_mag",
]  # fmt: skip
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def gaia_new(tids, ra, dec):
    """Return Gaia DR3 sources within 2 arcsec of new targets from the ESA archive, cached.

    The columns match the 2026-09 CDS XMatch table: tid, angDist (arcsec) and GAIA_COLS.
    """
    import urllib.parse
    import urllib.request

    from astropy.coordinates import SkyCoord
    from astropy.io import ascii

    cache = INPUTS / "gaia_xmatch_new.fits"
    if cache.exists():
        return Table.read(cache)
    INPUTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, a, d in zip(tids, ra, dec, strict=True):
        q = (
            f"SELECT ra, dec, {', '.join(GAIA_SOURCE)} FROM gaiadr3.gaia_source WHERE "
            f"1 = CONTAINS(POINT(ra, dec), CIRCLE({a}, {d}, {2.0 / 3600}))"
        )
        body = urllib.parse.urlencode(
            {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q}
        ).encode()
        text = urllib.request.urlopen(GAIA_TAP, data=body, timeout=120).read().decode()
        g = ascii.read(text.splitlines(), format="csv")
        if len(g) == 0:
            continue
        sep = SkyCoord(a * u.deg, d * u.deg).separation(
            SkyCoord(np.asarray(g["ra"], float) * u.deg, np.asarray(g["dec"], float) * u.deg)
        )
        part = Table({"tid": np.full(len(g), t, np.int64), "angDist": sep.arcsec})
        for name, src in zip(GAIA_COLS, GAIA_SOURCE, strict=True):
            part[name] = np.ma.filled(np.ma.asarray(g[src], dtype=float), np.nan)
        rows.append(part)
    out = vstack(rows) if rows else Table(names=["tid", "angDist", *GAIA_COLS])
    out.write(cache)
    return out


def morph_new(ls_ids, tids, ra, dec):
    """Return the Legacy Surveys DR9 type of new targets, matched by LS_ID, cached.

    Types come from the Legacy Survey viewer catalog of a 3 arcsec box around each target.
    LS_ID is release << 40 | brickid << 16 | objid.
    """
    import io
    import urllib.request

    cache = INPUTS / "morph_new.csv"
    if cache.exists():
        return {int(r["target_id"]): r["type"].strip() for r in csv.DictReader(open(cache))}
    INPUTS.mkdir(parents=True, exist_ok=True)
    out = {}
    r = 3.0 / 3600
    for t, ls, a, d in zip(tids, ls_ids, ra, dec, strict=True):
        url = (
            "https://www.legacysurvey.org/viewer/ls-dr9/cat.fits?"
            f"ralo={a - r}&rahi={a + r}&declo={d - r}&dechi={d + r}"
        )
        data = urllib.request.urlopen(url, timeout=60).read()
        typ = ""
        if data.startswith(b"SIMPLE"):
            c = Table.read(io.BytesIO(data))
            ids = (
                (np.asarray(c["release"], np.int64) << 40)
                | (np.asarray(c["brickid"], np.int64) << 16)
                | np.asarray(c["objid"], np.int64)
            )
            hit = np.flatnonzero(ids == int(ls))
            typ = str(c["type"][hit[0]]).strip() if hit.size else ""
        out[int(t)] = typ
    with open(cache, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_id", "type"])
        w.writerows(out.items())
    return out


def main():
    """Apply every step to the parent, write the cut flow and the scored sample."""
    v = validate_module()
    OUT.mkdir(parents=True, exist_ok=True)
    parent = np.load(PARENT)["targetid"].astype(np.int64)
    snr, z_pipe = parent_snr(parent)
    old_keep = set(np.load(v.REPLAY / "prescreen_state.npz")["keep"].astype(np.int64).tolist())

    h = fits.open(v.VAC, memmap=True)
    d, md = h["FASTSPEC"].data, h["METADATA"].data
    row = {t: i for i, t in enumerate(np.asarray(md["TARGETID"]).astype(np.int64))}
    z_vac = np.asarray(md["Z"], float)

    flow = [("parent", parent.size)]
    m = snr > SNR_MIN
    flow.append(("S/N > 3", int(m.sum())))
    m &= np.array([t in row for t in parent])
    flow.append(("FastSpecFit row", int(m.sum())))
    zv = np.array([z_vac[row[t]] if t in row else np.nan for t in parent])
    m &= (zv >= 0.01) & (zv <= 0.6)
    flow.append(("VAC z in [0.01, 0.6]", int(m.sum())))
    m &= np.abs(z_pipe - zv) / (1 + zv) <= 0.01
    flow.append(("z agrees", int(m.sum())))
    s4 = parent[m]

    ii = np.array([row[t] for t in s4])
    ra, dec = np.asarray(md["RA"], float)[ii], np.asarray(md["DEC"], float)[ii]
    new = np.array([t not in old_keep for t in s4])

    cols = ["tid", "angDist", *GAIA_COLS]
    g_old = Table.read(v.REPLAY / "gaia_xmatch.fits")[cols]
    g_new = gaia_new(s4[new], ra[new], dec[new]) if new.any() else g_old[:0]
    gtab = vstack([g_old, g_new[cols]]) if len(g_new) else g_old
    gaia = v.gaia_flags(s4, gtab)

    morph = {
        int(r["target_id"]): r["type"].strip()
        for r in csv.DictReader(open(v.REPLAY / "morph_bulk.csv"))
    }
    weak = np.array([forbidden(d, i)[1] < NSPECIES_MIN for i in ii])
    need = weak & np.array([t not in morph for t in s4])
    if need.any():
        ls = np.asarray(md["LS_ID"]).astype(np.int64)[ii]
        morph.update(morph_new(ls[need], s4[need], ra[need], dec[need]))
    psf = np.array([morph.get(int(t), "") == "PSF" for t in s4])

    sga = v.sga_flags(ra, dec)
    agn = np.array([bpt(d, i)[0] == "AGN" for i in ii])
    with fits.open(v.CATALOG, memmap=True) as hc:
        c = hc["ZCATALOG"].data
        c_tid = np.asarray(c["TARGETID"]).astype(np.int64)
        c_ra, c_dec, c_z = (np.asarray(c[k], float) for k in ("TARGET_RA", "TARGET_DEC", "Z"))
    cpos = {t: i for i, t in enumerate(c_tid)}
    jj = np.array([cpos[t] for t in s4])
    real = c_tid > 0
    shred = v.shred_flags(c_ra[jj], c_dec[jj], c_z[jj], c_ra[real], c_dec[real], c_z[real])

    keep = np.ones(s4.size, bool)
    for name, flag in (
        ("Gaia star", gaia),
        ("SGA", sga),
        ("PSF with weak gas", psf & weak),
        ("BPT AGN", agn),
        ("shred", shred),
    ):
        keep &= ~flag
        flow.append((name, int(keep.sum())))
    scored = np.sort(s4[keep])

    ref = np.load(PATHS["RESULTS"] / "prescreened_sample_v1.npz")["keep"].astype(np.int64)
    lines = [f"{name:24s} {n:8d}" for name, n in flow]
    lines += [
        "",
        f"new inputs queried: Gaia for {int(new.sum())} targets, morphology for {int(need.sum())}",
        f"old scored {ref.size}: kept {np.isin(ref, scored).sum()}, "
        f"left {(~np.isin(ref, scored)).sum()}; new scored not in old {(~np.isin(scored, ref)).sum()}",
    ]
    text = "\n".join(lines)
    (OUT / "screen.txt").write_text(text + "\n")
    print(text)
    np.savez(OUT / "scored_bgs.npz", keep=scored)


if __name__ == "__main__":
    main()
