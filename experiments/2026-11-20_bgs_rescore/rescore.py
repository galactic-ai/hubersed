"""Score the new BGS scored sample with the flows behind the 310 and apply the same outlier rules.

QUESTION: which galaxies of the new 105,118 scored sample are outliers under the 310's flows,
thresholds, voting and SIMBAD screen, and how does the set compare with the 310?
HYPOTHESIS: none stated before the run. Check first: the old scored sample must give back the
310 (full 231, continuum 92) exactly, or for v2 the 181 (full 92, cont 94).
INPUTS: results/2026-11-20_survey_screen/scored_bgs.npz; data/latents/spender_spec_<tag>_snr3.h5
(the 310's DESI latents); data/datasets/desi_sv3_bright_v1 for spectra missing from them;
data/checkpoints; results/noised_cue_meanzero_wide_flow (flows, thresholds);
results/simbad_screen_v2.csv and results/2026-11-09_mocks_v2_outliers/simbad_new.csv (SIMBAD cache).
SEED: none.
COMMAND: ``uv run python experiments/2026-11-20_bgs_rescore/rescore.py``; for the v2 mocks add
``--flows results/2026-11-09_mocks_v2_outliers/flows --ref
results/2026-11-09_mocks_v2_outliers/postsimbad_v2.npz --name v2``.
RESULT: the old scored sample gives back the 310 exactly (full 231, cont 92; log p within 5.4e-3
of the saved values). The new 105,118 give 301 outliers (full 225, cont 86, both 10): the 310 minus
the 9 that left the parent, and no new ones among the 213 newly encoded spectra. The v2 flows give
back the 181 and on the new sample 173 (full 92, cont 86, both 5), 8 dropped, none new.
Assumptions: same flows, thresholds and SIMBAD cache as the 310; the 9 drops are parent cuts.
FIGURES: none (rescore.txt, outliers_bgs.npz).
"""

import argparse
import csv
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from astropy.coordinates import SkyCoord
from astropy.io import fits
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import DATASET, _index
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-20_bgs_rescore"
FLOWS = RES / "noised_cue_meanzero_wide_flow"
LATENTS = PATHS["DATA"] / "latents"
TAGS = ("6latent", "10latent", "15latent", "cont10latent", "cont15latent")
CKPT = {
    "6latent": "spender_asc_run_6latent_zmax.pt",
    "10latent": "spender_asc_run_10latent_zmax.pt",
    "15latent": "spender_asc_run_15latent_zmax.pt",
    "cont10latent": "spender_desi_cont_10latent_zmax.pt",
    "cont15latent": "spender_desi_cont_15latent_zmax.pt",
}
# As experiments/2026-11-09_mocks_v2_outliers/simbad.py and scripts/simbad_screen.py
SEP_MAX = 2.0  # arcsec
AGN_TYPES = {"BLL", "QSO", "AGN", "Sy1", "Sy2", "SyG", "QSO_Candidate", "Bla", "LIN", "rG"}
STAR_TYPES = {"Star", "Pe*", "WD*", "HB*", "RGB*", "*", "PM*", "HV*"}
SIMBAD_SERVER = "simbad.harvard.edu"  # the CDS server returned 404 on 2026-11-10 and 2026-11-20
NEW_CACHE = OUT / "simbad_new.csv"


def normalized_spectra(tids):
    """Return the stored (normalized) spectra of TARGETIDs from desi_sv3_bright_v1."""
    files, all_tid, fidx, rows = _index(str(DATASET))
    out = []
    for t in tids:
        k = np.searchsorted(all_tid, t)
        assert all_tid[k] == t, f"{t} not in {DATASET}"
        pf = pq.ParquetFile(files[fidx[k]])
        row = int(rows[k])
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            if row < n:
                break
            row -= n
        spec = pf.read_row_group(g, columns=["spec"])["spec"].combine_chunks()
        out.append(spec.values.to_numpy().reshape(-1, spec.type.list_size)[row])
    return np.stack(out).astype(np.float32)


def latents(tag, tids, models):
    """Return the latents of TARGETIDs: from the 310's file, encoding any that are missing."""
    with h5py.File(LATENTS / f"spender_spec_{tag}_snr3.h5", "r") as f:
        ftid = np.asarray(f["target_ids"], np.int64)
        flat = np.asarray(f["latents"], np.float32)
    pos = {t: i for i, t in enumerate(ftid.tolist())}
    lat = np.zeros((len(tids), flat.shape[1]), np.float32)
    have = np.array([t in pos for t in tids.tolist()])
    lat[have] = flat[[pos[t] for t in tids[have].tolist()]]
    if (~have).any():
        if tag not in models:
            models[tag] = load_model(
                str(PATHS["DATA"] / "checkpoints" / CKPT[tag]), DESI(), map_location="cpu",
                weights_only=False,
            ).float().eval()  # fmt: skip
        with torch.no_grad():
            spec = torch.from_numpy(normalized_spectra(tids[~have]))
            lat[~have] = models[tag].encode(spec).numpy()
    return lat, int((~have).sum())


def flow_logp(tag, lat, flows=FLOWS):
    """Return log p under the saved flow of ``tag``, its threshold and the saved DESI log p."""
    b = torch.load(flows / f"flow_nsf_{tag}.pt", map_location="cpu", weights_only=False)
    nde = build_flow(b["method"], b["dim"], b["hidden"], b["num_transforms"], b["num_bins"])
    nde.load_state_dict(b["state_dict"])
    nde.eval()
    x = (lat - np.asarray(b["scaler_mean"])) / np.asarray(b["scaler_scale"])
    with torch.no_grad():
        lp = nde.log_prob(torch.from_numpy(x).float()).numpy()
    old = torch.load(
        flows / f"desi_outliers_flow_nsf_{tag}_snr3.pt", map_location="cpu", weights_only=False
    )
    saved = dict(
        zip(np.asarray(old["desi_target_ids"], np.int64).tolist(), old["log_p_desi"], strict=True)
    )
    return lp, float(old["threshold"]), saved


def simbad_cache():
    """Return the cached SIMBAD flag of every TARGETID queried before."""
    cache = {}
    with open(RES / "simbad_screen_v2.csv") as f:
        for r in csv.DictReader(f):
            cache[int(r["target_id"])] = r["simbad_agn"] == "True" or r["simbad_star"] == "True"
    for path in (RES / "2026-11-09_mocks_v2_outliers" / "simbad_new.csv", NEW_CACHE):
        if path.exists():
            with open(path) as f:
                for r in csv.DictReader(f):
                    cache[int(r["target_id"])] = r["flagged"] == "True"
    return cache


def query_simbad(todo, cache):
    """Query SIMBAD within SEP_MAX of each TARGETID, append to NEW_CACHE and update ``cache``."""
    from astroquery.simbad import Simbad

    meta = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)["METADATA"].data
    row = {int(t): i for i, t in enumerate(meta["TARGETID"])}
    s = Simbad()
    s.server = SIMBAD_SERVER
    for fld in ("otype", "sp_type"):
        try:
            s.add_votable_fields(fld)
        except Exception:
            pass
    new_file = not NEW_CACHE.exists()
    with open(NEW_CACHE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["target_id", "otype", "sp_type", "sep_arcsec", "flagged"])
        for t in todo:
            i = row[t]
            here = SkyCoord(float(meta["RA"][i]), float(meta["DEC"][i]), unit="deg")
            otype, sptype, sep = "", "", np.nan
            r = s.query_region(here, radius=SEP_MAX * u.arcsec)
            if r is not None and len(r):
                o = SkyCoord(float(r[0]["ra"]), float(r[0]["dec"]), unit="deg")
                sep = here.separation(o).arcsec
                otype = str(r[0]["otype"]).strip()
                sptype = str(r[0].get("sp_type", "") or "").strip()
            flagged = bool(
                sep < SEP_MAX
                and (otype in AGN_TYPES or otype in STAR_TYPES or sptype.startswith("dC"))
            )
            w.writerow([t, otype, sptype, f"{sep:.3f}", flagged])
            f.flush()
            cache[t] = flagged


def outliers(keep, models, lines, flows=FLOWS):
    """Return the full and continuum outlier sets of a scored sample before SIMBAD."""
    flagged = {}
    for tag in TAGS:
        lat, n_enc = latents(tag, keep, models)
        lp, thr, saved = flow_logp(tag, lat, flows)
        both = [i for i, t in enumerate(keep.tolist()) if t in saved]
        diff = float(np.abs(lp[both] - np.array([saved[keep[i]] for i in both])).max())
        flagged[tag] = set(keep[lp <= thr].tolist())
        lines.append(
            f"  {tag}: {len(keep)} scored, {n_enc} encoded here, threshold {thr:.3f}, "
            f"flagged {len(flagged[tag])}, max |log p - saved| {diff:.2e}"
        )
    full = flagged["6latent"] & flagged["10latent"] & flagged["15latent"]
    cont = flagged["cont10latent"] & flagged["cont15latent"]
    return full, cont


def main():
    """Reproduce the 310 from the old scored sample, then score the new one."""
    OUT.mkdir(parents=True, exist_ok=True)
    models = {}
    cache = simbad_cache()
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", type=Path, default=FLOWS, help="folder of the five flows")
    ap.add_argument(
        "--ref",
        type=Path,
        default=RES / "prescreened_outliers_postsimbad.npz",
        help="full and cont outliers of these flows on the old scored sample",
    )
    ap.add_argument("--name", default="", help="suffix of the output files")
    args = ap.parse_args()
    sfx = f"_{args.name}" if args.name else ""
    ref = np.load(args.ref)
    ref_full, ref_cont = (set(np.asarray(ref[k], np.int64).tolist()) for k in ("full", "cont"))
    lines = []

    old_keep = np.sort(np.load(RES / "prescreened_sample_v1.npz")["keep"].astype(np.int64))
    lines.append(f"check: old scored sample {old_keep.size}")
    f_old, c_old = outliers(old_keep, models, lines, args.flows)
    assert all(t in cache for t in f_old | c_old), "old candidates missing from the SIMBAD cache"
    f_old = {t for t in f_old if not cache[t]}
    c_old = {t for t in c_old if not cache[t]}
    lines.append(
        f"  after SIMBAD: full {len(f_old)}, cont {len(c_old)}, union {len(f_old | c_old)}; "
        f"equal to the reference sets: full {f_old == ref_full}, cont {c_old == ref_cont}"
    )

    new_keep = np.sort(np.load(RES / "2026-11-20_survey_screen" / "scored_bgs.npz")["keep"])
    lines.append(f"new scored sample {new_keep.size}")
    f_new, c_new = outliers(new_keep, models, lines, args.flows)
    todo = sorted(t for t in f_new | c_new if t not in cache)
    lines.append(f"  candidates full {len(f_new)}, cont {len(c_new)}; SIMBAD queries {len(todo)}")
    if todo:
        query_simbad(todo, cache)
    f_new = {t for t in f_new if not cache[t]}
    c_new = {t for t in c_new if not cache[t]}
    union, ref_union = f_new | c_new, ref_full | ref_cont
    lines += [
        f"  after SIMBAD: full {len(f_new)}, cont {len(c_new)}, both {len(f_new & c_new)}, "
        f"union {len(union)}",
        f"  vs the reference: kept {len(union & ref_union)}, dropped {len(ref_union - union)}, "
        f"new {len(union - ref_union)}",
        f"  dropped: {sorted(ref_union - union)}",
        f"  new: {sorted(union - ref_union)}",
    ]
    text = "\n".join(lines)
    (OUT / f"rescore{sfx}.txt").write_text(text + "\n")
    print(text)
    np.savez(
        OUT / f"outliers_bgs{sfx}.npz", full=np.array(sorted(f_new)), cont=np.array(sorted(c_new))
    )


if __name__ == "__main__":
    main()
