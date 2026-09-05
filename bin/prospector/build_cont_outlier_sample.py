import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck18
from scipy.stats import rankdata

from hubersed.paths import PATHS

# corrected-[OII]3729 run. The repo copy of noised_cue_meanzero_wide_flow/ is the stale
# Jul-21 PRE-fix output -- do not point at it.
DEFAULT_FLOW_DIR = PATHS["RESULTS"] / "wide_flow_corrected"
TAGS = ("cont10latent", "cont15latent")


def flow_scores(tag, flow_dir):
    """(target_id -> log p) and (target_id -> DESI rank percentile) plus the outlier set."""
    d = torch.load(Path(flow_dir) / f"desi_outliers_flow_nsf_{tag}_snr3.pt", weights_only=False)
    tid = np.asarray(d["desi_target_ids"], np.int64)
    lp = np.asarray(d["log_p_desi"], np.float64)
    pct = rankdata(lp, "average") / len(lp)
    return (dict(zip(tid.tolist(), lp)), dict(zip(tid.tolist(), pct)),
            set(int(x) for x in d["outlier_target_ids"]), float(d["threshold"]))


LAM_MAX = 9824.0   # DESI red-arm cutoff; Halpha 6563 leaves it at z = 0.497


def source_class(S, i, z):
    """'emission' | 'weak-em' | 'continuum' for row i of the FASTSPEC table.

    Above z = 0.497 Halpha is off the red end, so HALPHA_EW = 0 means NOT MEASURED and
    the classification falls back to Hbeta / [OII] / [OIII]. Getting this wrong labels
    every high-z object 'featureless'.
    """
    def snr(ln):
        f, iv = float(S[f"{ln}_FLUX"][i]), float(S[f"{ln}_FLUX_IVAR"][i])
        return f * np.sqrt(iv) if iv > 0 else 0.0

    if 6563.0 * (1 + z) < LAM_MAX:
        ew, s_ha = float(S["HALPHA_EW"][i]), snr("HALPHA")
        if s_ha > 5 and ew > 10:
            return "emission"
        return "weak-em" if s_ha > 5 and ew > 3 else "continuum"

    strong = max(snr("HBETA"), snr("OIII_5007"), min(snr("OII_3726"), snr("OII_3729")))
    nspec = sum(x > 3 for x in [snr("OIII_5007"), snr("NII_6584"), snr("OI_6300"),
                                min(snr("SII_6716"), snr("SII_6731")),
                                min(snr("OII_3726"), snr("OII_3729"))])
    if strong > 5 and nspec >= 2:
        return "emission"
    return "weak-em" if strong > 3 else "continuum"


def read_screen(path, value_col=None):
    """(set of flagged TARGETIDs, {TARGETID: value_col}) from a contam_screens.py CSV.

    Deliberately not tolerant of a missing file: silently skipping a contamination screen
    is how 7 known star contaminants got back into the sample during planning.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    flag = {int(r["target_id"]) for r in rows if r["flagged"] == "True"}
    vals = {int(r["target_id"]): float(r[value_col]) for r in rows} if value_col else {}
    return flag, vals


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default=str(PATHS["RESULTS"] / "cont_outlier_sample20.npz"))
    p.add_argument("-n", "--n-targets", type=int, default=20)
    p.add_argument("--flow-dir", default=str(DEFAULT_FLOW_DIR),
                   help="dir holding desi_outliers_flow_nsf_<tag>_snr3.pt from the "
                        "CORRECTED-[OII]3729 run")
    p.add_argument("--vac", default=str(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"))
    p.add_argument("--lines", default=str(PATHS["RESULTS"] / "lineEW_flow" / "desi_lines.h5"),
                   help="h5 carrying the pipeline redshifts used to build the latents")
    p.add_argument("--shred-kpc", type=float, default=10.0,
                   help="proper-kpc radius for the shred neighbour search")
    p.add_argument("--shred-zmax", type=float, default=0.02,
                   help="run the shred test below this redshift. The original 0.02 misses "
                        "shreds at 0.02-0.06 that a 1.5-arcsec fibre still lands on a knot "
                        "of: 39627758174736675 has THREE DESI targets inside 2.6 kpc and "
                        "Dn4000 = 0.809, below any stellar population. 10 proper kpc is "
                        "self-limiting at high z (1.6 arcsec at z=0.5), so 1.0 is safe.")
    p.add_argument("--keep-flagged", action="store_true",
                   help="keep zbad/shred/z/gaia/sga contaminants instead of dropping them")
    p.add_argument("--zmin", type=float, default=0.01,
                   help="drop very nearby resolved systems below this redshift")
    p.add_argument("--zmax", type=float, default=0.6,
                   help="drop the high-z end above this redshift")
    p.add_argument("--gaia-list", default=str(PATHS["RESULTS"] / "gaia_star_screen.csv"),
                   help="CSV from contam_screens.py; must exist")
    p.add_argument("--sga-list", default=str(PATHS["RESULTS"] / "sga_proximity.csv"),
                   help="CSV from contam_screens.py; must exist")
    p.add_argument("--source-class", nargs="*", default=None,
                   choices=["emission", "weak-em", "continuum"],
                   help="keep only these SOURCE types. Independent of the flow selection: "
                        "every candidate is a continuum-FLOW outlier regardless.")
    p.add_argument("--extra-list", nargs="*", default=None,
                   help="optional CSV from agn_star_screen.py (BPT AGN + non-stellar point "
                        "sources). Omit to reproduce the pre-2026-08-31 sample exactly.")
    args = p.parse_args(argv)

    lp, pct, out, thr = {}, {}, {}, {}
    for t in TAGS:
        lp[t], pct[t], out[t], thr[t] = flow_scores(t, args.flow_dir)
        print(f"{t}: {len(out[t])} outliers (thr={thr[t]:.3f})")
    common = out[TAGS[0]] & out[TAGS[1]]
    print(f"common to both continuum flows: {len(common)}")

    hdu = fits.open(args.vac)
    M, S = hdu["METADATA"].data, hdu["FASTSPEC"].data
    iv = {int(t): i for i, t in enumerate(M["TARGETID"])}
    zvac = M["Z"]

    import h5py
    with h5py.File(args.lines, "r") as f:
        zpipe = dict(zip(f["target_ids"][:].astype(np.int64).tolist(), f["zs"][:].tolist()))

    sel = sorted(t for t in common if t in iv)
    print(f"  of which in the FastSpecFit VAC: {len(sel)}")

    # contaminant flags
    sky = SkyCoord(M["RA"] * u.deg, M["DEC"] * u.deg)
    zbad, shred = {}, {}
    for t in sel:
        i = iv[t]
        zp = zpipe.get(t, np.nan)
        zbad[t] = bool(np.isfinite(zp) and abs(zp - zvac[i]) / (1 + zvac[i]) > 0.01)
        if zvac[i] < args.shred_zmax:
            kpc_per_as = Planck18.kpc_proper_per_arcmin(zvac[i]).to(u.kpc / u.arcsec).value
            rad = args.shred_kpc / kpc_per_as          # physical radius -> arcsec at this z
            sep = sky[i].separation(sky).arcsec
            shred[t] = bool(((sep < rad) & (sep > 0) & (np.abs(zvac - zvac[i]) < 0.002)).sum() > 0)
        else:
            shred[t] = False
    print(f"  flagged zbad={sum(zbad.values())}  shred={sum(shred.values())}")

    # external contamination screens (bin/prospector/contam_screens.py)
    gaia_flag, _ = read_screen(args.gaia_list)
    sga_flag, sga_r = read_screen(args.sga_list, "r_ell")
    extra_flag = set()
    for x in (args.extra_list or []):
        extra_flag |= read_screen(x)[0]
    zcut = {t: not (args.zmin <= zvac[iv[t]] <= args.zmax) for t in sel}
    n_z, n_gaia, n_sga = (sum(zcut.values()),
                          sum(t in gaia_flag for t in sel),
                          sum(t in sga_flag for t in sel))
    n_extra = sum(t in extra_flag for t in sel)
    print(f"  flagged z<{args.zmin} or z>{args.zmax}={n_z}  gaia={n_gaia}  sga={n_sga}"
          f"  agn/point-source={n_extra}")

    keep = sel if args.keep_flagged else [
        t for t in sel
        if not zbad[t] and not shred[t] and not zcut[t]
        and t not in gaia_flag and t not in sga_flag and t not in extra_flag
    ]
    print(f"  kept after contaminant cut: {len(keep)}")

    if args.source_class:
        cls = {t: source_class(S, iv[t], float(zvac[iv[t]])) for t in keep}
        n_by = {c: sum(v == c for v in cls.values()) for c in
                ("emission", "weak-em", "continuum")}
        keep = [t for t in keep if cls[t] in args.source_class]
        print(f"  source class {n_by} -> keeping {args.source_class}: {len(keep)}")

    score = {t: 0.5 * (pct[TAGS[0]][t] + pct[TAGS[1]][t]) for t in keep}
    top = sorted(keep, key=lambda t: score[t])[: args.n_targets]
    I = np.array([iv[t] for t in top])

    ns = S["NARROW_SIGMA"][I].astype(np.float64)
    ns = np.where(np.isfinite(ns) & (ns > 0), ns, 100.0)  # MAP seed only; 100 km/s fallback

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        outp,
        target_ids=np.array(top, np.int64),
        z=zvac[I].astype(np.float64),
        logmstar=S["LOGMSTAR"][I].astype(np.float64),
        narrow_sigma=ns,
        dn4000=S["DN4000"][I].astype(np.float64),
        halpha_ew=S["HALPHA_EW"][I].astype(np.float64),
        snr_halpha=S["HALPHA_AMP"][I].astype(np.float64) * np.sqrt(np.maximum(S["HALPHA_AMP_IVAR"][I], 0)),
        logp_cont10=np.array([lp[TAGS[0]][t] for t in top]),
        logp_cont15=np.array([lp[TAGS[1]][t] for t in top]),
        pct_cont10=np.array([pct[TAGS[0]][t] for t in top]),
        pct_cont15=np.array([pct[TAGS[1]][t] for t in top]),
        zbad=np.array([zbad[t] for t in top]),
        shred=np.array([shred[t] for t in top]),
        gaia_flagged=np.array([t in gaia_flag for t in top]),
        sga_flagged=np.array([t in sga_flag for t in top]),
        sga_r_ell=np.array([sga_r.get(t, np.nan) for t in top]),
        provenance=np.array(
            f"cont10latent AND cont15latent outliers, corrected-[OII]3729 run; "
            f"VAC-matched; ranked by mean DESI rank-percentile (NOT mock-CDF); "
            f"contaminants {'kept' if args.keep_flagged else 'dropped'} "
            f"(zbad={sum(zbad.values())}, shred={sum(shred.values())}, "
            f"z outside [{args.zmin}, {args.zmax}]={n_z}, gaia={n_gaia}, sga={n_sga}, "
            f"agn/point-source={n_extra}); "
            f"screens {Path(args.gaia_list).name} + {Path(args.sga_list).name} "
            f"(SGA-2020 stands in for SGA-2025); {len(keep)} survivors before the top-"
            f"{args.n_targets} cut"),
    )
    print(f"\nwrote {outp}  n={len(top)}")
    print(f"{'TARGETID':>18} {'z':>7} {'logM':>6} {'sigma':>7} {'pct10':>8} {'pct15':>8}")
    for t in top:
        i = iv[t]
        print(f"{t:>18} {zvac[i]:7.4f} {S['LOGMSTAR'][i]:6.2f} "
              f"{S['NARROW_SIGMA'][i]:7.1f} {100*pct[TAGS[0]][t]:7.3f}% {100*pct[TAGS[1]][t]:7.3f}%")


if __name__ == "__main__":
    main()
