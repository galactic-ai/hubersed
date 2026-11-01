"""DESI SV3 and SDSS BOSS spectra of 39633325949715572, robust6 target 6 (#17), in one panel.

QUESTION: is the wide bump near rest 6400-6600 A of the single DESI exposure also in an independent
spectrum? DR1 has no other DESI spectrum (desi_dr1.zpix: sv3/bright only, 1 exposure, 295 s).
HYPOTHESIS: none registered.
INPUTS: DESI SV3 bright coadd via load_spectrum (spender chunk, as the fits use); SDSS BOSS
plate 6793, MJD 56430, fiber 154 (z 0.266657, sciencePrimary), fetched with astroquery.sdss.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_boss_39633325949715572/analysis.py``.
RESULT: the bump is in DESI only. Median flux at rest 6450-6600 over the sidebands (6200-6350,
6700-6850) is 1.46 in DESI and 1.04 in BOSS. The spectra agree below ~4400 A rest; redder, DESI
falls below BOSS and is noisy past ~7000 A, a colour change a grey aperture offset cannot make.
Assumptions: no rescaling between the fibres; median filters run over unmasked pixels only, so
they bridge masked gaps.
Same-petal check (neighbours.py, neighbours.txt): the z-arm ratio at observed 8106-8360 A over
the sidebands is > 1.4 for 8 of 11 target fibres on EXPID 85210 petal 6 (median 2.00), and for 0 of
306 fibres on other petals of that exposure and 0 of 3807 in other exposures (medians 1.02). The
bump belongs to petal 6 of exposure 85210, not to the galaxy. The petal neighbours span z 0.08-1.18
and include a star (ratio 1.29), so no shared rest-frame feature; 39633325945524065 has 2.06 there
and 1.00, 1.06 in two other exposures (neighbour_checks.txt). Hardware against sky subtraction or
calibration on that petal is not separated here.
Whole petal (petal_excess.txt, cframes z6 and z5 of EXPID 85210): on z6 the excess swings with fibre
number, positive near fibres 3100-3190, 3300-3420 and 3450-3500 and negative near 3000-3050 and
3200-3300, and the 45 sky fibres follow the same curve (16-84% -4.1 to 1.7); on z5 sky fibres sit at
0 (16-84% -0.06 to 0.04). So the excess is in the sky-subtracted z6 frame, not in the objects.
FIGURES: results/2026-10-31_boss_39633325949715572/ boss_spectrum.npz, desi_boss.png,
neighbours.txt, neighbours.png, neighbour_checks.txt, petal_excess.txt, petal_excess.png
(cframes cached in data/iron_exposures/00085210/; <TID>_neighbours.npz of neighbours.slurm).
"""

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS
from scipy.signal import medfilt

from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39633325949715572
OUT = PATHS["RESULTS"] / "2026-10-31_boss_39633325949715572"
PLATE, MJD, FIBER = 6793, 56430, 154
MED_DESI, MED_BOSS = 15, 9  # pixels, about 12 A and 13-17 A observed


def boss():
    """Return BOSS wavelength (A), flux and ivar (1e-17 cgs) of the plate, MJD and fiber."""
    path = OUT / "boss_spectrum.npz"
    if not path.exists():
        c = SkyCoord(243.03034416822862, 55.21805676902358, unit="deg")
        m = SDSS.query_region(c, radius=3 * u.arcsec, spectro=True,
                              specobj_fields=["plate", "mjd", "fiberID", "run2d"])  # fmt: skip
        m = m[(m["plate"] == PLATE) & (m["mjd"] == MJD) & (m["fiberID"] == FIBER)]
        assert len(m) == 1, f"{len(m)} BOSS matches"
        d = SDSS.get_spectra(matches=m)[0][1].data
        np.savez(path, wave=10 ** np.asarray(d["loglam"], float), flux=np.asarray(d["flux"], float),
                 ivar=np.asarray(d["ivar"], float), and_mask=np.asarray(d["and_mask"]))  # fmt: skip
    b = np.load(path)
    return b["wave"], b["flux"], b["ivar"], b["and_mask"]


def main():
    """Plot both spectra, raw and median-filtered, against rest wavelength."""
    use_apj_style()
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_spectrum(TID)
    z = float(s.redshift.value)
    dw = np.asarray(s.spectral_axis.value, float)
    df = np.asarray(s.flux.value, float)
    dok = ~s.mask
    bw, bf, bi, bm = boss()
    bok = (bi > 0) & (bm == 0)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(dw[dok] / (1 + z), df[dok], color="#92c5de", lw=0.4, label="DESI SV3 bright, raw")
    ax.plot(bw[bok] / (1 + z), bf[bok], color="#f4a582", lw=0.4, label="BOSS 6793-56430-154, raw")
    ax.plot(dw[dok] / (1 + z), medfilt(df[dok], MED_DESI), color="#2166ac", lw=1.1,
            label=f"DESI, median filter {MED_DESI} px")  # fmt: skip
    ax.plot(bw[bok] / (1 + z), medfilt(bf[bok], MED_BOSS), color="#b2182b", lw=1.1,
            label=f"BOSS, median filter {MED_BOSS} px")  # fmt: skip
    for lam in (6400.0, 6600.0):
        ax.axvline(lam, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel(rf"rest wavelength [$\AA$], z = {z:.5f} (DESI)")
    ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax.set_title(f"{TID}: DESI 1.5'' fibre and BOSS 2'' fibre, not rescaled; dotted 6400, 6600 A")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(OUT / "desi_boss.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    r = {}
    for name, w, f, ok in (("DESI", dw, df, dok), ("BOSS", bw, bf, bok)):
        rest = w / (1 + z)
        side = ok & (((rest > 6200) & (rest < 6350)) | ((rest > 6700) & (rest < 6850)))
        mid = ok & (rest > 6450) & (rest < 6600)
        r[name] = np.median(f[mid]) / np.median(f[side])
        print(f"{name}: median flux rest 6450-6600 over 6200-6350 and 6700-6850: {r[name]:.3f}")
    print(f"wrote {OUT / 'desi_boss.png'}")


# %%
def neighbours():
    """Compare the bump ratio of the target's exposure and petal with every other fibre's."""
    use_apj_style()
    d = np.load(OUT / f"{TID}_neighbours.npz")
    expid, petal = int(d["EXPID"][0]), int(d["PETAL_LOC"][0])
    r = d["all_ratio"]
    same = (d["all_expid"] == expid) & (d["all_petal"] == petal)
    same_exp = (d["all_expid"] == expid) & ~same
    other = d["all_expid"] != expid
    lines = [f"# {TID}: z-arm flux at observed 8106-8360 A over 7854-8043 and 8486-8676 A, "
             f"target fibres of healpix 10155; target EXPID {expid}, petal {petal}, "
             f"ratio {float(d['target_ratio'][0]):.3f}"]  # fmt: skip
    for name, m in (("same exposure and petal", same), ("same exposure, other petals", same_exp),
                    ("other exposures", other)):  # fmt: skip
        x = r[m & np.isfinite(r)]
        q = np.percentile(x, [16, 50, 84]) if x.size else [np.nan] * 3
        lines.append(f"{name:>28s}: N {x.size:4d}, median {q[1]:.3f}, 16-84% {q[0]:.3f}-{q[2]:.3f}, "
                     f"fraction > 1.4 {np.mean(x > 1.4) if x.size else np.nan:.2f}")  # fmt: skip
    (OUT / "neighbours.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 4, 41)
    for name, m, c in (("other exposures", other, "0.6"), ("same exposure, other petals", same_exp,
                       "#2166ac"), (f"EXPID {expid} petal {petal}", same, "#b2182b")):  # fmt: skip
        x = np.clip(r[m & np.isfinite(r)], bins[0], bins[-1])
        ax.hist(
            x,
            bins=bins,
            density=True,
            histtype="step",
            color=c,
            lw=1.5,
            label=f"{name} (N {x.size})",
        )
    ax.axvline(float(d["target_ratio"][0]), color="k", ls="--", lw=1, label="target 6")
    ax.set_xlabel("flux ratio, observed 8106-8360 over sidebands (clipped at 0 and 4)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "neighbours.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'neighbours.png'}")


# %%
def neighbour_checks():
    """Return the petal neighbours' redshifts and their ratios in other exposures of the same file."""
    from astroquery.utils.tap.core import TapPlus

    from hubersed.detect.sky_screens import DL_TAP, query

    d = np.load(OUT / f"{TID}_neighbours.npz")
    expid, petal = int(d["EXPID"][0]), int(d["PETAL_LOC"][0])
    same = (d["all_expid"] == expid) & (d["all_petal"] == petal)
    tids = np.unique(d["all_targetid"][same])
    q = ("SELECT targetid, survey, program, z, zwarn, spectype FROM desi_dr1.zpix "
         f"WHERE targetid IN ({','.join(str(int(t)) for t in tids)}) "
         "AND survey = 'sv3' AND program = 'bright'")  # fmt: skip
    zt = query(lambda: TapPlus(url=DL_TAP).launch_job(q).get_results())
    zmap = {int(r["targetid"]): (float(r["z"]), int(r["zwarn"]), str(r["spectype"])) for r in zt}
    lines = [f"# petal {petal} of EXPID {expid}: TARGETID, z, ZWARN, SPECTYPE, ratio there, "
             "ratios of the same TARGETID in other exposures of healpix 10155"]  # fmt: skip
    for t in tids:
        here = d["all_ratio"][same & (d["all_targetid"] == t)]
        elsewhere = d["all_ratio"][(d["all_targetid"] == t) & (d["all_expid"] != expid)]
        z, zw, st = zmap.get(int(t), (np.nan, -1, "?"))
        lines.append(f"{int(t)} z {z:.4f} zwarn {zw} {st:7s} ratio {np.round(here, 3).tolist()} "
                     f"elsewhere {np.round(elsewhere, 3).tolist()}")  # fmt: skip
    (OUT / "neighbour_checks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# %%
def petal(cams=("z6", "z5")):
    """Excess flux at observed 8106-8360 A for every fibre of EXPID 85210, petals 6 and 5 (control)."""
    import urllib.request

    from astropy.io import fits

    use_apj_style()
    base = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/exposures/20210417/00085210"
    cache = PATHS["DATA"] / "iron_exposures" / "00085210"
    cache.mkdir(parents=True, exist_ok=True)
    band = (8106.0, 8360.0)
    side = ((7854.0, 8043.0), (8486.0, 8676.0))
    fig, ax = plt.subplots(len(cams), 1, figsize=(12, 3.2 * len(cams)), sharey=True)
    lines = ["# EXPID 85210 cframes: median flux at observed 8106-8360 A minus the median of the "
             "sidebands 7854-8043 and 8486-8676 A, 1e-17 cgs, by OBJTYPE"]  # fmt: skip
    for a, cam in zip(np.atleast_1d(ax), cams, strict=True):
        path = cache / f"cframe-{cam}-00085210.fits.gz"
        if not path.exists():
            urllib.request.urlretrieve(f"{base}/{path.name}", path)
        with fits.open(path) as h:
            w = np.asarray(h["WAVELENGTH"].data, float)
            f, iv = np.asarray(h["FLUX"].data, float), np.asarray(h["IVAR"].data, float)
            mk = np.asarray(h["MASK"].data)
            fm = h["FIBERMAP"].data
            fib, obj = np.asarray(fm["FIBER"]), np.asarray(fm["OBJTYPE"]).astype(str)
        ok = (iv > 0) & (mk == 0)
        b = (w > band[0]) & (w < band[1])
        sd = ((w > side[0][0]) & (w < side[0][1])) | ((w > side[1][0]) & (w < side[1][1]))
        ex = np.array([np.median(f[i][ok[i] & b]) - np.median(f[i][ok[i] & sd])
                       if (ok[i] & b).sum() > 50 and (ok[i] & sd).sum() > 50 else np.nan
                       for i in range(len(f))])  # fmt: skip
        for kind, c in (("TGT", "#2166ac"), ("SKY", "#b2182b")):
            m = obj == kind
            a.plot(fib[m], ex[m], ".", color=c, ms=4, label=f"{kind} (N {m.sum()})")
            x = ex[m & np.isfinite(ex)]
            q = np.percentile(x, [16, 50, 84])
            lines.append(f"{cam} {kind}: N {x.size}, median {q[1]:.3f}, 16-84% {q[0]:.3f} to "
                         f"{q[2]:.3f}, fraction > 1 {np.mean(x > 1):.2f}")  # fmt: skip
        a.axhline(0, color="0.4", lw=0.6)
        a.set_title(f"cframe-{cam}-00085210", fontsize=10)
        a.set_ylabel("band minus sidebands")
        a.legend(frameon=False, fontsize=8)
    np.atleast_1d(ax)[-1].set_xlabel("FIBER")
    np.atleast_1d(ax)[0].set_ylim(-3, 8)
    fig.savefig(OUT / "petal_excess.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    (OUT / "petal_excess.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT / 'petal_excess.png'}")


if __name__ == "__main__":
    main()
    if (OUT / f"{TID}_neighbours.npz").exists():
        neighbours()
        neighbour_checks()
    petal()
