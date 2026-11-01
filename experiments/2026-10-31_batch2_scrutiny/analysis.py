"""Scrutiny of follow-up outliers 6-14 of follow28 (#17): broad lines, BPT, WHAN and masked Halpha.

QUESTION: which of the nine are AGN, star-forming or quiescent, and which can go to a pocoMC fit?
HYPOTHESIS: none registered. Visual calls (Nikhil, 2026-10-31): 06 good, 07 AGN, 08 AGN, 09
star-forming, 10 good, 11 good, 12 good, 13 AGN, 14 good.
INPUTS: FastSpecFit iron sv3 bright; spender chunk coadds (load_by_index) for the Halpha mask;
survivors.txt for logM, Dn4000 and the exposure count. BPT lines Kauffmann+03 (astro-ph/0304239) and
Kewley+01 (astro-ph/0106324), WHAN classes Cid Fernandes+11 (1012.4426), checked in the TeX.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_batch2_scrutiny/analysis.py``.
RESULT: broad Halpha (S/N 28-34, sigma 671-1074 km/s) in 07, 09, 13: AGN. 14 composite BPT, WHAN
strong AGN. 06, 10, 11 star-forming (06 and 10 at logM 7.6 and 7.4); 08 star-forming (single exposure,
Halpha masked); 12 retired (Dn4000 1.95, Halpha EW 0.6, single exposure). Fits (Nikhil, 2026-10-31):
06, 10, 11, 12 in 2026-10-31_poco_batch2; not 08 and 14.
Assumptions: FastSpecFit 2.5.1 fluxes; BPT needs S/N >= 3 in all four lines.
FIGURES: results/2026-10-31_batch2_scrutiny/ scrutiny.txt.
"""

# %%
import numpy as np
from astropy.io import fits
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-31_batch2_scrutiny"
BATCH = (39633123138340748, 39627793855676586, 39627751501600279, 39627836700492877,
         39627896897143091, 39633443809656951, 39633427376375905, 39628362662019515,
         39633123142533240)  # fmt: skip
FIRST = 6  # follow28 number of the first target
SN = 3.0


def line(r, name):
    """Return flux, S/N and sigma of a FastSpecFit line, NaN S/N without ivar."""
    f, iv = float(r[f"{name}_FLUX"]), float(r[f"{name}_FLUX_IVAR"])
    return f, (f * np.sqrt(iv) if iv > 0 else np.nan), float(r[f"{name}_SIGMA"])


def bpt(r):
    """Return the [NII] BPT class at S/N >= SN for all four lines, or why there is none."""
    v = {k: line(r, k) for k in ("HALPHA", "NII_6584", "HBETA", "OIII_5007")}
    low = [k for k, (f, s, _) in v.items() if not (s >= SN and f > 0)]
    if low:
        return f"none ({', '.join(low)} below S/N {SN:g})"
    x = np.log10(v["NII_6584"][0] / v["HALPHA"][0])
    y = np.log10(v["OIII_5007"][0] / v["HBETA"][0])
    if x >= 0.47 or y > 0.61 / (x - 0.47) + 1.19:
        c = "AGN"
    elif x >= 0.05 or y > 0.61 / (x - 0.05) + 1.3:
        c = "composite"
    else:
        c = "star-forming"
    return f"{c} (x {x:.2f}, y {y:.2f})"


def whan(r):
    """Return the WHAN class from Halpha and [NII] EWs and log [NII]/Halpha."""
    ha, nii = float(r["HALPHA_EW"]), float(r["NII_6584_EW"])
    if not (float(r["HALPHA_FLUX_IVAR"]) > 0):
        return "none (no Halpha)"
    if ha < 0.5 and nii < 0.5:
        return f"passive (EW {ha:.1f})"
    if ha < 3:
        return f"retired (EW {ha:.1f})"
    x = (
        np.log10(float(r["NII_6584_FLUX"]) / float(r["HALPHA_FLUX"]))
        if r["NII_6584_FLUX"] > 0
        else -9
    )
    if x < -0.4:
        return f"star-forming (EW {ha:.1f}, x {x:.2f})"
    return f"{'strong' if ha > 6 else 'weak'} AGN (EW {ha:.1f}, x {x:.2f})"


def main():
    """Write one block per target."""
    OUT.mkdir(parents=True, exist_ok=True)
    surv = {}
    for s in (
        (PATHS["RESULTS"] / "2026-10-31_screen_all" / "survivors.txt").read_text().splitlines()
    ):
        p = s.split()
        if p and p[0].isdigit() and len(p[0]) == 17:
            surv[int(p[0])] = dict(logm=p[2], dn=p[3], nexp=p[7], status=" ".join(p[10:]))
    wave = DESI._wave_obs.numpy()
    out = [f"# follow28 targets {FIRST}-{FIRST + len(BATCH) - 1}: FastSpecFit iron sv3 bright"]
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True) as h:
        meta, fs = h["METADATA"].data, h["FASTSPEC"].data
        for k, t in enumerate(BATCH, FIRST):
            i = np.flatnonzero(meta["TARGETID"] == t)[0]
            r, z = fs[i], float(meta["Z"][i])
            _, ivar, _, _ = load_by_index(int(tids_to_indices(np.array([t]))[0]))
            ha_px = np.abs(wave - 6562.8 * (1 + z)) < 4
            masked = float(np.mean(np.asarray(ivar)[ha_px] <= 0)) if ha_px.any() else np.nan
            b = {n: line(r, n) for n in ("HALPHA_BROAD", "HBETA_BROAD", "MGII_2803")}
            s = surv[t]
            out.append(
                f"{k:02d} {t} z {z:.3f} logM {s['logm']} Dn4000 {s['dn']} nexp {s['nexp']} ({s['status']})\n"
                f"   broad Ha S/N {b['HALPHA_BROAD'][1]:.1f} sigma {b['HALPHA_BROAD'][2]:.0f}; "
                f"broad Hb S/N {b['HBETA_BROAD'][1]:.1f} sigma {b['HBETA_BROAD'][2]:.0f}; "
                f"MgII2803 S/N {b['MGII_2803'][1]:.1f} sigma {b['MGII_2803'][2]:.0f}\n"
                f"   BPT {bpt(r)}; WHAN {whan(r)}; Halpha pixels masked by spender {masked:.0%}"
            )  # fmt: skip
    (OUT / "scrutiny.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
