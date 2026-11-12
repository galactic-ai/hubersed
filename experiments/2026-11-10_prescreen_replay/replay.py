"""Replay the prescreen that wrote results/prescreened_sample_v1.npz and record the step that
removes each DESI SV3 bright spectrum.

The original driver ran as an inline heredoc on 2026-09-05 (session a3ef5882, backup transcript
~/claude-backup/projects/-Users-ng27753-Documents-UberSED/a3ef5882-....jsonl, lines 2135 and 2468).
This file repeats its logic unchanged. The inputs are copies of that session's intermediates from
~/claude-backup/jobs/a3ef5882/tmp/ in results/2026-11-10_prescreen_replay/inputs/ (md5 in
inputs_md5.txt). The scripts that made the Gaia, SGA, morphology, BPT and shred inputs are not
ported yet.

Steps, in order:
1. encoded: in the DESI set the flows scored (raw_sets.npz "desi", 114,693). The latents keep
   spectra with median per-pixel S/N > 3 (scripts/get_latent_space.py l.77-84: median of
   flux * sqrt(ivar) over unmasked pixels of the 7781-pixel observed grid, 3600-9824 A).
2. in the fastspecfit SV3 bright VAC.
3. VAC z in [0.01, 0.6]; 4. pipeline z agrees with the VAC z within 0.01 (1 + z).
5. Gaia star; 6. SGA; 7. PSF morphology with no gas; 8. BPT AGN; 9. shredded photometry.

COMMAND: ``uv run python experiments/2026-11-10_prescreen_replay/replay.py`` from the repository
root on the Mac.
OUTPUT: results/2026-11-10_prescreen_replay/prescreen_steps.npz with tid, step (0 kept, else the
step number above that removes it), snr and z from data/dataset_z_snr.npz, for all 254,976
spectra in that file.
"""

import csv

import numpy as np
from astropy.io import fits
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-10_prescreen_replay"
IN = OUT / "inputs"
STEPS = {
    0: "kept",
    1: "not encoded (S/N <= 3)",
    2: "no fastspecfit row",
    3: "z outside 0.01-0.6",
    4: "pipeline z disagrees",
    5: "Gaia star",
    6: "SGA",
    7: "PSF + no gas",
    8: "BPT AGN",
    9: "shred",
}


def gaia_flags():
    """Return the Gaia star flag per target, as in the 2026-09-05 driver."""
    t = Table.read(IN / "gaia_xmatch.fits")
    order = np.argsort(np.asarray(t["angDist"], float))
    seen, near = set(), {}
    for i in order:
        tid = int(t["tid"][i])
        if tid not in seen:
            seen.add(tid)
            near[tid] = i

    def col(i, name):
        v = t[name][i]
        try:
            v = float(v)
        except Exception:
            return np.nan
        return v if np.isfinite(v) else np.nan

    gflag = {}
    for tid, i in near.items():
        sep = col(i, "angDist")
        plx, eplx = col(i, "Plx"), col(i, "e_Plx")
        plx_snr = plx / eplx if np.isfinite(plx) and np.isfinite(eplx) and eplx > 0 else np.nan
        pm = col(i, "PM")
        pmra, pmde = col(i, "pmRA"), col(i, "pmDE")
        epmra, epmde = col(i, "e_pmRA"), col(i, "e_pmDE")
        pme = np.hypot(pmra * epmra, pmde * epmde) / pm if np.isfinite(pm) and pm > 0 else np.nan
        pm_snr = pm / pme if np.isfinite(pme) and pme > 0 else np.nan
        ruwe, g = col(i, "RUWE"), col(i, "Gmag")
        onsrc = bool(
            sep < 1.0
            and np.isfinite(ruwe)
            and ruwe < 1.4
            and ((np.isfinite(plx_snr) and plx_snr > 5) or (np.isfinite(pm_snr) and pm_snr > 5))
        )
        wing = bool(
            np.isfinite(g)
            and g < 16.0
            and ((np.isfinite(plx_snr) and plx_snr > 5) or (np.isfinite(pm) and pm > 3.0))
        )
        gflag[tid] = onsrc or wing
    return gflag


def main():
    """Run the replay, check it against prescreened_sample_v1.npz and write the steps."""
    raw = np.load(IN / "raw_sets.npz")
    encoded = {int(x) for x in raw["desi"]}
    meta = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits")["METADATA"].data
    iv = {int(t): i for i, t in enumerate(meta["TARGETID"])}
    zvac = np.asarray(meta["Z"], float)
    zs = np.load(PATHS["DATA"] / "dataset_z_snr.npz")
    zpipe = dict(zip(zs["tid"].astype(np.int64).tolist(), zs["z"].tolist(), strict=True))

    # Steps 2-4 in the order of the flows' DESI set, as the driver did.
    k2 = []
    for t in raw["desi"]:
        t = int(t)
        if t not in iv:
            continue
        i = iv[t]
        if not (0.01 <= zvac[i] <= 0.6):
            continue
        zp = zpipe.get(t, np.nan)
        if np.isfinite(zp) and abs(zp - zvac[i]) / (1 + zvac[i]) > 0.01:
            continue
        k2.append(t)
    keep = np.load(IN / "prescreen_state.npz")["keep"]
    assert np.array_equal(np.array(k2, np.int64), keep), "steps 1-4 differ from prescreen_state"

    gflag = gaia_flags()
    k3 = {t for t in map(int, keep) if not gflag.get(t, False)}

    def flagged(name):
        f = np.load(IN / name)
        return {int(t) for t, x in zip(keep, f, strict=True) if x}

    k4 = k3 - flagged("sga_flag_full.npy")
    morph = {int(r["target_id"]): r["type"] for r in csv.DictReader(open(IN / "morph_bulk.csv"))}
    k6a = k4 - {t for t in flagged("weak_mask.npy") if morph.get(t, "") == "PSF"}
    k6b = k6a - flagged("bpt_agn_full.npy")
    kf = k6b - flagged("shred_flag_full.npy")

    ref = np.load(PATHS["RESULTS"] / "prescreened_sample_v1.npz")
    full, cont = {int(x) for x in raw["full"]}, {int(x) for x in raw["cont"]}
    assert np.array_equal(np.array(sorted(kf), np.int64), ref["keep"])
    assert np.array_equal(np.array(sorted(full & kf), np.int64), ref["full"])
    assert np.array_equal(np.array(sorted(cont & kf), np.int64), ref["cont"])

    ks2 = set(int(x) for x in keep)
    tid = zs["tid"].astype(np.int64)
    step = np.zeros(tid.size, np.int8)
    for j, t in enumerate(map(int, tid)):
        if t in kf:
            continue
        if t not in encoded:
            s = 1
        elif t not in iv:
            s = 2
        elif t not in ks2:
            z = zvac[iv[t]]
            s = 3 if not (0.01 <= z <= 0.6) else 4
        elif t not in k3:
            s = 5
        elif t not in k4:
            s = 6
        elif t not in k6a:
            s = 7
        elif t not in k6b:
            s = 8
        else:
            s = 9
        step[j] = s
    np.savez(OUT / "prescreen_steps.npz", tid=tid, step=step, snr=zs["snr"], z=zs["z"])
    for s, name in STEPS.items():
        print(f"{s} {name:24s} {np.sum(step == s):7d}")
    print("matches prescreened_sample_v1.npz (keep, full, cont)")


if __name__ == "__main__":
    main()
