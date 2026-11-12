"""Targets and per-exposure inputs for the data-quality screen (plan
~/.claude/plans/data_quality_screen_plan.md, first version: fibre QA, exposure conditions, Gaia).

Targets: the old (310 run) and new (v2 mocks) candidates after the prescreen, before SIMBAD (flag
rule (6 & 10 & 15) | (cont10 & cont15)), plus a control of 2,000 random parent galaxies flagged by
neither (seed 0). Fetched from the Astro Data Lab: desi_dr1.ztile rows (survey sv3, program bright)
of every target, and desi_dr1.exposure rows of their tiles. Gaia DR3 sources within 5 arcsec from
the ESA Gaia archive (cross-match on uploaded positions).
COMMAND: ``uv run --no-sync python experiments/2026-11-10_data_quality_screen/fetch.py``.
OUTPUT: results/2026-11-10_data_quality_screen/{targets.npz, ztile.fits, exposure.fits, gaia5.fits}.
"""

import warnings

import astropy.units as u
import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table, vstack
from astroquery.utils.tap.core import TapPlus

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-10_data_quality_screen"
FLOWS = {
    "old": RES / "noised_cue_meanzero_wide_flow",
    "new": RES / "2026-11-09_mocks_v2_outliers" / "flows",
}
TAGS = ("6", "10", "15", "cont10", "cont15")
N_CONTROL = 2000
GAIA_RADIUS = 5.0  # arcsec


def candidates(folder, keep):
    """Return the flag-rule candidates of the flows in folder, inside the parent."""
    fl = {}
    for t in TAGS:
        d = torch.load(folder / f"desi_outliers_flow_nsf_{t}latent_snr3.pt", weights_only=False)
        fl[t] = {int(x) for x in d["outlier_target_ids"]} & keep
    return (fl["6"] & fl["10"] & fl["15"]) | (fl["cont10"] & fl["cont15"])


def chunks(ids, n=500):
    """Yield comma-joined TARGETID chunks for IN clauses."""
    ids = np.asarray(ids, np.int64)
    for i in range(0, ids.size, n):
        yield ",".join(map(str, ids[i : i + n]))


def main():
    """Write the target list and the fetched tables."""
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    keep = set(np.load(RES / "prescreened_sample_v1.npz")["keep"].tolist())
    old, new = candidates(FLOWS["old"], keep), candidates(FLOWS["new"], keep)
    pool = np.array(sorted(keep - old - new), np.int64)
    control = np.random.default_rng(0).choice(pool, N_CONTROL, replace=False)
    tids = np.array(sorted(old | new | set(control.tolist())), np.int64)
    np.savez(
        OUT / "targets.npz",
        old=np.array(sorted(old), np.int64),
        new=np.array(sorted(new), np.int64),
        control=np.sort(control),
    )
    print(f"old {len(old)}, new {len(new)}, control {control.size}, all {tids.size}", flush=True)

    tap = TapPlus(url="https://datalab.noirlab.edu/tap", verbose=False)
    if (OUT / "ztile.fits").exists() and (OUT / "exposure.fits").exists():
        zt = Table.read(OUT / "ztile.fits")
        print(f"ztile rows {len(zt)} (saved)", flush=True)
    else:
        zt = fetch_ztile(tap, tids)
    pos_and_gaia(tids)


def fetch_ztile(tap, tids):
    """Fetch and write the ztile and exposure tables; return ztile."""
    zt = vstack(
        [
            tap.launch_job(
                "SELECT targetid, tileid, survey, program, coadd_fiberstatus, coadd_numexp, "
                "coadd_exptime, tsnr2_bgs, std_fiber_ra, std_fiber_dec, mean_fiber_ra, mean_fiber_dec "
                f"FROM desi_dr1.ztile WHERE survey = 'sv3' AND program = 'bright' AND targetid IN ({c})"
            ).get_results()
            for c in chunks(tids)
        ]
    )
    zt.write(OUT / "ztile.fits", overwrite=True)
    tiles = np.unique(np.asarray(zt["tileid"], np.int64))
    print(
        f"ztile rows {len(zt)} on {tiles.size} tiles; targets with rows {np.unique(zt['targetid']).size}",
        flush=True,
    )

    ex = tap.launch_job(
        "SELECT tileid, night, expid, exptime, efftime_spec, sky_mag_g_spec, sky_mag_r_spec, "
        "transparency_gfa, seeing_etc, airmass, survey, program FROM desi_dr1.exposure "
        f"WHERE tileid IN ({','.join(map(str, tiles))})"
    ).get_results()
    ex.write(OUT / "exposure.fits", overwrite=True)
    print(f"exposure rows {len(ex)}", flush=True)
    return zt


def pos_and_gaia(tids):
    """Write Gaia DR3 sources within GAIA_RADIUS of every target to gaia5.fits."""
    meta = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)["METADATA"].data
    row = {int(t): i for i, t in enumerate(meta["TARGETID"])}
    pos = Table(
        {
            "tid": tids,
            "ra": np.array([float(meta["RA"][row[t]]) for t in tids]),
            "dec": np.array([float(meta["DEC"][row[t]]) for t in tids]),
        }
    )
    # ESA Gaia archive, async, 50 indexed cones per query. On 2026-11-10 the ESA upload cross-match
    # gave HTTP 500, CDS X-Match timed out and Data Lab TAP has no spatial index for boxes.
    from astropy.coordinates import SkyCoord
    from astroquery.gaia import Gaia

    parts = []
    for k in range(0, len(pos), 50):
        b = pos[k : k + 50]
        cones = " OR ".join(
            f"1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {r:.7f}, {d:.7f}, "
            f"{GAIA_RADIUS / 3600:.7f}))"
            for r, d in zip(b["ra"], b["dec"], strict=True)
        )
        g = Gaia.launch_job_async(
            "SELECT source_id, ra, dec, phot_g_mean_mag, parallax, parallax_error, pmra, pmdec, "
            f"ruwe FROM gaiadr3.gaia_source WHERE {cones}"
        ).get_results()
        if len(g) == 0:
            continue
        sc_t = SkyCoord(b["ra"], b["dec"], unit="deg")
        sc_g = SkyCoord(np.asarray(g["ra"], float), np.asarray(g["dec"], float), unit="deg")
        it, ig, sep, _ = sc_g.search_around_sky(sc_t, GAIA_RADIUS * u.arcsec)
        m = g[ig]
        m["tid"] = np.asarray(b["tid"])[it]
        m["sep"] = sep.arcsec
        parts.append(m)
    gj = vstack(parts)
    gj.write(OUT / "gaia5.fits", overwrite=True)
    print(f"Gaia sources within {GAIA_RADIUS}'': {len(gj)} for {np.unique(gj['tid']).size} targets")


if __name__ == "__main__":
    main()
