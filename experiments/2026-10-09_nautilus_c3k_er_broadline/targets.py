"""Targets of the recoadd with the patched flux calibration ivar.

39627770174637084 is the #14 target these scripts were written for. The other four are the
free-line candidates of experiments/2026-10-18_candidate_recoadd (exposures.txt there). Each entry
gives the redshift, the DR1 healpix, the output folder and the exposures as {expid: (night, petal,
fiber)}. The reference redshift is the SV3 ZCAT_PRIMARY row of zall-pix-iron-bgs.fits, the others
are fastspec-iron-sv3-bright METADATA Z. It imports no hubersed code, so that fluxcal_rerun.py can
use it from the desispec 0.56.2 environment.

39627757533007793 is the continuum outlier of experiments/2026-10-29_continuum_recoadd, observed in
SV3 and in the main survey. The environment variable RECOADD_SURVEY=main picks the main-survey
entries of MAIN_TARGETS, and RECOADD_OUT replaces an entry's output folder, for example to rerun
the reference elsewhere. Environment variables reach the child processes of fluxcal_rerun.py.
"""

import os
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
REFERENCE = 39627770174637084
CANDIDATE_OUT = RESULTS / "2026-10-18_candidate_resolution_matrix"
CONTINUUM_OUT = RESULTS / "2026-10-29_continuum_recoadd"

TARGETS = {
    REFERENCE: dict(
        z=0.025319,
        healpix=26004,
        out=RESULTS / "2026-10-09_nautilus_c3k_er_broadline" / str(REFERENCE),
        exposures={87113: ("20210503", 1, 724), 88269: ("20210513", 1, 704)},
    ),
    39628357133926597: dict(
        z=0.030801724143882887,
        healpix=28115,
        out=CANDIDATE_OUT / "39628357133926597",
        exposures={
            87470: ("20210506", 2, 1028),
            87936: ("20210510", 2, 1495),
            88146: ("20210512", 2, 1119),
        },
    ),
    39633278973513279: dict(
        z=0.025612783146619068,
        healpix=11404,
        out=CANDIDATE_OUT / "39633278973513279",
        exposures={
            84643: ("20210413", 8, 4053),
            84644: ("20210413", 8, 4053),
            84816: ("20210414", 8, 4053),
            86515: ("20210428", 8, 4092),
        },
    ),
    39632961393395307: dict(
        z=0.03441589211270936,
        healpix=10512,
        out=CANDIDATE_OUT / "39632961393395307",
        exposures={
            83535: ("20210405", 2, 1090),
            84818: ("20210414", 2, 1109),
            84819: ("20210414", 2, 1109),
            85094: ("20210416", 2, 1223),
            86503: ("20210428", 2, 1308),
            86841: ("20210501", 2, 1310),
        },
    ),
    39633001071509599: dict(
        z=0.014973174193937393,
        healpix=10518,
        out=CANDIDATE_OUT / "39633001071509599",
        exposures={83535: ("20210405", 5, 2911)},
    ),
    # z of the SV3 redrock fit (spender chunk), exposure from desi_dr1.ztile/exposure/fiberassign
    39627757533007793: dict(
        z=0.17071714997291565,
        healpix=25598,
        out=CONTINUUM_OUT / "sv3",
        exposures={83526: ("20210405", 0, 229)},
    ),
}
MAIN_TARGETS = {
    # z of the main row of desi_dr1.zpix (Astro Data Lab TAP, 2026-10-28)
    39627757533007793: dict(
        z=0.1706559587111022,
        healpix=25598,
        out=CONTINUUM_OUT / "main",
        exposures={122064: ("20220210", 1, 523)},
    ),
}


def target(tid):
    """Return the entry of ``tid``, from MAIN_TARGETS if RECOADD_SURVEY is main.

    RECOADD_OUT, if set, replaces the output folder of the entry.
    """
    survey = os.environ.get("RECOADD_SURVEY", "sv3")
    assert survey in ("sv3", "main"), f"RECOADD_SURVEY {survey} is not sv3 or main"
    entry = dict((MAIN_TARGETS if survey == "main" else TARGETS)[int(tid)])
    if os.environ.get("RECOADD_OUT"):
        entry["out"] = Path(os.environ["RECOADD_OUT"])
    return entry
