"""Targets of the recoadd with the patched flux calibration ivar.

39627770174637084 is the #14 target these scripts were written for. The other four are the
free-line candidates of experiments/2026-10-18_candidate_recoadd (exposures.txt there). Each entry
gives the redshift, the DR1 healpix, the output folder and the exposures as {expid: (night, petal,
fiber)}. The reference redshift is the SV3 ZCAT_PRIMARY row of zall-pix-iron-bgs.fits, the others
are fastspec-iron-sv3-bright METADATA Z. It imports no hubersed code, so that fluxcal_rerun.py can
use it from the desispec 0.56.2 environment.
"""

from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
REFERENCE = 39627770174637084
CANDIDATE_OUT = RESULTS / "2026-10-18_candidate_resolution_matrix"

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
}


def target(tid):
    """Return the entry of ``tid``."""
    return TARGETS[int(tid)]
