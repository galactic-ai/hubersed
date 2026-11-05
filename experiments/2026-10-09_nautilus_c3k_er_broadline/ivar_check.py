"""Check the iron flux-calibration ivar of a target against the formula fixed in desispec.

QUESTION: does the iron cframe ivar of this target follow the old desispec 0.56 formula, and how far
off is it from the formula fixed in desispec PR 2422, with iron's own calibration ivar?

desispec 0.56.2 (iron) apply_flux_calibration divides the flux by the calibration vector C before
it computes the ivar, so Var(F') = 1/(ivar_F C^2) + F'^2/(ivar_C C^4). PR 2422 computes it from the
uncalibrated flux, Var(F') = 1/(ivar_F C^2) + F'^2/(ivar_C C^2). PR 2496 then found the iron ivar_C
itself too small (an error floor left in the final variance solve). So the fixed formula with iron
ivar_C is not the corrected ivar. This script only checks the reading and shows the size of each
term, before the calibration is rerun.

For each exposure and camera of this target it checks F/C against the cframe flux, the old formula
against the cframe ivar, and the old formula inverted for ivar_F against the sframe ivar
(sky-subtracted, before calibration). It then reports sigma_C / C and the inflation sqrt(Var_fixed /
Var_iron) in bins of S/N.

It was written for 39627770174637084 (exposures 87113 and 88269, petal 1), the default. --tid picks
another target of targets.py. The cameras follow each exposure's petal.

INPUTS: the iron cframe, sframe and fluxcalib files of the target's exposures and petal, under
data/desi_iron_exposures/<night>/<expid>/. SEED: none.

COMMAND: on an ls6 compute node from the repository root,
``.venv/bin/python experiments/2026-10-09_nautilus_c3k_er_broadline/ivar_check.py``, with
``--tid <targetid>`` for the others.

OUTPUT: ivar_check.txt in the target's output folder of targets.py.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from targets import REFERENCE, target

from hubersed.paths import PATHS

# IRON_EXPOSURES overrides the folder, for nodes that do not mount Corral
DATA = Path(os.environ.get("IRON_EXPOSURES", PATHS["DATA"] / "desi_iron_exposures"))


def load(night, expid, cam, TID):
    """Return the target's cframe, sframe and fluxcalib arrays for one camera."""
    d = DATA / night / f"{expid:08d}"
    with (
        fits.open(d / f"cframe-{cam}-{expid:08d}.fits.gz") as cf,
        fits.open(d / f"sframe-{cam}-{expid:08d}.fits.gz") as sf,
        fits.open(d / f"fluxcalib-{cam}-{expid:08d}.fits.gz") as fc,
    ):
        row = int(np.where(cf["FIBERMAP"].data["TARGETID"] == TID)[0][0])
        assert sf["FIBERMAP"].data["TARGETID"][row] == TID
        out = dict(
            row=row,
            fiber=int(cf["FIBERMAP"].data["FIBER"][row]),
            version=cf[0].header.get("DEPVER18", ""),
            Fp=cf["FLUX"].data[row].astype(float),
            iv=cf["IVAR"].data[row].astype(float),
            F=sf["FLUX"].data[row].astype(float),
            ivF=sf["IVAR"].data[row].astype(float),
            C=fc["FLUXCALIB"].data[row].astype(float),
            ivC=fc["IVAR"].data[row].astype(float),
        )
    return out


def main(TID=REFERENCE):
    t = target(TID)
    petals = ", ".join(str(p) for p in sorted({p for _, p, _ in t["exposures"].values()}))
    lines = [f"Iron flux-calibration ivar of {TID} (petal {petals})"]
    for expid, (night, petal, _) in t["exposures"].items():
        for cam in (f"{arm}{petal}" for arm in "brz"):
            x = load(night, expid, cam, TID)
            Fp, iv, F, ivF, C, ivC = (x[k] for k in ("Fp", "iv", "F", "ivF", "C", "ivC"))
            ok = (iv > 0) & (ivF > 0) & (ivC > 0) & (C > 0)
            var_old = 1 / (ivF * C**2) + (F / C) ** 2 / (ivC * C**4)
            var_fix = 1 / (ivF * C**2) + (F / C) ** 2 / (ivC * C**2)
            # invert the old formula for 1/(ivar_F C^2) from the cframe alone
            inv_term = 1 / iv - Fp**2 / (ivC * C**4)
            q = np.percentile
            lines.append(
                f"{expid} {cam} row {x['row']} fiber {x['fiber']} desispec {x['version']}: "
                f"|F/C / F'-1| 99% {q(abs(F[ok] / C[ok] / Fp[ok] - 1), 99):.1e}, "
                f"old formula / cframe var 1-99% {q(var_old[ok] * iv[ok] - 1, 1):+.1e} "
                f"{q(var_old[ok] * iv[ok] - 1, 99):+.1e}, inverted ivar_F / sframe 1-99% "
                f"{q(1 / (inv_term[ok] * C[ok] ** 2) / ivF[ok] - 1, 1):+.1e} "
                f"{q(1 / (inv_term[ok] * C[ok] ** 2) / ivF[ok] - 1, 99):+.1e}"
            )
            fc_ = 1 / (C * np.sqrt(ivC))
            infl = np.sqrt(var_fix * iv)
            sn = Fp * np.sqrt(iv)
            row = f"   C median {np.median(C[ok]):.1f}, sigma_C/C median {np.median(fc_[ok]):.4f};"
            for lo, hi in ((0, 5), (5, 20), (20, 40), (40, 80), (80, np.inf)):
                m = ok & (sn >= lo) & (sn < hi)
                if m.sum():
                    row += f" S/N {lo:g}-{hi:g}: n {m.sum()} infl {np.median(infl[m]):.2f};"
            lines.append(row)
    text = "\n".join(lines) + "\n"
    print(text)
    t["out"].mkdir(parents=True, exist_ok=True)
    (t["out"] / "ivar_check.txt").write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, default=REFERENCE, help="target of targets.py")
    main(ap.parse_args().tid)
