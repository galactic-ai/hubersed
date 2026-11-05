"""Save 10 Gyr SSPs, MIST isochrones and library-swapped SSPs from the FSPS build in use.

FSPS fixes its stellar library at compile time, so this runs once per build. MILES runs with the
miles_mac venv, C3K_HR with the hubersed venv. Settings are Kroupa (imf_type 2), no ISM dust,
no nebular emission, AGB dust at the FSPS default.

The swap sums IMF weight times the FSPS stellar spectrum over the 10 Gyr isochrone of node a,
with the spectral library of node b. Diagonal entries rebuild the node SSPs.

COMMAND: see analysis.py.
OUTPUT: results/2026-10-30_miles_zcap_test/ssp_<name>.npz.
"""

import argparse
from pathlib import Path

import numpy as np

# The MILES venv has no hubersed, so the results path comes from this file's location.
OUT = Path(__file__).resolve().parents[2] / "results" / "2026-10-30_miles_zcap_test"
TAGE = 10.0  # Gyr
LOGZSOL = (0.0, 0.2, 0.35, 0.5)
ZMET = (11, 12, 13)  # MIST nodes at log Z/Zsun 0.0, +0.25, +0.5
SP_KW = dict(imf_type=2, dust_type=4, dust1=0.0, dust2=0.0, add_neb_emission=False)


def isochrone(sp, zmet, name):
    """Return the 10 Gyr isochrone rows at node zmet as a structured array."""
    sp.params["zmet"] = zmet
    iso = sp.isochrones(outfile=f"zcap_test_{name}_{zmet}")
    return iso[np.isclose(iso["age"], np.log10(TAGE * 1e9), atol=1e-3)]


def swap_ssp(sp, iso, zmet_spec):
    """Sum IMF weight times stellar spectrum over an isochrone, with the library of zmet_spec."""
    sp.params["zmet"] = zmet_spec
    out = np.zeros(sp.wavelengths.size)
    for r in iso:
        w = 10 ** r["log(weight)"]
        s = sp._get_stellar_spectrum(
            r["mact"],
            r["logt"],
            10 ** r["logl"],
            r["logg"],
            r["phase"],
            r["composition"],
            mdot=r["log(mdot)"],
            weight=w,
            peraa=True,
        )
        out += w * s
    return out


def main():
    """Compute and save the SSP grid, the node SSPs, the isochrones and the swap grid."""
    import fsps

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="output tag, miles or c3k_hr")
    name = ap.parse_args().name

    sp1 = fsps.StellarPopulation(zcontinuous=1, **SP_KW)
    spec = np.zeros((len(LOGZSOL), sp1.wavelengths.size))
    for i, z in enumerate(LOGZSOL):
        sp1.params["logzsol"] = z
        _, spec[i] = sp1.get_spectrum(tage=TAGE, peraa=True)

    sp0 = fsps.StellarPopulation(zcontinuous=0, **SP_KW)
    node = np.zeros((len(ZMET), sp0.wavelengths.size))
    for i, zm in enumerate(ZMET):
        sp0.params["zmet"] = zm
        _, node[i] = sp0.get_spectrum(tage=TAGE, peraa=True)

    isos = [isochrone(sp0, zm, name) for zm in ZMET]
    swap = np.zeros((len(ZMET), len(ZMET), sp0.wavelengths.size))
    for a, iso in enumerate(isos):
        for b, zm in enumerate(ZMET):
            swap[a, b] = swap_ssp(sp0, iso, zm)

    libs = [x.decode() if isinstance(x, bytes) else str(x) for x in sp1.libraries]
    cols = ("mact", "logl", "logt", "logg", "phase", "log(weight)")
    key = {c: c.replace("log(weight)", "logweight") for c in cols}
    iso_arrays = {
        f"iso{zm}_{key[c]}": iso[c] for zm, iso in zip(ZMET, isos, strict=True) for c in cols
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ssp_{name}.npz"
    np.savez(
        path,
        wave=sp1.wavelengths,
        spec=spec,
        logzsol=LOGZSOL,
        node=node,
        zmet=ZMET,
        node_logzsol=np.log10(sp0.zlegend[np.array(ZMET) - 1] / sp0.zlegend[10]),
        swap=swap,
        libraries=np.array(libs),
        fsps_file=fsps.__file__,
        **iso_arrays,
    )
    print(f"{name}: libraries {libs}, fsps {fsps.__file__}, wrote {path}")


if __name__ == "__main__":
    main()
