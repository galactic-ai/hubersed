"""Save dust-free SSP spectra from the FSPS build on PYTHONPATH, for the C3K against MILES test.

QUESTION: is the C3K continuum bowed against MILES at the same age and metallicity? The fits of
39633433328091196 imply the C3K best model is about 6% fainter at rest 3400 A and 4% at 7000 A,
relative to V, than the MILES one (2026-10-29_continuum_39633433328091196_map/reddening.py).

Both builds use MIST isochrones (build_miles_fsps.sh; build_er_fsps.sh). SSPs have the prospect
default Kroupa IMF (imf_type 2), no dust, no nebular emission and zcontinuous 1. FSPS MILES spectra stop
at Z 0.030 (log Z/Zsun +0.2) and are not extrapolated, so logzsol -0.3 and 0 lie inside its grid and
+0.3 shows the cap.

COMMAND: on a stampede3 compute node from the repository root, through ssp_s3.slurm.
OUTPUT: results/2026-10-29_c3k_vs_miles_ssp/ssp_<library>.npz.
"""

import argparse

import numpy as np

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-29_c3k_vs_miles_ssp"
AGES = (1.0, 3.0, 6.0, 10.0, 12.0)  # Gyr
LOGZSOL = (-0.3, 0.0, 0.3)


def main():
    """Compute the SSP grid and save it with the library name and resolution."""
    import fsps

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="output tag, for example c3k_hr or miles")
    name = ap.parse_args().name
    sp = fsps.StellarPopulation(
        zcontinuous=1, imf_type=2, dust_type=4, dust1=0.0, dust2=0.0, add_neb_emission=False
    )
    spec = np.zeros((len(LOGZSOL), len(AGES), sp.wavelengths.size))
    for i, z in enumerate(LOGZSOL):
        sp.params["logzsol"] = z
        for j, age in enumerate(AGES):
            _, spec[i, j] = sp.get_spectrum(tage=age, peraa=True)
    libs = [x.decode() if isinstance(x, bytes) else str(x) for x in sp.libraries]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ssp_{name}.npz"
    np.savez(
        path,
        wave=sp.wavelengths,
        spec=spec,
        ages=AGES,
        logzsol=LOGZSOL,
        libraries=np.array(libs),
        resolutions=sp.resolutions,
        fsps_file=fsps.__file__,
    )
    print(f"{name}: libraries {libs}, fsps {fsps.__file__}, wrote {path}")


if __name__ == "__main__":
    main()
