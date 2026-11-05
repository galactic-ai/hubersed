"""Attenuation that the ER MAP and MILES max-L dust parameters of 39633433328091196 imply.

FSPS dust_type 4 (Kriek and Conroy 2013) with dust1 = 0, so only the diffuse screen acts. A_lambda is
-2.5 log10 of the ratio of a 10 Gyr solar SSP with and without dust2, read at 4400 and 5500 A. The
SSP only carries the curve, so its age and metallicity do not matter.

COMMAND: from the repository root on the Mac,
``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-29_continuum_39633433328091196_map/reddening.py``.
OUTPUT: printed.
"""

import numpy as np

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map"
RUNS = {
    "C3K ER MAP": OUT / "39633433328091196_map.npz",
    "MILES max-L": PATHS["RESULTS"]
    / "2026-10-08_continuum_outlier"
    / "39633433328091196_miles_miles_neboff_nodeg_pred.npz",
}
SHAPE_WAVE = np.array([3400.0, 3800.0, 4400.0, 5500.0, 6200.0, 7000.0])  # rest A, fitted range


def main():
    """Print A_V, A_B, E(B-V) and A_B/A_V for each fit."""
    import fsps

    sp = fsps.StellarPopulation(zcontinuous=1, logzsol=0.0, dust_type=4, dust1=0.0, dust2=0.0)
    wave, clean = sp.get_spectrum(tage=10.0, peraa=True)
    rel = {}
    for name, path in RUNS.items():
        d = np.load(path)
        p = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
        sp.params["dust2"] = float(p["dust2"])
        sp.params["dust_index"] = float(p["dust_index"])
        _, dusty = sp.get_spectrum(tage=10.0, peraa=True)
        a = -2.5 * np.log10(dusty / clean)
        a_b, a_v = np.interp(4400.0, wave, a), np.interp(5500.0, wave, a)
        print(
            f"{name:12s} dust2 {p['dust2']:.3f} dust_index {p['dust_index']:+.3f}: "
            f"A_V {a_v:.3f}, A_B {a_b:.3f}, E(B-V) {a_b - a_v:.3f}, A_B/A_V {a_b / a_v:.3f}"
        )
        rel[name] = np.interp(SHAPE_WAVE, wave, a) - a_v
    # A_lambda - A_V over the fitted rest range: the shape left after logmass takes the grey part
    print("A_lambda - A_V in mag at rest " + " ".join(f"{w:.0f}" for w in SHAPE_WAVE) + " A")
    for name, r in rel.items():
        print(f"  {name:12s} " + " ".join(f"{x:+.3f}" for x in r))
    a, b = rel.values()
    print(f"  {'difference':12s} " + " ".join(f"{x:+.3f}" for x in a - b))


if __name__ == "__main__":
    main()
