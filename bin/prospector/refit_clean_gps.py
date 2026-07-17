"""
Confirmation test: refit the 3 clean compact green peas with the lowered
logzsol prior floor (-2; check fit_config.py line ~32 reads mini=-2.0).

Mirrors map_chi2.py exactly (same map_chi2_one), Cue + DESI LSF.

Budget: CHEAP by default (cont 1 / full 1 seed, maxfev 3k) -- the log already
established rich ~= cheap at floor -1 (clean GPs 12.2->11.1, 10.9->8.8,
13.7->12.0), so cheap gives a fast first answer (~1-3 min total). Pass --rich
(cont 3 / full 5 / maxfev 30k) to re-confirm any GP that RE-RAILS at -2, where
an optimizer artifact would otherwise be ambiguous.

Saves a _full.pkl in the SAME schema as the worst8 baseline (now including
theta_dict, keyed by param name -> no 26-vs-18 misalignment).

Run from bin/prospector/ with the project venv:
    python refit_clean_gps.py            # cheap, fast
    python refit_clean_gps.py --rich     # decisive, slower
"""

import sys
import pickle
import numpy as np

from map_chi2 import map_chi2_one, tids_to_indices
import parameter_file as P  # noqa: F401  (ensures same module init path as map_chi2)
from hubersed.paths import PATHS

# The 3 clean compact green peas (the ones that railed logzsol at the -1 floor)
CLEAN_GP_TIDS = np.array(
    [
        39627788302422585,
        39627769662934656,
        39633136555921233,
    ],
    dtype=np.int64,
)

RICH = "--rich" in sys.argv
# cheap (matches map_chi2.py default) vs rich (matches map_chi2.py --rich)
CONT_NSEEDS, FULL_NSEEDS, MAXFEV = (3, 5, 30_000) if RICH else (1, 1, 3_000)


def main():
    print(
        f"budget: {'RICH (cont 3 / full 5 / maxfev 30k)' if RICH else 'CHEAP (cont 1 / full 1 / maxfev 3k)'}"
    )
    idxs = tids_to_indices(CLEAN_GP_TIDS)
    # self-check: load_by_index round-trips the requested TARGETID
    from map_chi2 import load_by_index

    for g, t in zip(idxs, CLEAN_GP_TIDS):
        assert load_by_index(int(g))[3] == int(t), f"TID->index mismatch for {t}"
    print(f"self-check passed; fitting {len(idxs)} clean GPs (Cue+LSF, RICH)")

    out = []
    for k, (gi, tid) in enumerate(zip(idxs, CLEAN_GP_TIDS)):
        r = map_chi2_one(
            int(gi),
            use_cue=True,
            cont_nseeds=CONT_NSEEDS,
            full_nseeds=FULL_NSEEDS,
            maxfev=MAXFEV,
        )
        out.append(r)
        if r.get("status") == "ok":
            lz = float(np.asarray(r["theta_dict"]["logzsol"]).ravel()[0])
            print(
                f"[{k + 1}/{len(idxs)}] TID {tid}  chi2_red={r['chi2_red']:.2f}  logzsol={lz:.4f}"
            )
        else:
            print(f"[{k + 1}/{len(idxs)}] TID {tid}  status={r.get('status')}")

    full = {
        "wave": np.asarray(P.WAVE_OBS, dtype=np.float32),
        "use_cue": True,
        "rich": RICH,
        "logzsol_floor": -2.0,
        "results": out,
    }
    tag = "rich" if RICH else "cheap"
    outpath = PATHS["RESULTS"] / f"map_chi2_cue_cleanGP3_floor2_lsf_{tag}_full.pkl"
    with open(outpath, "wb") as f:
        pickle.dump(full, f)
    print(f"saved -> {outpath}")


if __name__ == "__main__":
    main()
