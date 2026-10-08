"""QUESTION: how does the fit quality of 37084 compare across the nautilus/MAP steps of 2026-09-24 to
2026-09-30 when measured on the same pixels?
METHOD: recompute the max-L chi2 of runs A (shared) and B (free) on (i) their own full-window pixels,
(ii) the 4448 sky-masked MILES-window pixels the 09-25/09-26 runs used, (iii) the 24 unmasked sky
pixels, and (iv) the red 7400.8-9000 A pixels. Read the C/O MAP chi2 from the 09-26 map_co pickles.
CAVEAT: from 09-25 on the data are degraded to 43.6 km/s but keep the raw DESI ivar (keep_ivar=True),
which overstates the noise of the smoothed flux, so chi2_nu is not in noise units (09-25 analysis.py
L145). The 09-24 MILES run (2.52) is on a different scale. Compare chi2 on a fixed pixel set instead.
INPUTS: results/2026-09-30_nautilus_c3k_broadline_split45/39627770174637084_broad_{shared,free}_*.h5
(or a copy of B via --b-dir), results/2026-09-26_nautilus_c3k_broadline/map_co/*_map_co_{miles,full}.pkl
SEED: none needed.
COMMAND: uv run python experiments/2026-09-30_nautilus_c3k_broadline_split45/chi2_ladder.py [--b-dir DIR]
RESULT: results/2026-09-30_nautilus_c3k_broadline_split45/chi2_ladder.txt. First run 2026-10-08 with B
at N_eff 453 (B_prelim_1002):

| step | setup | pixels | chi2_nu | chi2 |
|---|---|---|---|---|
| 09-24 | MILES lib, tau_in = t_H, nautilus | 4448 MILES window, undegraded | 2.52 | separate scale |
| 09-25 | C3K_HR, narrow only, nautilus | 4448 | 1.77 (quoted by Nikhil) | |
| 09-26 | C3K, broadline, nautilus (split 100) | 4448 | 1.32 (quoted by Nikhil) | |
| 09-26 | + C/O tied / free, MAP | 4448 | 0.980 / 0.979 | 4333.0 / 4329.6 |
| 09-26 | C/O tied / free, MAP, full window | 5840 (3511-9581 A) | 1.268 / 1.264 | 7370.5 / 7349.5 |
| 09-30 A | split 45, full window < 9000 A, sky unmasked | 5621 | 0.921 | 5150.0 |
| 09-30 A | same max-L point, MILES pixels only | 4448 | 0.998 | 4440.8 |
| 09-30 A | unmasked sky pixels | 24 | 0.861 per pixel | 20.7 |
| 09-30 A | red 7400.8-9000 A | 1032 | 0.551 per pixel | 568.6 |
| 09-30 B | f_forb free (prelim, N_eff 453) | 5621 | 0.921 | 5149.5 |

Reading: on the same 4448 pixels A is worse than the C/O MAP by ~108 in chi2 (A fits a wider window
jointly, split 45 vs 100, and a sampled max-L point vs an optimizer's MAP). The full-window 0.921 is
low because the red pixels sit at chi2/pixel 0.55, not because the fit improved.
"""

import argparse
import importlib.util
import pickle

import numpy as np
from nautilus import Sampler

from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.fitting.map_fits import chi2_parts, get_sps
from hubersed.paths import PATHS

TID = 39627770174637084
RES = PATHS["RESULTS"] / "2026-09-30_nautilus_c3k_broadline_split45"
MAP_CO = PATHS["RESULTS"] / "2026-09-26_nautilus_c3k_broadline/map_co"
MILES = (3601.8, 7400.8)


def load_module(path, name):
    """Import a fit.py from another experiment under its own module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


parser = argparse.ArgumentParser()
parser.add_argument("--b-dir", default=None, help="directory holding a copy of B's checkpoint")
args = parser.parse_args()

out = []
for window in ("miles", "full"):
    r = pickle.load(open(MAP_CO / f"{TID}_map_co_{window}.pkl", "rb"))
    npix = int(np.asarray(r["data"]["good"], bool).sum())
    for k in ("tied", "free"):
        b = r[k]
        out.append(
            f"map_co {window:5s} {k:5s}: npix {npix}, ndim {len(b['labels'])}, "
            f"chi2 {float(b['chi2']):.1f}, chi2_nu {float(b['chi2_red']):.3f}"
        )

fit = load_module(
    PATHS["ROOT"] / "experiments/2026-09-30_nautilus_c3k_broadline_split45/fit.py", "fit0930"
)
cue = get_sps(zero_library_resolution=False)["cue"]
z, flux, unc, good, _ = fit.load_data(TID, 9000.0, True)
_, _, _, good_masked, _ = fit.load_data(TID, 9000.0, False)
rest = WAVE_OBS / (1 + z)
miles = (rest > MILES[0]) & (rest < MILES[1])
subsets = {
    "full < 9000": good,
    "MILES, sky-masked pixels": good_masked & miles,
    "unmasked sky pixels": good & ~good_masked,
    "red 7400.8-9000": good & (rest >= MILES[1]),
}
for tag, mode in (("A", "shared"), ("B", "free")):
    src = RES if (tag == "A" or args.b_dir is None) else PATHS["ROOT"] / args.b_dir
    m = fit.make_model(z, mode, 45.0)
    stem = f"{TID}_broad_{mode}_split45_unmask_rest9000_seed0"
    s = Sampler(
        lambda x: x,
        lambda x: 0.0,
        n_dim=m.ndim,
        n_live=1000,
        filepath=str(src / f"{stem}.h5"),
        resume=True,
    )
    cube, _, log_l = s.posterior()
    sp, _ = chi2_parts(
        m,
        m.prior_transform(cube[np.argmax(log_l)]),
        fit.make_obs(flux, unc, good),
        cue,
        np.zeros_like(good),
    )
    chi2 = np.where(good, ((flux - sp) / np.where(good, unc, 1.0)) ** 2, 0.0)
    out.append(f"{tag} ({mode}, N_eff {s.n_eff:.0f}, ndim {m.ndim}):")
    for name, sel in subsets.items():
        c = chi2[sel].sum()
        out.append(
            f"  {name:26s} npix {sel.sum():5d}  chi2 {c:8.1f}  chi2/npix {c / sel.sum():.3f}  "(
                f"chi2_nu {c / (sel.sum() - m.ndim):.3f}" if sel.sum() > 100 else ""
            )
        )

text = "\n".join(out) + "\n"
print(text)
(RES / "chi2_ladder.txt").write_text(text)
