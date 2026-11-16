"""Compare the max-L chi2 of 37084 across the 2026-09-24 to 2026-09-30 fits on fixed pixel sets.

QUESTION: How does the fit quality of 37084 compare across the nautilus and MAP steps of 2026-09-24
to 2026-09-30 when measured on the same pixels?
METHOD: Recompute the max-L chi2 of runs A (shared) and B (free) on their full-window pixels, the
4448 MILES-window pixels of 09-25/09-26, the 24 unmasked sky pixels and red 7400.8-9000 A pixels.
INPUTS: results/2026-09-30_nautilus_c3k_broadline_split45/39627770174637084_broad_{shared,free}_*.h5
(or a copy of B via --b-dir), results/2026-09-26_nautilus_c3k_broadline/map_co/*_map_co_{miles,full}.pkl
SEED: none needed.
COMMAND: uv run python experiments/2026-09-30_nautilus_c3k_broadline_split45/chi2_ladder.py [--b-dir DIR]
RESULT: results/2026-09-30_nautilus_c3k_broadline_split45/chi2_ladder.txt. On the 4448 pixels A
is ~108 worse in chi2 than the C/O MAP. Red pixels at chi2/pixel 0.55 pull A's chi2_nu to 0.921.
Assumptions: from 09-25 on, degraded data keep the raw DESI ivar, so chi2_nu is not in noise units.
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
