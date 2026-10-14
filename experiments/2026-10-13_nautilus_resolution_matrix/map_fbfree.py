"""MAP fit of 39627770174637084 with a free broad fraction for the non-Balmer lines.

QUESTION: map.py ties the broad fraction of every non-Balmer line to the Balmer one (forbidden
broad "shared"). With that fraction free, where does it go, how much does chi2 drop, and do the
gas parameters move?
HYPOTHESIS: the literature (knowledge/lit_broad_emission_components_2026-10-13.md, section 6) finds
broad fractions that differ by line, so a free fraction should leave the Balmer value and lower
chi2 in the forbidden-line windows. Whether the gas parameters move is open.

Same setup as the rmatrix mask run of map.py: C3K_ER, Cue broadline with sigma split 45, the
target's row-weighted R, corrected ivar, missing-line mask, SFH hyperparameters fixed at the 10-09
max-L. The only change is ``eline_fbroad_forb``, free with the TopHat(0, 0.9) prior of
``add_broad_params(forbidden_broad="free")``. One Powell start at the rmatrix mask MAP point with
``eline_fbroad_forb`` equal to its ``eline_fbroad``. The shared model is the free model with
``eline_fbroad_forb = eline_fbroad``, so the start is the shared optimum.

INPUTS: map_rmatrix_mask.npz (start), plus the inputs of map.py.
SEED: none (one start, no jitter).
COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/map_fbfree.py [--maxfev N]``, then the same with
``report`` and with ``scan``.
OUTPUT: map_rmatrix_mask_fbfree.npz, map_fbfree.txt and map_fbfree_scan.txt in
results/2026-10-13_nautilus_resolution_matrix/.
"""

import argparse
import time

import numpy as np
from map import HYPERS, MASKS, OUT, START, describe
from model_setup import get_cue, load_data, make_model, make_obs, make_params
from prospect.models import priors

from hubersed.fitting.map_fits import map_fit

SHARED = OUT / "map_rmatrix_mask.npz"
FREE = OUT / "map_rmatrix_mask_fbfree.npz"


def build(free):
    """Return the rmatrix mask model, observation, Cue source and the shared MAP point.

    With ``free`` the non-Balmer broad fraction is a free parameter. The returned point is the
    shared MAP, with ``eline_fbroad_forb`` set to its ``eline_fbroad`` when free.
    """
    d = np.load(START, allow_pickle=True)
    maxl = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    z, flux, unc, good = load_data(MASKS["mask"])
    params = make_params(z)
    for k in HYPERS:
        params[k]["isfree"] = False
        params[k]["init"] = float(maxl[k])
    s = np.load(SHARED)
    shared = dict(zip([str(x) for x in s["labels"]], s["theta"], strict=True))
    if free:
        params["eline_fbroad_forb"] = dict(
            N=1,
            isfree=True,
            init=float(shared["eline_fbroad"]),
            prior=priors.TopHat(mini=0.0, maxi=0.9),
        )
        shared["eline_fbroad_forb"] = shared["eline_fbroad"]
    model = make_model(params, "rmatrix")
    theta = np.array([shared[k] for k in model.theta_labels()])
    return model, make_obs(flux, unc, good, "rmatrix"), get_cue(), theta


def run(maxfev):
    """Run the free MAP from the shared optimum and save it."""
    model, obs, cue, theta0 = build(free=True)
    o = obs[0]
    pred0 = model.predict(theta0, observations=obs, sps=cue)[0][0]
    chi2_0 = float(np.sum(((o.flux - pred0)[o.mask] / o.uncertainty[o.mask]) ** 2))
    print(f"chi2 at start {chi2_0:.1f} (shared MAP in map.txt: 6091.8)")
    t0 = time.time()
    best, info = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=theta0, tag="fbfree")
    np.savez(
        FREE,
        labels=np.array(model.theta_labels()),
        theta0=theta0,
        theta=best.x,
        fun=best.fun,
        nfev=best.nfev,
        success=best.success,
        maxfev=maxfev,
    )
    print(
        f"done in {time.time() - t0:.0f} s, -lnp {best.fun:.2f}, nfev {best.nfev}, "
        f"success {best.success}"
    )


def report():
    """Compare the shared and free MAPs and write map_fbfree.txt."""
    masked_good = load_data(MASKS["mask"])[3]
    s, f = np.load(SHARED), np.load(FREE)
    rows = {}
    for name, free, r in [("shared", False, s), ("free", True, f)]:
        model, obs, cue, _ = build(free)
        rows[name] = (describe(model, obs, cue, r["theta"], masked_good), r)
    out = [
        "MAP of 39627770174637084, rmatrix mask, forbidden broad shared vs free.",
        f"shared: -lnp {float(s['fun']):.2f}, nfev {int(s['nfev'])}",
        f"free:   -lnp {float(f['fun']):.2f}, nfev {int(f['nfev'])} of {int(f['maxfev'])}, "
        f"success {bool(f['success'])}",
        "",
        f"{'chi2':>17} {'shared':>10} {'free':>10} {'free-shared':>12}",
    ]
    for k in rows["shared"][0][0]:
        a, b = rows["shared"][0][0][k], rows["free"][0][0][k]
        out.append(f"{k:>17} {a:10.1f} {b:10.1f} {b - a:12.1f}")
    out += ["", f"{'Cue flux 1e-17':>17} {'shared':>10} {'free':>10} {'free/shared':>12}"]
    for k in rows["shared"][0][1]:
        a, b = rows["shared"][0][1][k], rows["free"][0][1][k]
        out.append(f"{k:>17} {a:10.1f} {b:10.1f} {b / a:12.4f}")
    sl = [str(x) for x in s["labels"]]
    fl = [str(x) for x in f["labels"]]
    out += ["", f"{'parameter':>17} {'shared':>10} {'free':>10} {'free-shared':>12}"]
    for i, k in enumerate(fl):
        b = f["theta"][i]
        a = s["theta"][sl.index(k)] if k in sl else s["theta"][sl.index("eline_fbroad")]
        note = "  (shared value is eline_fbroad)" if k not in sl else ""
        out.append(f"{k:>17} {a:10.4f} {b:10.4f} {b - a:12.4f}{note}")
    text = "\n".join(out) + "\n"
    print(text)
    (OUT / "map_fbfree.txt").write_text(text)


# rest-frame windows of the non-Balmer lines for the scan, in Angstrom
SCAN_WINDOWS = {
    "[OII]3726,9": (3718, 3740),
    "[OIII]4959": (4950, 4970),
    "[OIII]5007": (4995, 5022),
    "[OI]6300": (6292, 6310),
    "[NII]6548": (6540, 6556),
    "[NII]6584": (6576, 6592),
    "[SII]6716,31": (6708, 6740),
}


def scan():
    """Scan chi2 over ``eline_fbroad_forb`` with every other parameter at the free MAP.

    This checks whether the free MAP stopped at a real minimum, and which line windows set it.
    Holding the other parameters fixed makes the curve steeper than a profile likelihood.
    """
    model, obs, cue, _ = build(free=True)
    f = np.load(FREE)
    theta = f["theta"].copy()
    i = list(model.theta_labels()).index("eline_fbroad_forb")
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)
    grid = np.unique(np.r_[np.linspace(0.0, 0.6, 25), theta[i]])
    out = [
        "chi2 scan over eline_fbroad_forb, all other parameters at the free MAP "
        f"(eline_fbroad {theta[list(model.theta_labels()).index('eline_fbroad')]:.4f}).",
        f"{'fb_forb':>8} {'all':>9} " + " ".join(f"{k:>12}" for k in SCAN_WINDOWS),
    ]
    for v in grid:
        theta[i] = v
        sp = model.predict(theta, observations=obs, sps=cue)[0][0]
        r2 = ((o.flux - sp) / o.uncertainty) ** 2
        parts = [r2[o.mask & (rest >= a) & (rest < b)].sum() for a, b in SCAN_WINDOWS.values()]
        out.append(f"{v:8.3f} {r2[o.mask].sum():9.1f} " + " ".join(f"{p:12.1f}" for p in parts))
    text = "\n".join(out) + "\n"
    print(text)
    (OUT / "map_fbfree_scan.txt").write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", nargs="?", choices=("run", "report", "scan"), default="run")
    ap.add_argument("--maxfev", type=int, default=40000)
    a = ap.parse_args()
    if a.action == "report":
        report()
    elif a.action == "scan":
        scan()
    else:
        run(a.maxfev)
