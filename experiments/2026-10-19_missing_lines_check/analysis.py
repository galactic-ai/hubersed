"""Find emission lines in the fitted range that are seen in the data but neither in Cue nor masked.

QUESTION: Which emission lines in the fitted range are neither in Cue's line list nor masked, and
seen in the data?
HYPOTHESIS: None registered before this analysis.
INPUTS: pmc_final.state of the free_dust runs of 39627770174637084, 39633278973513279 and
39633001071509599 in results/2026-10-15_broadline_free_amplitude/.
SEED: none.
COMMAND: from the repository root with the C3K_ER build, once per target,
FIT_TID=<TID> PYTHONPATH=<er_path>:experiments/2026-10-15_broadline_free_amplitude:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=<fsps> uv run --no-sync python
experiments/2026-10-19_missing_lines_check/analysis.py
RESULT: see <TID>.txt.
Assumptions: a 4 A match window to Cue and MISSING_LINES lines. O I 7774 and 8446 come from memory.
FIGURES: none. <TID>.txt in results/2026-10-19_missing_lines_check/.
"""

# %%

import argparse
from importlib.resources import files

import numpy as np

from hubersed.paths import PATHS
from hubersed.sps.utils import _airtovac

OUT = PATHS["RESULTS"] / "2026-10-19_missing_lines_check"
R_H = 109677.583  # hydrogen Rydberg constant with reduced mass, cm^-1
HALF = 3.0  # rest A, scoring window
NEAR = 4.0  # rest A, distance to a known line
IONS = [("N", 1), ("N", 2), ("O", 1), ("O", 2), ("O", 3), ("Ne", 3), ("S", 2), ("S", 3)]
IONS += [("Ar", 3), ("Ar", 4), ("Cl", 2), ("Cl", 3), ("Fe", 3)]
TE, NE = 1e4, 100.0  # K and cm^-3 for the forbidden-line emissivities
MIN_REL = 0.01  # keep forbidden lines with at least this emissivity relative to the ion's strongest


def hydrogen(n1, name):
    """Return the series of n1 for upper levels 11 to 30, vacuum A."""
    return {f"{name}{n}": 1e8 / (R_H * (1 / n1**2 - 1 / n**2)) for n in range(11, 31)}


def candidates(lo, hi):
    """Return candidate lines between lo and hi rest vacuum A."""
    import pyneb as pn

    out = hydrogen(2, "H") | hydrogen(3, "Pa")
    for label in pn.RecAtom("He", 1).labels:
        w = float(label)
        out[f"He I {label}"] = _airtovac(w)
    for elem, spec in IONS:
        try:
            atom = pn.Atom(elem, spec)
        except Exception as err:  # pyneb lacks atomic data for some ions
            print(f"skip {elem} {spec}: {err}")
            continue
        emis = np.asarray(atom.getEmissivity(TE, NE))
        k = emis.shape[0]  # pyneb can compute fewer levels than wave_Ang lists
        waves = np.ravel(atom.wave_Ang[:k, :k])
        emis = np.ravel(emis)
        optical = np.isfinite(waves) & np.isfinite(emis) & (waves > lo) & (waves < hi)
        if not optical.any():
            continue
        strong = optical & (emis >= MIN_REL * emis[optical].max())
        for w in waves[strong]:
            out[f"[{elem} {spec}] {w:.1f}"] = _airtovac(float(w))
    out["O I 7774"] = _airtovac(7773.4)
    out["O I 8446"] = _airtovac(8446.4)
    return {k: v for k, v in out.items() if lo < v < hi}


# %%
def main():
    """Score the candidates and run the blind search for the target in FIT_TID."""
    import fit
    from compare_poco import load_sampler
    from model_setup import MISSING_LINES

    argparse.ArgumentParser(description=__doc__).parse_args()
    s = load_sampler(fit.OUT / f"poco_{fit.TID}_free_dust_neff1024" / "pmc_final.state")
    usamp, w, logl, _ = s.posterior()
    model, obs, cue = fit.process_state("free_dust")
    theta = np.array([model.prior_transform(x) for x in usamp])
    top = np.argsort(logl)[-300:]
    best = theta[top[np.argmax([logl[i] + model.prior_product(theta[i]) for i in top])]]
    spec = np.asarray(model.predict(best, observations=obs, sps=cue)[0][0])
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = np.asarray(o.wavelength) / (1 + z)
    m = o.mask & np.isfinite(o.uncertainty) & (o.uncertainty > 0)
    chi = np.where(m, (o.flux - spec) / o.uncertainty, 0.0)

    info = np.genfromtxt(
        files("cuejax.data").joinpath("cue_emlines_info.dat"),
        dtype=[("w", "f8"), ("n", "<U24")],
        delimiter=",",
    )
    known = np.concatenate([info["w"], list(MISSING_LINES.values())])
    lo, hi = rest[m].min(), rest[m].max()
    cand = candidates(lo, hi)
    cand = {k: v for k, v in cand.items() if np.min(np.abs(known - v)) >= NEAR}

    lines = [f"{fit.TID} z {z:.5f}, fitted rest {lo:.0f}-{hi:.0f} A, {len(cand)} candidates", ""]
    lines.append(f"{'candidate':>22} {'vac A':>8} {'n':>3} {'S':>6} {'max chi':>7}")
    rows = []
    for k, v in cand.items():
        sel = m & (np.abs(rest - v) < HALF)
        n = int(sel.sum())
        if n == 0:
            rows.append((-np.inf, f"{k:>22} {v:8.2f} {n:3d}  (no fitted pixels)"))
            continue
        score = float(chi[sel].sum() / np.sqrt(n))
        rows.append((score, f"{k:>22} {v:8.2f} {n:3d} {score:6.2f} {chi[sel].max():7.2f}"))
    rows.sort(key=lambda r: -r[0])
    lines += [r[1] for r in rows]

    smooth = np.convolve(chi, np.ones(3), "same") / np.sqrt(3)
    near = np.concatenate([known, list(cand.values())])
    hit = m & (smooth > 4) & (np.min(np.abs(rest[:, None] - near[None]), axis=1) >= NEAR)
    lines += ["", "blind search, runs of smoothed chi > 4 away from known lines and candidates:"]
    idx = np.flatnonzero(hit)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1) if idx.size else []:
        j = run[np.argmax(smooth[run])]
        lines.append(f"  rest {rest[j]:8.2f}  obs {o.wavelength[j]:8.2f}  peak {smooth[j]:5.1f}")
    text = "\n".join(lines)
    print(text)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{fit.TID}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
