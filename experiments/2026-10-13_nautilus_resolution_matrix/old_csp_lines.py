"""Share of each Cue line made by the old CSP at the rmatrix nautilus max-L point.

QUESTION: prospect runs Cue on both the young (< dust_tesc) and the old CSP, each with its own
fitted ionizing spectrum and Q but the same gas parameters (nebssp_basis.get_spectrum). Does the
old CSP's harder spectrum make a large part of the auroral lines, and so of the "too hot" gas?

The run wraps nebssp_basis.fit_4loglinear_ionparam and add_dust to record the ionizing parameters
and the line luminosities of each CSP, before dust and after dust (old lines get only dust2).

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/old_csp_lines.py``

OUTPUT: old_csp_lines.txt in results/2026-10-13_nautilus_resolution_matrix/.
"""

import numpy as np
from auroral_checks import LINES
from fit import stem
from map_unmask import OUT, build
from model_setup import make_model, make_params
from prospect.sources import nebssp_basis

CAP = {"ion": []}
_fit_ion = nebssp_basis.fit_4loglinear_ionparam
_add_dust = nebssp_basis.add_dust


def fit_ion(wave, spec):
    """Record the ionizing parameters of each CSP."""
    out = _fit_ion(wave, spec)
    CAP["ion"].append(dict(out))
    return out


def add_dust(wave, csps, ewave, lines, **kw):
    """Record the young and old line luminosities before and after dust."""
    zero = np.zeros_like(lines[0])
    CAP["young"], CAP["old"] = np.array(lines[0]), np.array(lines[1])
    CAP["young_att"] = _add_dust(wave, [c.copy() for c in csps], ewave, [lines[0], zero], **kw)[1]
    CAP["old_att"] = _add_dust(wave, [c.copy() for c in csps], ewave, [zero, lines[1]], **kw)[1]
    return _add_dust(wave, csps, ewave, lines, **kw)


nebssp_basis.fit_4loglinear_ionparam = fit_ion
nebssp_basis.add_dust = add_dust

_, obs, cue, _ = build(True)
d = np.load(OUT / f"{stem('rmatrix', 'mask')}_pred.npz")
model = make_model(make_params(float(d["z"])), "rmatrix")
assert model.theta_labels() == [str(x) for x in d["labels"]]
model.predict(d["best"], observations=obs, sps=cue)
names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
assert names.size == CAP["young"].size

young, old = CAP["ion"][-2], CAP["ion"][-1]
out = ["ionizing parameters fitted to each CSP (young, old)"]
for k in young:
    out.append(f"{k:>20} {float(np.ravel(young[k])[0]):10.3f} {float(np.ravel(old[k])[0]):10.3f}")
out += ["", f"{'line':>16} {'old/total intrinsic':>20} {'old/total attenuated':>21}"]
for key, cue_names, _ in LINES:
    sel = np.isin(names, cue_names)
    y, o = CAP["young"][sel].sum(), CAP["old"][sel].sum()
    ya, oa = CAP["young_att"][sel].sum(), CAP["old_att"][sel].sum()
    out.append(f"{key:>16} {o / (y + o):20.3f} {oa / (ya + oa):21.3f}")
text = "\n".join(out)
print(text)
(OUT / "old_csp_lines.txt").write_text(text + "\n")
