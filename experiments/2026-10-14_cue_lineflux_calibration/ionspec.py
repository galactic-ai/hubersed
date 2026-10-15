"""Ionizing spectrum of an assumed young population for the line-flux Cue fits, against the pipeline's.

QUESTION: the line-flux fits have no continuum, so Cue's ionizing spectrum comes from an assumed
young population: constant SFR for 10 Myr (the pipeline's young/old split, dust_tesc), at fixed
stellar Z. How far are its 7 Cue ionizing parameters from those the full-spectrum pipeline used at
the rmatrix nautilus max-L of 39627770174637084 (old_csp_lines.txt), and how much does that move
the line ratios at the same gas parameters?

The population is built with the pipeline's FSPS object and settings (C3K_ER build, the model's
imf_type and other FSPS parameters), with sfh 1, const 1, tage 0.01 Gyr. The young CSP of FSPS is
fitted with cuejax fit_4loglinear_ionparam, as prospect does. Line ratios use Cue at the max-L gas
parameters for three stellar Z: -0.41 (the target's fitted logzsol), 0.0 and -0.75. SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-14_cue_lineflux_calibration/ionspec.py``

OUTPUT: ionspec.txt in results/2026-10-14_cue_lineflux_calibration/.
"""

import jax.numpy as jnp
import numpy as np
from cuejax.emulator import fast_line_prediction
from cuejax.utils import fit_4loglinear_ionparam
from fit import stem
from map_unmask import OUT as RM_OUT
from model_setup import get_cue, make_model, make_params

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
TAGE_GYR = 0.01
ZSTARS = (-0.41, 0.0, -0.75)
ION_KEYS = (
    "ionspec_index1",
    "ionspec_index2",
    "ionspec_index3",
    "ionspec_index4",
    "ionspec_logLratio1",
    "ionspec_logLratio2",
    "ionspec_logLratio3",
)
GAS_KEYS = ("gas_logu", "gas_lognH", "gas_logz", "gas_logno", "gas_logco")
# pipeline young-CSP ionizing parameters at the max-L point, from old_csp_lines.txt
PIPELINE = (6.945, 7.520, 5.647, 1.707, 2.575, 0.430, 0.487)
# ratio name, Cue line names in the numerator, in the denominator
RATIOS = [
    ("[OIII]5007/Hb", ("O 3 5006.84A",), ("H 1 4861.32A",)),
    ("[OII]3727/Hb", ("O 2 3726.03A", "O 2 3728.81A"), ("H 1 4861.32A",)),
    ("[NII]6584/Ha", ("N 2 6583.45A",), ("H 1 6562.80A",)),
    ("[SII]6716,31/Ha", ("S 2 6716.44A", "S 2 6730.82A"), ("H 1 6562.80A",)),
    ("[OI]6300/Ha", ("O 1 6300.30A",), ("H 1 6562.80A",)),
    ("[NeIII]3869/Hb", ("Ne 3 3868.76A",), ("H 1 4861.32A",)),
    ("[OII]7325/3727", ("O 2 7323.00A", "O 2 7332.00A"), ("O 2 3726.03A", "O 2 3728.81A")),
    ("[SIII]6312/Hb", ("S 3 6312.06A",), ("H 1 4861.32A",)),
    ("[OIII]4363/5007", ("O 3 4363.21A",), ("O 3 5006.84A",)),
]


def young_ionspec(ssp, logzsol):
    """Return the 7 Cue ionizing parameters of the young CSP of a 10 Myr constant-SFR population."""
    ssp.params["logzsol"] = logzsol
    wave, _ = ssp.get_spectrum(tage=TAGE_GYR, peraa=False)
    young = np.array(ssp._csp_young_old[0])
    out = fit_4loglinear_ionparam(wave, young)
    return np.array([float(np.ravel(out[k])[0]) for k in ION_KEYS])


def ratios(cue, ion, gas):
    """Return the RATIOS of Cue at ionizing parameters ``ion`` and gas parameters ``gas``."""
    lum = np.asarray(fast_line_prediction(jnp.array([[*ion, *gas]]), cue.emul)[0])[0]
    names = [str(x).strip() for x in np.atleast_1d(cue.emul.line_names)]
    idx = {n: i for i, n in enumerate(names)}
    return [sum(lum[idx[n]] for n in a) / sum(lum[idx[n]] for n in b) for _, a, b in RATIOS]


cue = get_cue()
d = np.load(RM_OUT / f"{stem('rmatrix', 'mask')}_pred.npz")
labels = [str(x) for x in d["labels"]]
best = dict(zip(labels, d["best"], strict=True))
model = make_model(make_params(float(d["z"])), "rmatrix")
ssp = cue.ssp
for k in ("imf_type", "dust_tesc", "zred", "add_neb_emission", "add_neb_continuum"):
    if k in model.params:
        ssp.params[k] = model.params[k][0] if np.ndim(model.params[k]) else model.params[k]
ssp.params["add_neb_emission"] = False
ssp.params["sfh"] = 1
ssp.params["const"] = 1.0
ssp.params["fburst"] = 0.0
ssp.params["dust1"] = 0.0
ssp.params["dust2"] = 0.0
gas = [float(best[k]) for k in GAS_KEYS]

out = [
    f"FSPS imf_type {ssp.params['imf_type']}, dust_tesc {ssp.params['dust_tesc']}, sfh 1 const 1, "
    f"tage {TAGE_GYR} Gyr; gas at the max-L: "
    + ", ".join(f"{k} {v:.3f}" for k, v in zip(GAS_KEYS, gas, strict=True)),
    "",
    f"{'parameter':>20} {'pipeline':>9} " + " ".join(f"{'Z* ' + str(z):>9}" for z in ZSTARS),
]
ion = {z: young_ionspec(ssp, z) for z in ZSTARS}
for i, k in enumerate(ION_KEYS):
    out.append(f"{k:>20} {PIPELINE[i]:9.3f} " + " ".join(f"{ion[z][i]:9.3f}" for z in ZSTARS))
ref = ratios(cue, PIPELINE, gas)
rows = {z: ratios(cue, ion[z], gas) for z in ZSTARS}
out += [
    "",
    "Cue line ratios at the max-L gas; columns after the pipeline value are log10(new/pipeline)",
]
out.append(f"{'ratio':>20} {'pipeline':>9} " + " ".join(f"{'Z* ' + str(z):>9}" for z in ZSTARS))
for j, (name, _, _) in enumerate(RATIOS):
    out.append(
        f"{name:>20} {ref[j]:9.4f} "
        + " ".join(f"{np.log10(rows[z][j] / ref[j]):+9.3f}" for z in ZSTARS)
    )
text = "\n".join(out)
print(text)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ionspec.txt").write_text(text + "\n")
