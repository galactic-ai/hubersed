"""Compare gas-parameter biases of the mock fits with and without [SII]6731 fitted.

Masked fits are fits/, unmasked ones fits_unmask_sii6731/ (fit.py --unmask "[SII]6731"). Shared
model, best of the truth and maxl starts per seed. Writes compare_unmask.txt.

COMMAND: ``PYTHONPATH=experiments/2026-10-13_nautilus_resolution_matrix:experiments/2026-10-13_broadline_injection
uv run --no-sync python experiments/2026-10-13_broadline_injection/compare_unmask.py`` from the
repository root.
"""

from pathlib import Path

import numpy as np
from injection import OUT

GAS = ["gas_logz", "gas_logu", "gas_lognH", "gas_logno"]
STARTS = ("truth", "maxl")


def best(folder, sc, seed):
    """Return (theta - truth) of the best start for one seed, as a dict, or None."""
    fits = [
        np.load(f) for st in STARTS if (f := Path(folder) / f"{sc}_{seed}_shared_{st}.npz").exists()
    ]
    if not fits:
        return None
    d = min(fits, key=lambda x: float(x["fun"]))
    lab = [str(x) for x in d["labels"]]
    return {k: float(d["theta"][lab.index(k)] - d["theta_truth"][lab.index(k)]) for k in GAS}


lines = ["bias (fit - truth, dex), shared model; masked -> [SII]6731 fitted", ""]
lines.append(f"{'':8s}" + "".join(f"{k:>26s}" for k in GAS))
for sc in ("S0", "S1", "S3"):
    for label, seeds in (("noiseless", [-1]), ("seeds 0-4", [0, 1, 2, 3, 4])):
        m = [b for s in seeds if (b := best(OUT / "fits", sc, s))]
        u = [b for s in seeds if (b := best(OUT / "fits_unmask_sii6731", sc, s))]
        if not u:
            continue
        row = f"{sc} {label:10s}"
        for k in GAS:
            mm = np.median([b[k] for b in m])
            uu = np.median([b[k] for b in u])
            sm = np.std([b[k] for b in m]) if len(m) > 1 else 0.0
            su = np.std([b[k] for b in u]) if len(u) > 1 else 0.0
            row += f"   {mm:+.3f}({sm:.3f})->{uu:+.3f}({su:.3f})"
        lines.append(row + f"   n {len(m)}->{len(u)}")
text = "\n".join(lines)
print(text)
(OUT / "compare_unmask.txt").write_text(text + "\n")
