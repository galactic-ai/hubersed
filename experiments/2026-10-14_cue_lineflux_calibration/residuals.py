"""Which lines Cue misses in the real galaxies, metal-rich against metal-poor, from existing fits.

QUESTION: the mocks (analysis_mocks.py) show the strong lines separate Z and N/O within Cue, so the
real metal-rich galaxies' low-Z, high-N/O, low-U solutions point to spectra Cue does not reproduce.
Which lines does it miss in them?

For each galaxy of subset_20.txt, from fits/<TARGETID>_strong_emfit_mu0.4_zs-0.41.npz (flat priors):
the median over the 200 posterior draws of model/data for every line, and chi = (data - median
model) / sqrt(err^2 + emulation error^2), with the emulation error from Cue's line_frac_error at the
median model flux. Groups: metal-rich if observed N2 = log [NII]6584/Halpha > N2_CUT, metal-poor
otherwise. Lines with S/N < 3 are left out. SEED: none.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/residuals.py``
from the repository root.

OUTPUT: residuals.txt in results/2026-10-14_cue_lineflux_calibration/.
"""

import numpy as np
from astropy.table import Table
from cuejax.emulator import Emulator
from fit import LINES

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
N2_CUT = -0.8
SNR_MIN = 3.0

emul = Emulator()
names = [str(x).strip() for x in np.atleast_1d(emul.line_names)]
frac = np.asarray(emul.line_frac_error, float)
# one emulation fraction per table line, the larger member for summed lines
frac_line = {k: max(frac[names.index(n)] for n in v) for k, v in LINES.items()}

t = Table.read(OUT / "lineflux_table.fits")
ids = np.loadtxt(OUT / "subset_20.txt", dtype=np.int64)
rows = []
for tid in ids:
    d = np.load(OUT / "fits" / f"{tid}_strong_emfit_mu0.4_zs-0.41.npz")
    lines = [str(x) for x in d["lines"]]
    used = {str(x) for x in d["used"]}
    r = t[t["TARGETID"] == tid][0]
    n2 = float(np.log10(r["F_NII6584"] / r["F_HA"]))
    lab = [str(x) for x in d["labels"]]
    lw = d["log_w"]
    i = int(np.argmax(lw))
    z = float(d["points"][i, lab.index("gas_logz")])
    res = {}
    for j, k in enumerate(lines):
        obs, err = float(d["obs"][j]), float(d["err"][j])
        if not (np.isfinite(obs) and err > 0 and obs / err >= SNR_MIN):
            continue
        mod = float(np.median(d["draw_flux"][:, j]))
        chi = (obs - mod) / np.sqrt(err**2 + (frac_line[k] * mod) ** 2)
        res[k] = (mod / obs, chi, k in used)
    rows.append((n2, int(tid), float(r["LOGMSTAR"]), z, res))
rows.sort()

order = list(LINES)
out = [
    "model/data (chi) per line, flat strong-line fits; * = held out; sorted by observed N2",
    f"{'TARGETID':>18} {'logM*':>5} {'N2':>6} {'Z_maxpost':>9}  "
    + " ".join(f"{k:>14}" for k in order),
]
for n2, tid, m, z, res in rows:
    cells = []
    for k in order:
        if k in res:
            md, chi, fit = res[k]
            cells.append(f"{md:6.2f}({chi:+5.1f}){'' if fit else '*'}".rjust(14))
        else:
            cells.append(f"{'-':>14}")
    out.append(f"{tid:>18} {m:5.2f} {n2:+6.2f} {z:+9.2f}  " + " ".join(cells))
out += ["", f"group medians of model/data (N galaxies), metal-rich = N2 > {N2_CUT}"]
for k in order:
    rich = [res[k][0] for n2, _, _, _, res in rows if n2 > N2_CUT and k in res]
    poor = [res[k][0] for n2, _, _, _, res in rows if n2 <= N2_CUT and k in res]
    if rich or poor:
        fmt = lambda v: f"{np.median(v):.2f} ({len(v)})" if v else "-"  # noqa: E731
        out.append(f"{k:>10}: metal-rich {fmt(rich):>11}, metal-poor {fmt(poor):>11}")
text = "\n".join(out)
print(text)
(OUT / "residuals.txt").write_text(text + "\n")
