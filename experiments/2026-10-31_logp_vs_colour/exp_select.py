"""List the blue outliers for the single-exposure check and the iron spectra files they need (#17).

Targets are the 94 (sample94.npz) with dgr < -0.3 (FIBERTOTFLUX, qa_outliers.txt column dgr_tot)
and COADD_NUMEXP >= 2.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_logp_vs_colour/exp_select.py``.
OUTPUT: exp_targets.txt and exp_urls.txt next to this script.
"""

from pathlib import Path

from astropy.io import fits

from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
QA = PATHS["RESULTS"] / "2026-10-31_spectrum_qa_cont_outliers" / "qa_outliers.txt"
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/sv3/bright"
CUT = -0.3

lines = QA.read_text().splitlines()
col = {k: i for i, k in enumerate(lines[0].split())}
h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M = h["METADATA"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}
rows = []
for line in lines[1:]:
    r = line.split()
    dgr, nexp = float(r[col["dgr_tot"]]), int(float(r[col["numexp"]]))
    if dgr < CUT and nexp >= 2:
        t = int(r[col["TARGETID"]])
        rows.append((t, int(M["HEALPIX"][vrow[t]]), dgr, nexp))
(HERE / "exp_targets.txt").write_text(
    "".join(f"{t} {hp} {d:.3f} {n}\n" for t, hp, d, n in sorted(rows, key=lambda x: x[2]))
)
hps = sorted({hp for _, hp, _, _ in rows})
(HERE / "exp_urls.txt").write_text(
    "".join(f"{BASE}/{hp // 100}/{hp}/spectra-sv3-bright-{hp}.fits.gz\n" for hp in hps)
)
print(f"{len(rows)} targets in {len(hps)} healpix files")
