"""Line-flux mocks for the N/O and C/O prior test: Cue at known gas Z, with real galaxies' noise.

Truth: gas-phase 12+log(O/H) in OH_GRID (gas_logz = value - 8.71, Cue's [O/H] before depletion),
gas_logu in U_GRID, gas_lognH 2.0, av 0.5, the young-population ionizing spectrum at stellar Z
-0.41 (ionspec.npz), and N/O and C/O on the Nicholls+17 relations (arXiv:1612.03546,
abundances.tex:301-304, total abundances, so on Cue's total scale):
gas_logno = log10(10^-0.852 + 10^gas_logz) and gas_logco = log10(10^-0.43 + 10^(gas_logz + 0.02)).
Control mocks put N/O 0.3 dex above or below the relation at the CONTROL_OH values and U -2.5.

Noise: for each truth, the N_DONOR galaxies of lineflux_table.fits whose observed log [NII]/Halpha
is closest to the mock's give their fractional errors (E/F per line). A line missing in the donor
is missing in the mock. The mock flux is the Cue flux plus Gaussian noise of that fractional error,
plus Gaussian noise of Cue's own emulation error (line_frac_error), so the mocks are no closer to
Cue than real data. E_<line> holds the measurement error only; fit.py adds the emulation error.
Seed: (SEED, mock index).

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/mocks.py`` from
the repository root.

OUTPUT: mock_table.fits in results/2026-10-14_cue_lineflux_calibration/mocks/, in the
lineflux_table.fits format (TARGETID, F_<line>, E_<line>) plus the truth and donor columns.
"""

import numpy as np
from astropy.table import Table
from fit import LINES, LineModel, nicholls_co, nicholls_no

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
SEED = 0
CUE_OH_ZERO = 8.71
OH_GRID = (8.0, 8.2, 8.4, 8.6, 8.8, 9.0)
U_GRID = (-3.0, -2.5)
CONTROL_OH = (8.4, 8.6, 8.8)
CONTROL_OFFSET = (-0.3, 0.3)
LOGNH, AV, LOG_FHB, ZSTAR = 2.0, 0.5, 3.0, -0.41
N_DONOR = 3
ID0 = 900_000_000


grid = np.load(OUT / "ionspec.npz")
model = LineModel(grid["ionspec"][int(np.flatnonzero(np.isclose(grid["zstar"], ZSTAR))[0])], 0.4)
data = Table.read(OUT / "lineflux_table.fits")
with np.errstate(divide="ignore", invalid="ignore"):
    n2_data = np.log10(np.asarray(data["F_NII6584"], float) / np.asarray(data["F_HA"], float))
donor_ok = np.isfinite(n2_data)

truths = [(oh, u, 0.0) for oh in OH_GRID for u in U_GRID]
truths += [(oh, -2.5, off) for oh in CONTROL_OH for off in CONTROL_OFFSET]
rows = []
for oh, u, off in truths:
    z = oh - CUE_OH_ZERO
    x = np.array([[u, LOGNH, z, nicholls_no(z) + off, nicholls_co(z), AV, LOG_FHB]])
    flux, emu_err = model.fluxes(x)
    true = {k: float(flux[k][0]) for k in LINES}
    n2 = np.log10(true["NII6584"] / true["HA"])
    donors = np.flatnonzero(donor_ok)[np.argsort(np.abs(n2_data[donor_ok] - n2))[:N_DONOR]]
    for di in donors:
        i = len(rows)
        rng = np.random.default_rng([SEED, i])
        row = {"TARGETID": ID0 + i}
        for k in LINES:
            fd, ed = float(data[f"F_{k}"][di]), float(data[f"E_{k}"][di])
            if not (np.isfinite(fd) and np.isfinite(ed) and fd > 0 and ed > 0):
                row[f"F_{k}"], row[f"E_{k}"] = np.nan, np.nan
                continue
            sig = ed / fd * true[k]
            row[f"F_{k}"] = true[k] + rng.normal(0, sig) + rng.normal(0, float(emu_err[k][0]))
            row[f"E_{k}"] = sig
        row.update(
            TRUE_OH=oh,
            TRUE_LOGZ=z,
            TRUE_LOGU=u,
            TRUE_LOGNO=float(x[0, 3]),
            TRUE_LOGCO=float(x[0, 4]),
            NO_OFFSET=off,
            MOCK_N2=n2,
            DONOR_TARGETID=int(data["TARGETID"][di]),
            DONOR_N2=float(n2_data[di]),
        )
        rows.append(row)
t = Table(rows=rows)
(OUT / "mocks").mkdir(parents=True, exist_ok=True)
t.write(OUT / "mocks" / "mock_table.fits", overwrite=True)
print(f"wrote {len(t)} mocks")
cols = [
    "TARGETID",
    "TRUE_OH",
    "TRUE_LOGU",
    "NO_OFFSET",
    "TRUE_LOGNO",
    "TRUE_LOGCO",
    "MOCK_N2",
    "DONOR_N2",
]
for r in t:
    snr = {k: r[f"F_{k}"] / r[f"E_{k}"] for k in ("HB", "NII6584", "OII3727", "OII7325")}
    print(
        " ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols)
        + "  S/N "
        + " ".join(f"{k} {v:.0f}" for k, v in snr.items())
    )
