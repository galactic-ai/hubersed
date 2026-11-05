"""How much the attenuation curve moves each line diagnostic, at the Balmer decrement of the target.

QUESTION: the calibration test fits Cue to line fluxes with an assumed attenuation curve. Which
diagnostics does the curve choice move, by how much, and how large is that in temperature?

Every curve is scaled so A(Hbeta) - A(Halpha) matches the FastSpecFit Halpha/Hbeta of
39627770174637084 for an intrinsic 2.86. Curves: CCM89 R_V 3.1 (PyNeb), Kriek & Conroy 2013 (the
pipeline's dust_type 4, prospect fake_fsps.attenuate) at several dust_index with no birth cloud,
and the pipeline's line curve at the target's rmatrix nautilus medians (dust_index 0.259,
dust_ratio 0.390, dust1_index -1). Temperature slopes d ln R / d ln T are PyNeb emissivity ratios
at 1e4 K and n_e 250 cm^-3. SEED: none.

COMMAND: ``uv run python experiments/2026-10-14_dust_curve_leverage/analysis.py`` from the repository root.

RESULT: see results/2026-10-14_dust_curve_leverage.txt.
"""

import numpy as np
import pyneb as pn
from astropy.io import fits
from prospect.sources.fake_fsps import attenuate

from hubersed.paths import PATHS

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-10-14_dust_curve_leverage.txt"
HA_HB_INTRINSIC = 2.86
WAVE = {
    "[OII]3727": 3728.0,
    "[OIII]4363": 4364.4,
    "Hbeta": 4862.7,
    "[OIII]5007": 5008.2,
    "[SIII]6312": 6313.8,
    "Halpha": 6564.6,
    "[NII]6584": 6585.3,
    "[SII]6716": 6718.3,
    "[OII]7325": 7325.0,
    "[SIII]9069": 9071.1,
}
# ratio name, numerator line, denominator line, what it constrains
PAIRS = [
    ("[OII]7325/3727", "[OII]7325", "[OII]3727", "T(O+), n_e"),
    ("[SIII]6312/9069", "[SIII]6312", "[SIII]9069", "T(S++)"),
    ("[OIII]4363/5007", "[OIII]4363", "[OIII]5007", "T(O++)"),
    ("[OII]3727/[OIII]5007", "[OII]3727", "[OIII]5007", "U"),
    ("[OII]3727/Hbeta", "[OII]3727", "Hbeta", "Z (R23 part)"),
    ("[OIII]5007/Hbeta", "[OIII]5007", "Hbeta", "Z, U"),
    ("[NII]6584/Halpha", "[NII]6584", "Halpha", "Z, N/O"),
    ("[SII]6716/Halpha", "[SII]6716", "Halpha", "Z, U"),
]
# PyNeb atom and (upper, lower) level pairs summed in the numerator and denominator
T_SLOPE = {
    "[OII]7325/3727": ("O", 2, ((4, 2), (4, 3), (5, 2), (5, 3)), ((2, 1), (3, 1))),
    "[SIII]6312/9069": ("S", 3, ((5, 4),), ((4, 2),)),
    "[OIII]4363/5007": ("O", 3, ((5, 4),), ((4, 3),)),
}


def kc13(lam, dust_index, dust_ratio=0.0):
    """Return optical depth per unit dust2 of the pipeline line curve (KC13 plus birth cloud)."""
    one = np.ones_like(lam)
    diff = -np.log(attenuate(one, lam, dust_type=4, dust_index=dust_index, dust2=1.0)[0])
    return diff + dust_ratio * (lam / 5500.0) ** -1.0


def ccm89(lam):
    """Return A(lambda)/E(B-V) of CCM89 with R_V 3.1."""
    rc = pn.RedCorr(law="CCM89", R_V=3.1)
    return np.asarray(rc.X(lam), float)


def t_slope(elem, ion, num, den):
    """Return d ln R / d ln T of a summed emissivity ratio at 1e4 K and n_e 250."""
    atom = pn.Atom(elem, ion)

    def ratio(t):
        e = lambda pairs: sum(atom.getEmissivity(t, 250.0, lev_i=u, lev_j=lo) for u, lo in pairs)  # noqa: E731
        return e(num) / e(den)

    return (np.log(ratio(1.01e4)) - np.log(ratio(0.99e4))) / (np.log(1.01) - np.log(0.99))


f = fits.getdata(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", "FASTSPEC")
i = np.flatnonzero(f["TARGETID"] == TID)[0]
dec = 2.5 * np.log10(f["HALPHA_FLUX"][i] / f["HBETA_FLUX"][i] / HA_HB_INTRINSIC)
lam = np.array(list(WAVE.values()))
raw = {"CCM89": ccm89(lam)}
for di in (-1.0, -0.5, 0.0, 0.3):
    raw[f"KC13 dust_index {di:+.1f}"] = kc13(lam, di)
raw["pipeline line curve (target)"] = kc13(lam, 0.259, 0.390)
curves = {}
for name, a in raw.items():
    a = dict(zip(WAVE, a, strict=True))
    s = dec / (a["Hbeta"] - a["Halpha"])
    curves[name] = {k: v * s for k, v in a.items()}

out = [
    f"{TID}: A(Hbeta) - A(Halpha) = {dec:.3f} mag from FastSpecFit, every curve scaled to it",
    "",
]
out.append("A(Halpha) in mag: " + ", ".join(f"{n} {c['Halpha']:.2f}" for n, c in curves.items()))
out += ["", "change of each observed ratio from dust, A(den) - A(num) in mag, by curve"]
out.append(f"{'ratio':>21} {'constrains':>13} " + " ".join(f"{n[:16]:>16}" for n in curves))
for name, a, b, what in PAIRS:
    row = [c[b] - c[a] for c in curves.values()]
    out.append(f"{name:>21} {what:>13} " + " ".join(f"{x:16.3f}" for x in row))
out += ["", "spread over curves (max - min) and what it means for temperature"]
for name, a, b, _ in PAIRS:
    row = np.array([c[b] - c[a] for c in curves.values()])
    spread = row.max() - row.min()
    frac = 10 ** (0.4 * spread) - 1
    line = f"{name:>21}: {spread:.3f} mag = {frac:6.1%} in the ratio"
    if name in T_SLOPE:
        s = t_slope(*T_SLOPE[name])
        line += f"; d ln R/d ln T {s:.2f}, so {np.log1p(frac) / s:6.1%} in T"
    out.append(line)
text = "\n".join(out)
print(text)
OUT.write_text(text + "\n")
