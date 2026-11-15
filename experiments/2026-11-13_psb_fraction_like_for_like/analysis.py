"""Post-starburst fractions of the DESI parent and the v2 mocks with the same cuts and measurements.

QUESTION: In matched redshift bins (and above log M* 10.6), what fraction of DESI parent galaxies and
of v2 mocks (noised as the flows saw them, and noiseless) passes the same spectral PSB cuts?
HYPOTHESIS: none registered. The mock SFH census (2026-11-13_mock_psb_coverage sfh.py) found 0.13% of
mocks PSB-like at every redshift, while Rowlands et al. (2018, arXiv:1707.07989, l.296) find 0.06-0.2%
at z < 0.35 rising to 2-3% at z ~ 0.7.
INPUTS: results/2026-11-13_psb_fraction_like_for_like/indices_{desi,noised,rebuilt}.npz (measure.py).
METHOD: every sample is first cut to median per-pixel S/N > 3, the parent cut and the cut of the mocks
the flows saw (post_mac.sh, get_latent_space.py --snr_min 3). Cuts on the raw indices of measure.py,
emission positive:
- Goto (2007, arXiv:0801.1106, ea5_astroph.tex l.333, with EW(Hdelta) replaced by HdA): HdA > 5 A,
  EW([OII]) < 2.5 A, and EW(Halpha) < 3 A where Halpha is measured (z < ~0.48).
- Leung et al. (2024, arXiv:2309.16626, l.200): HdA > 3 A, W(Halpha) < 10 A and
  log W(Halpha) < 0.23 HdA - 0.46 (W <= 0 passes), only where Halpha is measured.
Fractions per z bin of 0.1 from 0 to 0.6, all masses and log M* >= 10.6 (DESI FastSpecFit LOGMSTAR,
mocks logmass, the mass formed), and median per-pixel S/N > 8 as Leung et al. require (l.198-200; DESI
S/N from data/dataset_z_snr.npz, the latent-cut S/N; mocks from measure.py snr; noiseless mocks
count as above); errors sqrt(f (1 - f) / N).
Assumptions: raw indices keep emission infill and absorption under the lines, alike in data and mocks;
the cuts were calibrated on other surveys; mock mass formed exceeds stellar mass.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_psb_fraction_like_for_like/analysis.py``.
RESULT: 2026-11-14, fractions.txt. At z < 0.3 and S/N > 8, Goto passes 1.6-5.9% of noised mocks,
1.5-2.0% of noiseless rebuilt mocks and 0.2-0.8% of DESI; Leung 6.0-6.6%, 6.3-7.7% and 2.5-2.8%.
Noise raises the mock Goto fraction but not Leung; noiseless mocks still pass both cuts 2-10x more
than DESI, far above the 0.13% PSB-like SFHs (sfh.py), so the cuts do not trace the mock SFH.
Mock fractions barely change with S/N cut or mass. DESI above z 0.4 has under 15 spectra at
S/N > 8. Assumptions: raw indices, mock mass formed, Goto without Halpha above z ~ 0.48; rebuilt
is a random 20,000 of all 500,000 mocks, the noised set is cut at S/N > 3.
FIGURES: none (fractions.txt).
"""

import numpy as np

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-13_psb_fraction_like_for_like"
ZBINS = np.round(np.arange(0.0, 0.61, 0.1), 2)
MMIN = 10.6
SNR_BASE = 3.0  # parent latent cut (scripts/get_latent_space.py --snr_min 3), also the flows' mocks
SNR_MIN = 8.0  # Leung et al. 2024 l.198-200: median spectral SNR > 8 per pixel


def cuts(d):
    """Return the Goto and Leung masks and the masks of where each can be applied."""
    hda, eha, eoii = d["hda"], d["ew_ha"], d["ew_oii"]
    ha_ok = np.isfinite(eha)
    goto_ok = np.isfinite(hda) & np.isfinite(eoii)
    goto = goto_ok & (hda > 5) & (eoii < 2.5) & (~ha_ok | (eha < 3))
    leung_ok = np.isfinite(hda) & ha_ok
    logw = np.where(eha > 0, np.log10(np.clip(eha, 1e-30, None)), -np.inf)
    leung = leung_ok & (hda > 3) & (eha < 10) & (logw < 0.23 * hda - 0.46)
    return {"goto": (goto, goto_ok), "leung": (leung, leung_ok)}


def snr_of(name, d):
    """Return the median per-pixel S/N of each spectrum (inf for the noiseless rebuilt mocks)."""
    if name == "desi":
        s = np.load(PATHS["DATA"] / "dataset_z_snr.npz")
        lookup = dict(zip(s["tid"].tolist(), s["snr"].tolist(), strict=True))
        return np.array([lookup.get(int(t), np.nan) for t in d["id"]])
    if name == "noised":
        s = np.load(OUT / "snr_noised.npz")
        lookup = dict(zip(s["id"].tolist(), s["snr"].tolist(), strict=True))
        return np.array([lookup.get(int(t), np.nan) for t in d["id"]])
    return np.full(d["z"].size, np.inf)


def table(name, d):
    """Return the lines of one sample's fractions."""
    c = cuts(d)
    snr = snr_of(name, d)
    lines = [f"{name}: {d['z'].size} spectra"]
    base = snr > SNR_BASE  # the parent cut and the flows' mock cut
    lines[0] += f", {base.sum()} with median S/N > {SNR_BASE}"
    for mlab, msel in (
        ("all masses", base),
        (f"log M* >= {MMIN}", base & (d["logmass"] >= MMIN)),
        (f"median S/N > {SNR_MIN} (all masses)", snr > SNR_MIN),
    ):
        lines.append(f"  {mlab}")
        for lo, hi in zip(ZBINS[:-1], ZBINS[1:], strict=True):
            zsel = msel & (d["z"] >= lo) & (d["z"] < hi)
            parts = []
            for cut, (hit, ok) in c.items():
                n = int(np.sum(zsel & ok))
                f = np.sum(zsel & hit) / n if n else np.nan
                err = np.sqrt(f * (1 - f) / n) if n else np.nan
                parts.append(f"{cut} {100 * f:6.2f} +- {100 * err:5.2f}% (N {n:6d})")
            lines.append(f"    z {lo:.1f}-{hi:.1f}: " + "; ".join(parts))
    return lines


def main():
    """Write the fraction tables."""
    lines = []
    for name in ("desi", "noised", "rebuilt"):
        p = OUT / f"indices_{name}.npz"
        if p.exists():
            lines += table(name, np.load(p)) + [""]
        else:
            lines += [f"{name}: not measured", ""]
    text = "\n".join(lines)
    (OUT / "fractions.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
