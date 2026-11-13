"""Methods 2a and 2b: do the v2 mock spectra reach the Balmer absorption of the gallery105 PSBs?

QUESTION: (2a) For each gallery105 PSB, what fraction of mocks at the same 4000 A break and redshift
has HdA at least as strong? (2b) What fraction of mocks passes the Leung et al. (2024) PSB cut?
HYPOTHESIS: none registered.
INPUTS: the rebuilt subset (rebuild.py: flux, continuum, lines, redshift, index), the stored mocks
(mocks_v2_500000.h5, for the reproduction check), psb_spectra.npz (export_psb.py).
METHOD:
- Check: the rebuilt flux must reproduce the stored flux of the same mock, and continuum + lines the
  rebuilt flux; the largest relative differences are reported.
- Mock fluxes are maggies (f_nu); f_lambda = f_nu / lambda^2 up to a constant, which no index uses.
- HdA: Worthey & Ottaviani (1997) Table 1 bands, taken as air and moved to vacuum (DAP lists them as
  air, Westfall et al. 2019, spectralindices.tex l.39), mean form (l2 - l1)(1 - <F/F_c>) on f_lambda.
- Dn4000: mean f_nu in 4000-4100 A over mean f_nu in 3850-3950 A (Balogh et al. 1999, PSG.tex l.307,
  l.326; the f_nu convention and the vacuum frame are ours, the paper states neither).
- 2a: raw total spectra for mocks and PSBs alike (lines included). For each PSB, mocks with
  |Dn4000 - Dn4000_psb| <= 0.05 and |z - z_psb| <= 0.05; fraction with HdA >= HdA_psb. Wild et al.
  (2007, arXiv:0706.3113, l.2162-2169) define PSBs by Balmer lines stronger than expected for the break.
- 2b: Leung et al. (2024, arXiv:2309.16626, l.200): HdA > 3 A, W(Ha) < 10 A and log W(Ha) < 0.23 HdA -
  0.46, as the MaNGA DAP measures them (Westfall et al. 2019, arXiv:1901.00856): HdA on the continuum
  (lines removed, spectralindices.tex l.102); W(Ha) = line flux of the lines part in 6557.6-6571.6 A
  over the pseudo-continuum from the medians of the total spectrum in 6483.0-6513.0 and 6623.0-6653.0 A
  at the line centre 6564.632 A, divided by 1 + z (emissionlines.tex l.61, Eq. lineew). The DAP does not
  label the Halpha bands air or vacuum; vacuum is assumed. W(Ha) <= 0 counts as passing the log cut.
  Only mocks with the red sideband on the grid (z < 0.4766) can be tested.
Assumptions: mocks are noiseless; data indices include noise and the sky mask of load_by_index.
SEED: none (the subset is fixed in rebuild.py).
COMMAND: ``python indices.py <rebuilt.h5> <mocks_v2_500000.h5> <psb_spectra.npz> <out dir>``.
RESULT: pending.
FIGURES: none (indices.txt, indices.npz).
"""

import sys
from pathlib import Path

import h5py
import numpy as np

from hubersed.sps.utils import _airtovac

HDA = ((4083.50, 4122.25), (4041.60, 4079.75), (4128.50, 4161.00))
DN_BLUE, DN_RED = (3850.0, 3950.0), (4000.0, 4100.0)
HA, HA_MAIN, HA_BLUE, HA_RED = 6564.632, (6557.6, 6571.6), (6483.0, 6513.0), (6623.0, 6653.0)
DDN, DZ = 0.05, 0.05
WMAX = 9824.0


def hda(rest, flam, good):
    """Return HdA in rest-frame A, NaN when a band has no good pixels."""
    vac = [(_airtovac(lo), _airtovac(hi)) for lo, hi in HDA]
    sel = [good & (rest >= lo) & (rest <= hi) for lo, hi in vac]
    if any(s.sum() == 0 for s in sel):
        return np.nan
    (f_lo, f_hi), (b_lo, b_hi), (r_lo, r_hi) = vac
    wb, wr = (b_lo + b_hi) / 2, (r_lo + r_hi) / 2
    cb, cr = flam[sel[1]].mean(), flam[sel[2]].mean()
    cont = cb + (cr - cb) * (rest[sel[0]] - wb) / (wr - wb)
    return float((f_hi - f_lo) * (1 - np.mean(flam[sel[0]] / cont)))


def dn4000(rest, fnu, good):
    """Return Dn4000 from mean f_nu in the Balogh bands, NaN when a band has no good pixels."""
    b = good & (rest >= DN_BLUE[0]) & (rest <= DN_BLUE[1])
    r = good & (rest >= DN_RED[0]) & (rest <= DN_RED[1])
    return float(fnu[r].mean() / fnu[b].mean()) if b.any() and r.any() else np.nan


def w_halpha(wave, z, total_flam, lines_flam):
    """Return the DAP-style rest-frame W(Halpha) in A (emission positive), NaN off the grid."""
    a = 1 + z
    if HA_RED[1] * a > WMAX:
        return np.nan
    obs = wave
    blue = (obs >= HA_BLUE[0] * a) & (obs <= HA_BLUE[1] * a)
    red = (obs >= HA_RED[0] * a) & (obs <= HA_RED[1] * a)
    main = (obs >= HA_MAIN[0] * a) & (obs <= HA_MAIN[1] * a)
    cb, cr = np.median(total_flam[blue]), np.median(total_flam[red])
    xb, xr = np.mean(obs[blue]), np.mean(obs[red])
    c = cb + (cr - cb) * (HA * a - xb) / (xr - xb)
    dl = np.gradient(obs)
    flux = np.sum(lines_flam[main] * dl[main])
    return float(flux / c / a)


def main(rebuilt, stored, psbfile, out):
    """Run the check, 2a and 2b, and write the tables."""
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(rebuilt, "r") as h:
        idx, z, ok = h["index"][:], h["redshift"][:], h["ok"][:]
        flux, cont, lines = h["flux"][:], h["continuum"][:], h["lines"][:]
    with h5py.File(stored, "r") as h:
        wave = h["wavelength"][:]
        sflux = np.array([h["fluxes"][i] for i in idx])
    lines_out = []
    good = ok & np.all(np.isfinite(flux), axis=1)
    rel = np.abs(flux[good] - sflux[good]) / np.median(np.abs(sflux[good]), axis=1)[:, None]
    rel2 = (
        np.abs(cont[good] + lines[good] - flux[good])
        / np.median(np.abs(flux[good]), axis=1)[:, None]
    )
    lines_out.append(f"rebuilt {good.sum()} of {idx.size}; max |rebuilt - stored| / median "
                     f"{np.nanmax(rel):.2e}; max |cont + lines - flux| / median {np.nanmax(rel2):.2e}")  # fmt: skip

    lam2 = wave**2
    m_hda, m_dn, c_hda, m_wha = (np.full(idx.size, np.nan) for _ in range(4))
    for k in np.flatnonzero(good):
        rest = wave / (1 + z[k])
        allpix = np.ones(wave.size, bool)
        m_hda[k] = hda(rest, flux[k] / lam2, allpix)
        m_dn[k] = dn4000(rest, flux[k], allpix)
        c_hda[k] = hda(rest, cont[k] / lam2, allpix)
        m_wha[k] = w_halpha(wave, z[k], flux[k] / lam2, lines[k] / lam2)

    # 2b: Leung et al. (2024) cut on the mocks where Halpha can be measured
    meas = np.isfinite(m_wha) & np.isfinite(c_hda)
    logw_ok = np.where(m_wha > 0, np.log10(np.clip(m_wha, 1e-30, None)) < 0.23 * c_hda - 0.46, True)
    leung = meas & (c_hda > 3) & (m_wha < 10) & logw_ok
    lines_out.append(f"2b Leung cut: {leung.sum()} of {meas.sum()} mocks with Halpha on the grid "
                     f"({leung.sum() / max(meas.sum(), 1):.5f})")  # fmt: skip

    # 2a: the PSBs against mocks at matched break and redshift
    ps = np.load(psbfile)
    lines_out += ["", "2a nn  z      Dn4000  HdA    N_match  frac(mock HdA >= PSB)"]
    rows = []
    for n, zz, f, iv in zip(ps["nn"], ps["z"], ps["flux"], ps["ivar"], strict=True):
        rest = ps["wave"] / (1 + zz)
        g = iv > 0
        p_hda = hda(rest, f, g)
        p_dn = dn4000(rest, f * ps["wave"] ** 2, g)
        sel = good & (np.abs(m_dn - p_dn) <= DDN) & (np.abs(z - zz) <= DZ) & np.isfinite(m_hda)
        frac = float(np.mean(m_hda[sel] >= p_hda)) if sel.any() else np.nan
        rows.append((n, zz, p_dn, p_hda, sel.sum(), frac))
        lines_out.append(
            f"   {n:3d} {zz:.3f}  {p_dn:5.2f}  {p_hda:5.2f}  {sel.sum():7d}  {frac:.4f}"
        )
    np.savez(out / "indices.npz", index=idx, z=z, ok=good, hda=m_hda, dn4000=m_dn, hda_cont=c_hda,
             w_halpha=m_wha, leung=leung, psb=np.array(rows))  # fmt: skip
    text = "\n".join(lines_out)
    (out / "indices.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
