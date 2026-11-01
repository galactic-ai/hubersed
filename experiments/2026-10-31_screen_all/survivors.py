"""Physics follow-up list: the screened outliers with their sky status, properties and seed agreement (#17).

INPUTS: screened94.npz (screen.py), sky_rescore.txt, qa_outliers.txt (numexp, sky_min), FastSpecFit
VAC, 20-seed flow ensembles (results/flow_ensemble, copied from /work). Status: "sky cleared" when
dropping sky g < 21 exposures ends the common flag; the rest stay on the list. Seeds counts the
ensemble seeds where both flows flag the target (log p at or below that seed's threshold).
COMMAND: ``uv run --no-sync python experiments/2026-10-31_screen_all/survivors.py``.
OUTPUT: results/2026-10-31_screen_all/survivors.txt and survivors.png.
RESULT: 2026-10-31. 36 of the 48 screened outliers stay on the list; 11 are flagged by both flows
in at least 15 of 20 seeds. 39627757533007793 is flagged in 6 of 20.
"""

# %%
import io
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_screen_all"
CUTS = RES / "2026-10-31_logp_vs_colour" / "cutouts"
CUT = "https://www.legacysurvey.org/viewer/cutout.jpg?ra={}&dec={}&layer=ls-dr9&pixscale=0.262&size=96"

keep = np.load(OUT / "screened94.npz")["target_ids"].tolist()
sky_status = {}
for line in (OUT / "sky_rescore.txt").read_text().splitlines():
    p = line.split()
    if p and p[0].isdigit():
        sky_status[int(p[0])] = " ".join(p[8:])
lines = (RES / "2026-10-31_spectrum_qa_cont_outliers" / "qa_outliers.txt").read_text().splitlines()
col = {k: i for i, k in enumerate(lines[0].split())}
qa = {int(p[0]): p for p in (x.split() for x in lines[1:])}

ens = {t: np.load(RES / "flow_ensemble" / f"ens_nsf_{t}latent.npz") for t in ("cont10", "cont15")}
etid = np.asarray(ens["cont10"]["desi_tid"], np.int64)
assert np.array_equal(etid, np.asarray(ens["cont15"]["desi_tid"], np.int64))
erow = {int(t): i for i, t in enumerate(etid)}
both = np.ones((20, len(etid)), bool)
for e in ens.values():
    both &= e["lp_desi"] <= e["thr"][:, None]

h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M, S = h["METADATA"].data, h["FASTSPEC"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}
rows = []
for t in keep:
    i, p = vrow[t], qa[t]
    n, smin = int(float(p[col["numexp"]])), float(p[col["sky_min"]])
    if t in sky_status:
        st = sky_status[t]
    elif n >= 2:
        st = "dark exposures only"
    else:
        st = "single exposure, bright sky" if smin < 21 else "single exposure, dark"
    rows.append(dict(
        tid=t, z=float(M["Z"][i]), logm=float(S["LOGMSTAR"][i]), dn=float(S["DN4000"][i]),
        ha=float(S["HALPHA_EW"][i]), snr=float(S["SNR_B"][i]), nexp=n, smin=smin, status=st,
        seeds=int(both[:, erow[t]].sum()), dgr=float(p[col["dgr_tot"]]),
    ))  # fmt: skip
follow = [r for r in rows if r["status"] != "cleared"]
follow.sort(key=lambda r: -r["seeds"])
hdr = (f"{'TARGETID':>18} {'z':>6} {'logM':>5} {'Dn4000':>6} {'HaEW':>6} {'SNR_B':>5} {'dgr':>6} "
       f"{'nexp':>4} {'sky min':>7} {'seeds':>5}  status")  # fmt: skip
txt = [f"{len(rows)} screened outliers; {len(rows) - len(follow)} cleared by the sky check; "
       f"{len(follow)} on the follow-up list (sorted by seeds, of 20).", "", hdr]  # fmt: skip
txt += [f"{r['tid']:>18} {r['z']:6.3f} {r['logm']:5.2f} {r['dn']:6.3f} {r['ha']:6.1f} {r['snr']:5.1f} "
        f"{r['dgr']:+6.3f} {r['nexp']:4d} {r['smin']:7.2f} {r['seeds']:5d}  {r['status']}"
        for r in follow]  # fmt: skip
text = "\n".join(txt)
(OUT / "survivors.txt").write_text(text + "\n")
print(text)

# %% gallery of the follow-up list


wave = DESI._wave_obs.numpy()
CUTS.mkdir(parents=True, exist_ok=True)
use_apj_style()
ncol = 4
nrow = int(np.ceil(len(follow) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(26, 3.6 * nrow))
for ax in axes.ravel()[len(follow) :]:
    ax.axis("off")
for ax, r in zip(axes.ravel(), follow, strict=False):
    t = r["tid"]
    flux, ivar, _, _ = load_by_index(int(tids_to_indices(np.array([t]))[0]))
    flux, ivar = np.asarray(flux, float), np.asarray(ivar, float)
    good = ivar > 0
    ax.plot(wave[good], flux[good], color="0.75", lw=0.4)
    ax.plot(wave[good], medfilt(flux[good], 21), color="k", lw=0.8)
    ax.set_title(f"{t}  z {r['z']:.3f}  Dn4000 {r['dn']:.2f}  seeds {r['seeds']}/20\n{r['status']}",
                 fontsize=10)  # fmt: skip
    jpg = CUTS / f"{t}.jpg"
    for _ in range(4):
        if jpg.exists():
            break
        i = vrow[t]
        try:
            url = CUT.format(float(M["RA"][i]), float(M["DEC"][i]))
            with urllib.request.urlopen(url, timeout=90) as resp:
                jpg.write_bytes(resp.read())
        except TimeoutError:
            print(f"{t}: cutout timeout")
    ins = ax.inset_axes([0.74, 0.5, 0.25, 0.48])
    ins.imshow(plt.imread(io.BytesIO(jpg.read_bytes()), format="jpg"), origin="upper")
    ins.add_patch(plt.Circle((47.5, 47.5), 0.75 / 0.262, fill=False, color="w", lw=1))
    ins.set_xticks([])
    ins.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "survivors.png", dpi=90, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'survivors.png'}")
