"""Which SV3 bright exposure carries the excess of each repeat-observed outlier of targets.txt (#17)?

QUESTION: for the outliers of targets.txt (the 16 labelled sky or data artifact in
2026-11-01_repeats92/classes.tsv, then the 5 whose repeat has a wrong redshift and 3 star-forming
galaxies with an SV3 offset, then the 22 gallery92 outliers still unsorted on 2026-11-02, which have
no repeat), which single SV3 bright exposures differ from the target's repeat spectrum (or from the
median of its SV3 exposures when there is no repeat), and do those
exposures have bright sky, poor seeing, a bad petal sky model, a fibre flag or a fibre offset?
HYPOTHESIS: none registered. If sky light causes the flag, the excess should sit in the exposures
with the brightest sky or a bad petal sky chi2.
INPUTS: results/2026-11-02_exposure_checks/exposures/<TARGETID>.npz (extract.py, DR1 healpix spectra files);
the reference spectrum, the other DR1 coadd of the same target with the highest TSNR2_BGS and the
same redshift, from results/2026-11-01_repeats92/sparcl_spectra.npz; when every repeat has a wrong
redrock redshift, the deepest repeat anyway (its flux does not depend on z), scored at the SV3
redshift; for NOREF targets, whose repeat is another object, the pixel median of the SV3 exposures; sky_mag_g_spec, seeing_gfa,
transparency_gfa and efftime_spec of each exposure from
results/2026-10-31_spectrum_qa_cont_outliers/sv3_bright_exposures.csv; PETALQA and FIBERQA of each
exposure-qa-<EXPID>.fits (iron exposures folder), cached in results/2026-11-02_exposure_checks/exposure_qa/;
encoders and seed 0 flows as 2026-11-01_repeats92/analysis.py.
Each exposure's arms are joined ivar-weighted as spender does; the excess is the median of exposure
minus reference flux in four observed windows (1e-17 erg/s/cm2/A), and each exposure is scored on
both flows after spender's steps.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-02_exposure_checks/analysis.py``.
RESULT: pending.
FIGURES: results/2026-11-02_exposure_checks/ exposure_checks.txt and exposures/<NN>_<TARGETID>.png.
"""

# %%
import csv
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from scipy.signal import medfilt
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
RES = PATHS["RESULTS"]
OUT = RES / "2026-11-02_exposure_checks"
FLOWS = RES / "noised_cue_meanzero_wide_flow"
SPARCL = RES / "2026-11-01_repeats92" / "sparcl_spectra.npz"
EXPCSV = RES / "2026-10-31_spectrum_qa_cont_outliers" / "sv3_bright_exposures.csv"
QAURL = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/exposures/{n}/{e:08d}/exposure-qa-{e:08d}.fits"
TAGS = ("cont10", "cont15")
WINDOWS = ((3600, 4500), (4500, 5800), (5800, 7600), (7600, 9824))  # observed Angstrom
N_MEDIAN = 9
ZDIFF = 0.01
# repeat looks like another object (2026-11-01_repeats92/classes.tsv, 12 and 41): no reference
NOREF = {39627776310905551, 39632961875740873}


def merge_arms(d, k):
    """Return flux, ivar and mask of exposure ``k`` on the DESI grid, arms joined as spender does."""
    wave = DESI._wave_obs.numpy()
    flux, ivar = np.zeros(wave.size), np.zeros(wave.size)
    mask = np.zeros(wave.size, int)
    ncam = np.zeros(wave.size)
    win = {}
    for b in "BRZ":
        imin = np.argmin(np.abs(d[f"{b}_wave"][0] - wave))
        win[b] = np.arange(imin, imin + len(d[f"{b}_wave"]))
        assert np.abs(d[f"{b}_wave"] - wave[win[b]]).max() < 1e-4, "arm grid differs from DESI grid"
        ncam[win[b]] += 1
    for b in "BRZ":
        j, fl, iv, mk = win[b], d[f"{b}_flux"][k], d[f"{b}_ivar"][k], d[f"{b}_mask"][k]
        good = mk == 0
        ivar[j] += iv * good
        flux[j] += iv * good * fl
        one, many = ncam[j] == 1, ncam[j] > 1
        mask[j[one]] = mk[one]
        mask[j[many]] = mask[j[many]] & mk[many]
    ok = ivar > 0
    flux[ok] /= ivar[ok]
    return flux, ivar, mask


def prepare(flux, ivar, mask, z):
    """Return spender's normalised spectrum and weights, as 2026-11-01_repeats92/analysis.py."""
    ivar = np.asarray(ivar, float).copy()
    ivar[(np.asarray(mask) != 0) | (ivar <= 1e-6)] = 0
    spec = torch.from_numpy(np.asarray(flux, float).astype(np.float32))
    w = torch.from_numpy(ivar.astype(np.float32))
    w[DESI._skyline_mask] = 0
    rest = DESI._wave_obs / (1 + torch.tensor(np.float32(z)))
    sel = (w > 0) & (rest > 5300) & (rest < 5850)
    norm = torch.median(spec[sel]) if sel.count_nonzero() > 0 else torch.tensor(0.0)
    if not torch.isfinite(norm):
        norm = torch.tensor(0.0)
    else:
        spec /= norm
    w *= norm**2
    return spec.numpy(), w.numpy()


def load_flow(tag):
    """Return the seed 0 flow of a tag in eval mode, with its scaler mean and scale."""
    d = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
    nde = build_flow(d["method"], d["dim"], d["hidden"], d["num_transforms"], d["num_bins"])
    nde.load_state_dict(d["state_dict"])
    nde.eval()
    return nde, np.asarray(d["scaler_mean"]), np.asarray(d["scaler_scale"]), d["encoder"]


def exposure_qa(expid, night):
    """Return the PETALQA and FIBERQA tables of one exposure, downloading the file once."""
    path = OUT / "exposure_qa" / f"exposure-qa-{expid:08d}.fits"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(QAURL.format(n=night, e=expid), path)
    with fits.open(path) as h:
        return h["PETALQA"].data.copy(), h["FIBERQA"].data.copy()


# %% targets, reference spectra, exposure conditions
targets = [line.split() for line in (HERE / "targets.txt").read_text().splitlines()]
sp = np.load(SPARCL, allow_pickle=True)
cond = {int(r["expid"]): r for r in csv.DictReader(open(EXPCSV))}
wave = DESI._wave_obs.numpy()
rows, figs = [], []
for tid, _, nn in targets:
    t = int(tid)
    d = np.load(OUT / "exposures" / f"{t}.npz", allow_pickle=True)
    idx = [i for i in range(len(sp["targetid"])) if int(sp["targetid"][i]) == t]
    i0 = [i for i in idx if (str(sp["survey"][i]), str(sp["program"][i])) == ("sv3", "bright")][0]
    z = float(sp["redshift"][i0])
    ok = [i for i in idx if i != i0 and int(sp["redshift_warning"][i]) == 0
          and int(sp["coadd_fiberstatus"][i]) == 0]  # fmt: skip
    same_z = [i for i in ok if abs(float(sp["redshift"][i]) - z) / (1 + z) <= ZDIFF]
    merged = [merge_arms(d, k) for k in range(len(d["EXPID"]))]
    if t in NOREF or not ok:
        iref = None
        stack = np.array([np.where((m == 0) & (iv > 0), f, np.nan) for f, iv, m in merged])
        ref = np.nanmedian(stack, axis=0)
        refgood = np.isfinite(ref)
        ref = np.where(refgood, ref, 0.0)
        refname = "median of the SV3 bright exposures"
    else:
        iref = max(same_z or ok, key=lambda i: float(sp["tsnr2_bgs"][i]))
        ref = np.asarray(sp["flux"][iref], float)
        refgood = (np.asarray(sp["mask"][iref]) == 0) & (np.asarray(sp["ivar"][iref]) > 0)
        refname = f"{sp['survey'][iref]}/{sp['program'][iref]}"
        if not same_z:
            refname += f" (redrock z {float(sp['redshift'][iref]):.4f}, wrong; scored at SV3 z)"
    refprep = (
        prepare(sp["flux"][iref], sp["ivar"][iref], sp["mask"][iref], z)
        if iref is not None
        else None
    )
    exps = []
    for k, expid in enumerate(d["EXPID"]):
        expid, night = int(expid), int(d["NIGHT"][k])
        flux, ivar, mask = merged[k]
        good = (mask == 0) & (ivar > 0) & refgood
        excess = [
            float(np.median((flux - ref)[good & (wave >= lo) & (wave < hi)])) for lo, hi in WINDOWS
        ]
        pq, fq = exposure_qa(expid, night)
        prow = pq[pq["PETAL_LOC"] == int(d["PETAL_LOC"][k])][0]
        frow = fq[fq["TARGETID"] == t]
        c = cond.get(expid, {})
        exps.append(dict(
            expid=expid, night=night, tile=int(d["TILEID"][k]), petal=int(d["PETAL_LOC"][k]),
            fiber=int(d["FIBER"][k]), fstat=str(d["fiberstatus_names"][k]) or "-",
            qafstat=int(frow["QAFIBERSTATUS"][0]) if len(frow) else -1,
            dxy=float(np.hypot(d["DELTA_X"][k], d["DELTA_Y"][k])) * 1000,
            sky=float(c.get("sky_mag_g_spec", "nan")), seeing=float(c.get("seeing_gfa", "nan")),
            efft=float(c.get("efftime_spec", "nan")),
            chi2=(float(prow["BSKYCHI2PDF"]), float(prow["RSKYCHI2PDF"]), float(prow["ZSKYCHI2PDF"])),
            excess=excess, flux=flux, good=(mask == 0) & (ivar > 0),
            prep=prepare(flux, ivar, mask, z),
        ))  # fmt: skip
    figs.append((nn, t, z, refname, ref, refgood, exps))
    rows.append([nn, t, z, refname, refprep, {}, exps])

# %% score every exposure on both flows
inst = DESI()
for tag in TAGS:
    nde, mean, scale, enc = load_flow(tag)
    model = load_model(str(PATHS["DATA"] / "checkpoints" / enc), inst, map_location="cpu",
                       weights_only=False).float().eval()  # fmt: skip
    o = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False,
                   map_location="cpu")  # fmt: skip
    thr = float(o["threshold"])
    for row in rows:
        exps, refprep = row[6], row[4]
        x = np.array([e["prep"][0] for e in exps] + ([refprep[0]] if refprep is not None else []))
        with torch.no_grad():
            lat = model.encode(torch.from_numpy(x)).numpy()
            lp = nde.log_prob(torch.from_numpy(((lat - mean) / scale).astype(np.float32))).numpy()
        for e, v in zip(exps, lp[: len(exps)], strict=True):
            e[tag] = (float(v), v <= thr)
        if refprep is not None:
            row[5][tag] = (float(lp[-1]), lp[-1] <= thr)

# %% table
lines = [
    "Per SV3 bright exposure: sky g (mag, brighter = smaller), seeing (arcsec), EFFTIME (s), petal sky "
    "chi2 B/R/Z (exposure QA), fibre offset |DELTA| (um), FIBERSTATUS, QAFIBERSTATUS, excess = median "
    "exposure - reference flux (1e-17) in observed 3600-4500, 4500-5800, 5800-7600, 7600-9824 A, and "
    "log p (* = flagged).",
    "",
]
for nn, t, z, refname, _, reflp, exps in rows:
    rl = "  ".join(f"{tag} {v[0]:.2f}{'*' if v[1] else ''}" for tag, v in reflp.items())
    lines.append(f"== {nn} {t} z {z:.4f}  reference {refname}" + (f": {rl}" if rl else ""))
    for e in exps:
        lp = "  ".join(f"{e[tag][0]:7.2f}{'*' if e[tag][1] else ' '}" for tag in TAGS)
        lines.append(
            f"  {e['expid']:6d} {e['night']} tile {e['tile']:4d} p{e['petal']} f{e['fiber']:4d}  "
            f"sky {e['sky']:5.2f} see {e['seeing']:4.2f} eff {e['efft']:5.0f}  "
            f"chi2 {e['chi2'][0]:5.2f}/{e['chi2'][1]:5.2f}/{e['chi2'][2]:5.2f}  d {e['dxy']:4.0f}  "
            f"fs {e['fstat']:>8} qa {e['qafstat']:>8}  excess "
            + " ".join(f"{x:+5.2f}" for x in e["excess"])
            + f"  logp {lp}"
        )  # fmt: skip
text = "\n".join(lines)
(OUT / "exposure_checks.txt").write_text(text + "\n")
print(text)

# %% figures: each exposure and the reference, and exposure minus reference
use_apj_style()
(OUT / "exposures").mkdir(exist_ok=True)
colours = plt.cm.viridis(np.linspace(0, 0.85, 8))
for nn, t, z, refname, ref, refgood, exps in figs:
    fig, ax = plt.subplots(
        2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
    )
    rest = wave / (1 + z)
    lab = f"{refname} (reference)"
    ax[0].plot(rest[refgood], medfilt(ref[refgood], N_MEDIAN), color="k", lw=0.8, label=lab)
    for e, c in zip(exps, colours, strict=False):
        g = e["good"]
        lab = (f"EXPID {e['expid']} (sky g {e['sky']:.2f}; cont10 {e['cont10'][0]:.1f}, "
               f"cont15 {e['cont15'][0]:.1f})")  # fmt: skip
        ax[0].plot(rest[g], medfilt(e["flux"][g], N_MEDIAN), color=c, lw=0.6, label=lab)
        both = g & refgood
        diff = np.full(wave.size, np.nan)
        diff[both] = medfilt(e["flux"][both] - ref[both], 51)
        ax[1].plot(rest, diff, color=c, lw=0.8)
    ax[1].axhline(0, color="0.3", lw=0.6)
    ax[0].set_title(f"{nn}/92   {t}   z {z:.3f}", fontsize=13)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]", fontsize=13)
    ax[1].set_ylabel("minus reference, 51 px", fontsize=12)
    ax[1].set_xlabel(r"rest wavelength [\AA]", fontsize=13)
    ax[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.savefig(OUT / "exposures" / f"{nn}_{t}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
print(f"wrote {len(figs)} figures to {OUT / 'exposures'}")
