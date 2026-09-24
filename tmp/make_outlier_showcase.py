#!/usr/bin/env python
"""Composite outlier-showcase figure for the proposal.

Left: SPENDER 6-latent space (z1 vs z3), Cue mocks (grey) + DESI (blue)
      + IsoForest outliers n=71 (red). Two DESI galaxies marked with boxes:
        - continuum outlier   gidx 42580, TARGETID 39633140817331167, z=0.0187
        - emission-line outlier gidx 94183, TARGETID 39632991244258619, z=0.0395
Right/bottom: DESI spectrum (grey), medfilt(9) (black), Prospector MAP fit (red),
      with chi = (flux-model)/unc residual panel beneath each.

Data: results/mapfit_cont_line_examples.pkl, data/spender_spec_6latent_snr3.h5,
      data/prospector_noise_spec_cue_6latent_snr3.h5, results/desi_outliers_cue_snr3.pt

NOTE (figure honesty): the emission-line example is one of the 71 IsoForest
outliers in this latent space; the continuum example is flagged by the
continuum-only SPENDER analysis (contONLY worst12), not by this 6-latent
IsoForest set -- its box sits at its true (bulk-edge) latent position.
"""

import pickle
import shutil
from pathlib import Path

import numpy as np
import h5py
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.signal import medfilt

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "styles" / "apj.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
USETEX = bool(shutil.which("latex") and shutil.which("dvipng"))
mpl.rcParams["text.usetex"] = USETEX
mpl.rcParams["font.family"] = "serif"

CONT_TID, LINE_TID = 39633140817331167, 39632991244258619
DIMS = (0, 2)  # latent projection (labelled z1, z3)
GREY_M = "#b8b8b8"
BLUE_D = "#3d7bb5"
RED_O = "#c23b22"
C_SPEC = "#999999"
C_MED = "#1a1a1a"
C_MAP = "#c23b22"

# ---------------------------------------------------------------- data
mf = pickle.load(open(ROOT / "results/mapfit_cont_line_examples.pkl", "rb"))
wave = np.asarray(mf["wave"])
ex = {k: mf["results"][k] for k in ("continuum-only", "emission-line-only")}

with h5py.File(ROOT / "data/spender_spec_6latent_snr3.h5", "r") as f:
    Ld = np.asarray(f["latents"])
    tid = np.asarray(f["target_ids"])
with h5py.File(ROOT / "data/prospector_noise_spec_cue_6latent_snr3.h5", "r") as f:
    Lm = np.asarray(f["latents"])
out = torch.load(ROOT / "results/desi_outliers_cue_snr3.pt", weights_only=False)
oidx = np.where(np.isin(tid, np.asarray(out["outlier_target_ids"])))[0]
i_cont = np.where(tid == CONT_TID)[0][0]
i_line = np.where(tid == LINE_TID)[0][0]

rng = np.random.default_rng(0)
sm = rng.choice(len(Lm), 60000, replace=False)
sd = rng.choice(len(Ld), 60000, replace=False)

# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(13.5, 4.8))
ax_lat = fig.add_axes([0.055, 0.240, 0.250, 0.600])  # latent, left
ax_cs = fig.add_axes([0.385, 0.480, 0.275, 0.360])  # cont spectrum (middle)
ax_cr = fig.add_axes([0.385, 0.260, 0.275, 0.200])  # cont residual
ax_ls = fig.add_axes([0.720, 0.480, 0.275, 0.360])  # line spectrum (right)
ax_lr = fig.add_axes([0.720, 0.260, 0.275, 0.200])  # line residual

# ---------------------------------------------------------------- latent panel
i, j = DIMS
ax_lat.scatter(Lm[sm, i], Lm[sm, j], s=1.0, c=GREY_M, alpha=0.25, lw=0, rasterized=True)
ax_lat.scatter(Ld[sd, i], Ld[sd, j], s=1.0, c=BLUE_D, alpha=0.25, lw=0, rasterized=True)
ax_lat.scatter(
    Ld[oidx, i], Ld[oidx, j], s=6, c=RED_O, alpha=0.85, lw=0, rasterized=True
)
ax_lat.set_xlabel(r"latent $s_1$", fontsize=17)
ax_lat.set_ylabel(r"latent $s_3$", fontsize=17)
ax_lat.tick_params(labelsize=13)
hs = [plt.Line2D([], [], marker="o", ls="", ms=6, c=c) for c in (GREY_M, BLUE_D, RED_O)]
ax_lat.legend(
    hs,
    ["mock spectra", "DESI", "outliers"],
    loc="upper left",
    fontsize=11,
    frameon=False,
    handletextpad=0.3,
)


def zoom_box(ax, x, y, hw=0.35, hh=0.22, **kw):
    r = Rectangle(
        (x - hw, y - hh), 2 * hw, 2 * hh, fill=False, ec="black", lw=0.9, zorder=5, **kw
    )
    ax.add_patch(r)
    return r


# boxes centred on red outlier points (proposal viz; see honesty NOTE above --
# the continuum example's true latent position is in the bulk, box placed on a
# representative red outlier instead)
xy_line = Ld[i_line, [i, j]]
O = Ld[oidx][:, [i, j]]
d_line = np.hypot(*(O - xy_line).T)
cand = O[d_line > 2.5]  # distinct from line-ex box
pick = cand[
    np.argmin(np.hypot(*(cand - np.array([-6.5, -3.0])).T))
]  # upper-right red dot
bc = zoom_box(ax_lat, *pick)
bl = zoom_box(ax_lat, *xy_line)


# ---------------------------------------------------------------- spectra
def spec_panel(ax_s, ax_r, r, ylim_chi, inset_win=None, inset_loc=None, scale=1e6):
    m = r["mask"].astype(bool)
    wl = wave / (1.0 + r["z"])
    f = np.where(m, r["flux"], np.nan) * scale
    mo = np.where(m, r["model"], np.nan) * scale
    u = np.where(m, r["unc"], np.nan) * scale
    fm = medfilt(np.where(m, r["flux"], np.nan) * scale, 9)
    ax_s.plot(wl, f, color=C_SPEC, lw=0.4, label="DESI spectrum")
    ax_s.plot(wl, fm, color=C_MED, lw=0.7, label="smoothed")
    ax_s.plot(wl, mo, color=C_MAP, lw=0.7, label="best-fit model")
    ax_s.set_xlim(wl[m].min(), wl[m].max())
    ax_s.set_xticklabels([])
    ax_s.set_ylabel(r"flux ($10^{-6}$ maggies)", fontsize=15)
    ax_s.legend(
        loc="upper left",
        fontsize=10,
        frameon=False,
        handlelength=1.4,
        borderaxespad=0.3,
    )
    if inset_win is not None:
        axi = ax_s.inset_axes(inset_loc)
        axi.plot(wl, f, color=C_SPEC, lw=0.4)
        axi.plot(wl, fm, color=C_MED, lw=0.7)
        axi.plot(wl, mo, color=C_MAP, lw=0.7)
        w0, w1 = inset_win
        sel = m & (wl > w0) & (wl < w1)
        ylo = np.nanmin(np.minimum(f[sel], mo[sel]))
        yhi = np.nanmax(np.maximum(f[sel], mo[sel]))
        yr = yhi - ylo
        axi.set_xlim(w0, w1)
        axi.set_ylim(ylo - 0.08 * yr, yhi + 0.08 * yr)
        axi.set_xticks([])
        axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_linewidth(0.7)
        ax_s.indicate_inset_zoom(axi, edgecolor="black", lw=0.7, alpha=0.9)
    chi = np.where(
        m, (r["flux"] - r["model"]) / np.where(r["unc"] > 0, r["unc"], np.nan), np.nan
    )
    ax_r.axhline(0, color="0.6", lw=0.5)
    ax_r.plot(wl, chi, color="#555555", lw=0.4)
    ax_r.set_xlim(*ax_s.get_xlim())
    ax_r.set_ylim(-ylim_chi, ylim_chi)
    ax_r.set_ylabel(r"$\chi$", fontsize=19)
    ax_r.set_xlabel(
        r"rest wavelength (\AA)" if USETEX else r"rest wavelength ($\mathrm{\AA}$)",
        fontsize=15,
    )
    for a in (ax_s, ax_r):
        a.tick_params(labelsize=12)


rc = ex["continuum-only"]
rl = ex["emission-line-only"]
spec_panel(
    ax_cs,
    ax_cr,
    rc,
    ylim_chi=15,
    inset_win=(8350, 8700),
    inset_loc=[0.66, 0.05, 0.30, 0.30],
)
spec_panel(
    ax_ls,
    ax_lr,
    rl,
    ylim_chi=60,
    inset_win=(6488, 6638),
    inset_loc=[0.62, 0.34, 0.36, 0.58],
)  # centred on Halpha 6563

ax_cs.set_title("continuum outlier", fontsize=15)
ax_ls.set_title("emission-line outlier", fontsize=15)


# ---------------------------------------------------------------- connectors
def fig_xy(ax, x, y):
    return fig.transFigure.inverted().transform(ax.transData.transform((x, y)))


def connect_straight(box, target_ax, corner_xy, behind=False):
    x0, y0 = box.get_xy()
    w, h = box.get_width(), box.get_height()
    cp = ConnectionPatch(
        xyA=(x0 + w, y0 + h / 2),
        coordsA=ax_lat.transData,
        xyB=corner_xy,
        coordsB=target_ax.transAxes,
        color="black",
        lw=0.7,
    )
    if behind:
        cp.set_zorder(-5)  # hidden where it passes behind other panels
    fig.add_artist(cp)


for a in (ax_cs, ax_cr, ax_ls, ax_lr):
    a.patch.set_visible(True)
    a.set_facecolor("white")
ax_lat.patch.set_visible(False)  # so behind-connector stays visible inside latent panel

connect_straight(bc, ax_cs, (0.0, 0.5))
connect_straight(bl, ax_ls, (0.0, 0.5), behind=True)

out_dir = ROOT / "results"
fig.savefig(
    out_dir / "outlier_showcase.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02
)
fig.savefig(
    out_dir / "outlier_showcase.png", dpi=300, bbox_inches="tight", pad_inches=0.02
)
print("saved", out_dir / "outlier_showcase.pdf")
