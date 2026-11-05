"""Does [O III] in 39627770174637084 need a second kinematic component once it has its own width?

QUESTION: the shared model ties every non-Balmer line to the Balmer second-component fraction and
to one narrow width. Once the high-ionization lines ([O III], [Ne III], [Ar IV], [Ne IV], He II)
have their own narrow width, do they still need a second component? EmFit (DESI DR1) kept one
17.3 km/s Gaussian for [O III] 5007 of this target, with a reduced chi2 of 6.43 in its [O III]
window (knowledge/summary_emfit_oiii_treatment.md, UberSED notes).

HYPOTHESIS: if the [O III] wings are real, F0 (one Gaussian of free width) cannot match both the
core and the wings, and F1 (free high-ionization fraction) lowers chi2 in the [O III] windows by
much more than one parameter's worth. If the shared fit only needed a wider [O III], F0 matches
F1. At DESI resolution a wider single Gaussian and a narrow core plus a second component are
partly degenerate, so the answer rests on the profile shape at the few-pixel level.

INPUTS: fit_*.npz and profile_*.npz of fit.py in results/2026-10-13_oiii_second_component/, the
shared MAP map_rmatrix_mask.npz of results/2026-10-13_nautilus_resolution_matrix/, and the data,
resolution matrix and mask of oiii_setup.py.

SEED: none. Every start is listed in oiii_setup.starts; none is random.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix:experiments/2026-10-13_oiii_second_component
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_oiii_second_component/analysis.py``

RESULT: (oiii_second_component.txt) yes. chi2 over 5530 pixels is 6091.8 for shared, 6398.7 for F0
and 6084.3 for F1, so F0 - F1 = 314.4 for one parameter (284 after scaling by the F1 reduced chi2
1.106), with 268.6 of it in the [O III] 5007 window and 52.6 in [O III] 4959. F0 widens [O III] to
33.2 km/s and leaves wing residuals on both sides of 5007. F1 puts 0.175 of the high-ionization
flux in the second component, with a narrow width of 22.3 km/s, close to the Balmer 0.179 and
24.9 km/s and the low-ionization 0.210 and 21.5 km/s. So the shared model was nearly right for
[O III] (F1 - shared = -7.5 for three more parameters). The profile over the high-ionization
fraction has its minimum at 0.185 with a dchi2 = 1 half-width of 0.011. The profile over the
low-ionization fraction has its minimum at 0.213 with a half-width of 0.016, and fraction 0
costs 161.6, mostly in [N II] 6584 and [S II] 6716. [Ne III] 3869 prefers F0 by 3.2. Each
optimum moved by less than 0.06 in chi2 on a rerun, and the F0 optimum is 0.7 above the profile
point at fraction 0. The statistical errors leave out the LSF and the tie of the
high-ionization second component to the Balmer width and shift.

FIGURES: oiii_windows.png (data, shared, F0 and F1 in the [O III] 4959, [O III] 5007, Hbeta,
Halpha + [N II] and [S II] windows, with chi below) and oiii_profile.png (chi2 against the
high-ionization fraction and against the low-ionization fraction, in total and in two windows).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from oiii_setup import (
    GAS,
    HIGH_WINDOWS,
    LINE_PROFILE,
    OUT,
    WINDOWS,
    at_prior_edge,
    build,
    chi2_parts,
    cue_fluxes,
    shared_map,
)
from scipy.stats import chi2 as chi2_dist

from hubersed.plotting.style import use_apj_style

HE_I_5016 = 5017.08  # rest vacuum, masked at +-4 A (model_setup.MISSING_LINES)
COLORS = {"shared": "#2a78d6", "F0": "#eb6834", "F1": "#1baf7a"}
STYLES = {"shared": "--", "F0": "-", "F1": "-"}
PLOT_WINDOWS = {
    "[O III] 4959": (4948, 4972),
    "[O III] 5007, He I 5016": (4995, 5025),
    r"H$\beta$": (4850, 4874),
    r"H$\alpha$ + [N II]": (6540, 6595),
    "[S II]": (6708, 6740),
}
lines = []


def say(s=""):
    """Print a line and keep it for the text summary."""
    print(s)
    lines.append(s)


# %% every start and every rerun
runs = {p.stem: dict(np.load(p)) for p in sorted(OUT.glob("fit_*.npz"))}
say("MAP fits of 39627770174637084, rmatrix LSF, corrected ivar, missing-line mask, C3K_ER.")
say("Every Powell run (map_fit, one start each). -lnp includes each model's prior constants,")
say("so compare -lnp only within a fit. chi2 is over the same 5530 pixels in every run.")
say(f"{'run':>16} {'start from':>14} {'-lnp':>13} {'chi2':>9} {'nfev':>6} {'success':>7} {'s':>6}")
for name, r in runs.items():
    say(
        f"{name:>16} {str(r['start']):>14} {float(r['fun']):13.3f} {float(r['chi2']):9.2f} "
        f"{int(r['nfev']):6d} {bool(r['success'])!s:>7} {float(r['seconds']):6.0f}"
    )
best = {}
for fit in ("F0", "F1"):
    mine = {k: v for k, v in runs.items() if k.startswith(f"fit_{fit}_")}
    k = min(mine, key=lambda n: float(mine[n]["fun"]))
    best[fit] = mine[k]
    say(f"best {fit}: {k}")
    for n, r in mine.items():
        if "rerun" in n:
            prev = runs[f"{str(r['start'])}"]
            dth = np.max(np.abs(r["theta"] - prev["theta"]))
            say(
                f"  {n} from {r['start']}: -lnp moved {float(r['fun']) - float(prev['fun']):+.4f}, "
                f"chi2 moved {float(r['chi2']) - float(prev['chi2']):+.3f}, max |dtheta| {dth:.2e}"
            )

# %% states at the three optima
p_shared, fun_shared = shared_map()
state = {}
for fit in ("shared", "F0", "F1"):
    model, obs, cue = build(fit)
    labels = model.theta_labels()
    theta = (
        np.array([p_shared[k] for k in labels], dtype=float)
        if fit == "shared"
        else np.array(best[fit]["theta"], dtype=float)
    )
    if fit != "shared":
        assert list(best[fit]["labels"]) == labels
    spec, parts, npix = chi2_parts(model, obs, cue, theta)
    state[fit] = dict(
        labels=labels,
        theta=theta,
        spec=spec,
        parts=parts,
        npix=npix,
        fluxes=cue_fluxes(model),
        lnprior=float(model.prior_product(theta)),
        edges=at_prior_edge(model, theta),
        params={k: float(np.atleast_1d(model.params[k])[0]) for k in model.params if k in labels}
        | {"eline_fbroad_forb": float(np.atleast_1d(model.params["eline_fbroad_forb"])[0])}
        | (
            {"eline_fbroad_forb_hi": float(np.atleast_1d(model.params["eline_fbroad_forb_hi"])[0])}
            if "eline_fbroad_forb_hi" in model.params
            else {}
        ),
        npar=len(labels),
    )
o = obs[0]
z = float(np.atleast_1d(model.params["zred"])[0])
rest = o.wavelength / (1 + z)

# %% chi2 per window
cols = ("shared", "F0", "F1")
say()
say("chi2 at each optimum. shared is map_rmatrix_mask.npz (map.txt); its high-ionization lines")
say("share eline_sigma_forb and the Balmer fraction. Pixels: fitted of total in the window.")
say(f"{'window':>16} {'pixels':>9} " + " ".join(f"{c:>9}" for c in cols) + f" {'F0-F1':>8}")
for k in state["F1"]["parts"]:
    n = state["F1"]["npix"][k]
    vals = [state[c]["parts"][k] for c in cols]
    say(
        f"{k:>16} {n[0]:4d}/{n[1]:<4d} "
        + " ".join(f"{v:9.2f}" for v in vals)
        + f" {vals[1] - vals[2]:8.2f}"
    )
say(f"{'free params':>16} {'':9} " + " ".join(f"{state[c]['npar']:9d}" for c in cols))
say(f"{'ln prior':>16} {'':9} " + " ".join(f"{state[c]['lnprior']:9.3f}" for c in cols))
say("ln prior includes constant TopHat terms that differ between models; within F0 and F1 only")
say("eline_fbroad_forb_hi differs (+0.105 for F1), the rest is the SFH prior.")

# %% which fitted pixels the masks remove in each window
say()
say("Unfitted pixels in each window (rest A). He I 5016 is masked at 5013.08-5021.08.")
for k, (a, b) in WINDOWS.items():
    w = (rest >= a) & (rest < b) & ~o.mask
    if w.any():
        idx = np.where(w)[0]
        runs_ = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        say(f"{k:>16}: " + ", ".join(f"{rest[r[0]]:.1f}-{rest[r[-1]]:.1f}" for r in runs_))

# %% the statistic
d = state["F0"]["parts"]["all"] - state["F1"]["parts"]["all"]
dof = o.mask.sum() - state["F1"]["npar"]
red = state["F1"]["parts"]["all"] / dof
say()
say("F0 against F1: F1 has one more free parameter, eline_fbroad_forb_hi.")
say(f"Delta chi2 (F0 - F1) = {d:.2f}")
say(
    f"  p (chi2, 1 dof) = {chi2_dist.sf(max(d, 0), 1):.2e}; with the null on the prior edge "
    f"(fraction 0), 0.5 chi2_0 + 0.5 chi2_1: p = {0.5 * chi2_dist.sf(max(d, 0), 1):.2e}"
)
say(
    f"  scaled by the F1 reduced chi2 {red:.3f} ({state['F1']['parts']['all']:.1f} / {dof}): "
    f"{d / red:.2f}"
)
say(
    f"  Delta chi2 in the high-ionization windows {sum(state['F0']['parts'][k] - state['F1']['parts'][k] for k in HIGH_WINDOWS):.2f}, "
    f"outside all windows {state['F0']['parts']['outside windows'] - state['F1']['parts']['outside windows']:.2f}"
)

# %% parameters, fluxes and prior edges
say()
say(f"{'parameter':>21} " + " ".join(f"{c:>10}" for c in cols))
keys = [*LINE_PROFILE, "eline_fbroad_forb", "eline_fbroad_forb_hi", *GAS, "sigma_smooth"]
for k in dict.fromkeys(keys):
    vals = []
    for c in cols:
        v = state[c]["params"].get(k)
        if v is None and c == "shared":
            v = {
                "eline_fbroad_forb": state[c]["params"]["eline_fbroad"],
                "eline_fbroad_forb_hi": state[c]["params"]["eline_fbroad"],
                "eline_sigma_forb_hi": state[c]["params"]["eline_sigma_forb"],
            }.get(k)
        if v is None and c == "F0" and k == "eline_fbroad_forb_hi":
            v = 0.0
        vals.append(f"{v:10.4f}" if v is not None else f"{'-':>10}")
    say(f"{k:>21} " + " ".join(vals))
say("shared values of eline_fbroad_forb(_hi) and eline_sigma_forb_hi are the tied ones.")
say()
say("Cue line fluxes in 1e-17 erg/s/cm2")
say(f"{'line':>21} " + " ".join(f"{c:>10}" for c in cols))
for k in state["F1"]["fluxes"]:
    say(f"{k:>21} " + " ".join(f"{state[c]['fluxes'][k]:10.2f}" for c in cols))
say()
for c in cols:
    say(f"at a TopHat prior edge (within 1% of the range), {c}: {state[c]['edges'] or 'none'}")

# %% profile likelihoods over the high- and low-ionization fractions
PROFILE_SHOWN = {
    "hi": ("eline_fbroad_forb_hi", (*HIGH_WINDOWS, "Hbeta", "Halpha", "[NII]6584", "[SII]6716")),
    "lo": (
        "eline_fbroad_forb",
        ("[NII]6548", "[NII]6584", "[SII]6716", "[OI]6300", "[OII]3726,9", "Halpha", "[OIII]5007"),
    ),
}
PROFILE_COLS = (
    ("eline_fbroad_forb_hi", "f_hi"),
    ("eline_sigma_forb_hi", "s_hi"),
    ("eline_fbroad_forb", "f_lo"),
    ("eline_sigma_forb", "s_lo"),
    ("eline_fbroad", "f_bal"),
    ("eline_sigma", "s_bal"),
    ("eline_sigma_broad", "s_b"),
    ("eline_vbroad", "v_b"),
)
lab = state["F1"]["labels"]
profiles = {}
for which, (pname, shown) in PROFILE_SHOWN.items():
    files = sorted(OUT.glob(f"profile_{which}_*.npz"))
    if not files:
        continue
    model, obs, cue = build("F1")
    prof = {}
    for p in files:
        r = dict(np.load(p))
        assert list(r["labels"]) == lab
        th = r["theta"]
        prof[float(th[lab.index(pname)])] = (th, chi2_parts(model, obs, cue, th)[1], r)
    profiles[which] = prof
    say()
    say(f"Profile over {pname}: at each value the other line profile parameters")
    say(f"({', '.join(k for k in (*LINE_PROFILE, 'eline_fbroad_forb_hi') if k != pname)})")
    say(f"and the gas parameters ({', '.join(GAS)}) are re-optimized from the best F1")
    say("point; every other parameter is held there. dchi2 is relative to the best F1 chi2.")
    say(
        f"{pname[6:]:>15} {'dchi2':>8} {'nfev':>5} "
        + " ".join(f"{k:>11}" for k in shown)
        + " "
        + " ".join(f"{c:>6}" for _, c in PROFILE_COLS)
    )
    for v in sorted(prof):
        th, parts, r = prof[v]
        say(
            f"{v:15.3f} {parts['all'] - state['F1']['parts']['all']:8.2f} {int(r['nfev']):5d} "
            + " ".join(f"{parts[k]:11.2f}" for k in shown)
            + " "
            + " ".join(
                f"{th[lab.index(k)]:6.2f}"
                if "sigma" in k or "vbroad" in k
                else f"{th[lab.index(k)]:6.3f}"
                for k, _ in PROFILE_COLS
            )
        )
    say(f"best F1 {pname} = {state['F1']['params'][pname]:.4f}")
    # parabola through the best F1 point and the grid points within dchi2 < 60 of it
    near = [(v, prof[v][1]["all"] - state["F1"]["parts"]["all"]) for v in prof]
    near = [(v, d) for v, d in near if d < 60] + [(state["F1"]["params"][pname], 0.0)]
    cq = np.polyfit(*np.array(near).T, 2)
    say(
        f"parabola through {len(near)} points with dchi2 < 60: minimum at {-cq[1] / (2 * cq[0]):.3f}, "
        f"dchi2 = 1 half-width {1 / np.sqrt(cq[0]):.3f} (statistical only, unscaled chi2)"
    )
if 0.0 in profiles.get("hi", {}):
    say(
        f"hi profile at 0 against F0 with every parameter free: chi2 "
        f"{profiles['hi'][0.0][1]['all']:.2f} vs {state['F0']['parts']['all']:.2f}"
    )

# %% line-window figure
use_apj_style()
fig, ax = plt.subplots(
    2,
    len(PLOT_WINDOWS),
    figsize=(3.3 * len(PLOT_WINDOWS), 5.2),
    gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.05},
    sharex="col",
)
for i, (title, (a, b)) in enumerate(PLOT_WINDOWS.items()):
    win = (rest >= a) & (rest < b)
    x, fitm = rest[win], o.mask[win]
    ax[0, i].plot(x, np.where(fitm, o.flux[win], np.nan), color="0.15", lw=1.0, label="DESI")
    for c in cols:
        sp = state[c]["spec"]
        m = win & o.mask
        c2 = float(np.sum(((o.flux[m] - sp[m]) / o.uncertainty[m]) ** 2))
        ax[0, i].plot(
            x,
            np.where(fitm, sp[win], np.nan),
            color=COLORS[c],
            ls=STYLES[c],
            lw=1.0,
            label=rf"{c}, $\chi^2$ = {c2:.1f}",
        )
        chi = (o.flux[win] - sp[win]) / o.uncertainty[win]
        ax[1, i].plot(x, np.where(fitm, chi, np.nan), color=COLORS[c], ls=STYLES[c], lw=0.9)
    if a < HE_I_5016 < b:
        for row in ax[:, i]:
            row.axvspan(HE_I_5016 - 4, HE_I_5016 + 4, color="0.9", zorder=0)
    ax[0, i].set_title(title)
    ax[0, i].legend(frameon=False, fontsize=6)
    for h in (-2, 2):
        ax[1, i].axhline(h, color="0.6", ls=":", lw=0.6)
    ax[1, i].axhline(0, color="0.4", lw=0.6)
    ax[1, i].set_ylim(-7, 7)
    ax[1, i].set_xlabel(r"rest wavelength [$\AA$]")
ax[0, 0].set_ylabel("flux [maggies]")
ax[1, 0].set_ylabel(r"$\chi$")
fig.savefig(OUT / "oiii_windows.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% profile figure
if profiles:
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ref = state["F1"]["parts"]["all"]
    for i, (which, prof) in enumerate(profiles.items()):
        pname = PROFILE_SHOWN[which][0]
        v = sorted(prof)
        ax[i].plot(v, [prof[x][1]["all"] - ref for x in v], "o-", color="0.2", ms=4, label="all")
        keys = ("[OIII]5007", "[OIII]4959") if which == "hi" else ("[NII]6584", "[SII]6716")
        for k, c in zip(keys, (COLORS["F1"], COLORS["F0"]), strict=True):
            base = state["F1"]["parts"][k]
            ax[i].plot(v, [prof[x][1][k] - base for x in v], "o-", color=c, ms=4, label=k)
        ax[i].set_xlabel(pname)
        ax[i].set_ylabel(r"$\chi^2 - \chi^2_{\rm F1}$")
        ax[i].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "oiii_profile.png", dpi=150)
    plt.close(fig)

# %% write the summary
(OUT / "oiii_second_component.txt").write_text("\n".join(lines) + "\n")
