"""Which continuum shape does the steep dust slope fit in the free_dust fit of 39627770174637084?

QUESTION: ridge_profile.py found ln L rising steadily as dust_index falls to -4, with Delta chi2
+80.9 at n = 0 against n = -4. Is the gain a broad colour, a step near one wavelength, or local
features?

Refits n = -4 and n = 0 from the free_dust pocoMC best point, as ridge_profile.py does, with the
dust_index prior widened to TopHat(-4, 1.5). Splits the chi2 difference into 200 A rest bins for
continuum and line pixels. Gives per bin the n = 0 to n = -4 continuum ratio and the median data
to model ratio of each fit.

RESULT: 2026-10-18, ridge_shape.txt and .png. Response is 1, so no calibration term. ln L at
n = -4 matches ridge_profile (101084.74 against 101084.89). The direct n = 0 refit lands 1.8 lower
than the stepped one, so Delta chi2 is 84.0 here against 80.9. The gain is continuum (+87.3), not
lines (-3.4). The two continua differ by a smooth wave of 1-3%, not a step. n = 0 over n = -4 is
0.97 below 3800 A, 0.985-0.995 at 3800-4800 A, 1.011 at 5000-5800 A, 0.99 at 6600-7600 A and
1.015 at 8600-9000 A. The largest part is 5000-5800 A (+36.6), where n = 0 leaves data/model
0.982. Then 7400-7800 A (+17.5), 4600-4800 A (+12.0) and 8800-9000 A (+11.1). n = 0 does better
at 4000-4200 A (-14.8) and 5800-6000 A (-14.2). Neither fit matches below 3800 A, where data/model
is 1.04-1.07 for n = -4 and 1.07-1.085 for n = 0.

INPUTS: pmc_final.state of the free_dust poco.py run. SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude:~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/ridge_shape.py --state <path>``.

OUTPUT: ridge_shape.txt, ridge_shape.npz and ridge_shape.png next to the state file.
"""

import argparse
from pathlib import Path

import fit
import matplotlib.pyplot as plt
import numpy as np
from mask_refit import best_point
from prospect.models.priors import TopHat
from region_chi2 import EDGES, LINE_FRAC, components
from ridge_profile import FREE, VARIANT, WIDE, refit

SLOPES = (-4.0, 0.0)


def main():
    """Run the two refits and write the table, arrays and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    model, obs, cue = fit.process_state(VARIANT)
    labels = list(model.theta_labels())
    i_n = labels.index("dust_index")
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)
    theta0 = best_point(args.state, model)  # before the prior changes prior_transform
    model.config_dict["dust_index"]["prior"] = TopHat(mini=WIDE[0], maxi=WIDE[1])

    fits = {}
    for n in SLOPES:
        start = theta0.copy()
        start[i_n] = n
        th, lnl, lnp = refit(model, start, FREE)
        spec, cont, line = components(model, obs, cue, th)
        resp = np.asarray(model._speccal, dtype=float)
        fits[n] = dict(theta=th, lnl=lnl, lnp=lnp, spec=spec, cont=cont, line=line, resp=resp)
        print(f"n {n}: lnL {lnl:.2f}, lnp {lnp:.2f}", flush=True)

    a, b = fits[SLOPES[0]], fits[SLOPES[1]]
    chi_a = ((o.flux - a["spec"]) / o.uncertainty) ** 2
    chi_b = ((o.flux - b["spec"]) / o.uncertainty) ** 2
    is_line = np.abs(a["line"]) > LINE_FRAC * np.abs(a["cont"])
    show = ("dust2", "dust_ratio", "frac_nodust", "logmass", "logzsol")
    lines = [f"start: pocoMC best point of {args.state}, dust_index prior TopHat{WIDE}"]
    for n, f in fits.items():
        g = ", ".join(f"{k} {f['theta'][labels.index(k)]:.3f}" for k in show)
        lines.append(f"n {n:5.1f}: lnL {f['lnl']:.2f}, lnp {f['lnp']:.2f}, {g}, "
                     f"response {f['resp'].min():.4f}-{f['resp'].max():.4f}")  # fmt: skip
    lines += [
        f"d chi2 (n {SLOPES[1]} minus n {SLOPES[0]}): {-2 * (b['lnl'] - a['lnl']):.1f} from ln L, "
        f"{np.sum((chi_b - chi_a)[o.mask]):.1f} from masked pixels",
        f"{'rest bin':>11} {'npix c':>6} {'dchi2 c':>8} {'npix l':>6} {'dchi2 l':>8} "
        f"{'c0/c-4':>8} {'d/m -4':>8} {'d/m 0':>8}",
    ]
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
        sel = o.mask & (rest >= lo) & (rest < hi)
        if not sel.any():
            continue
        c, ln = sel & ~is_line, sel & is_line
        ratio = np.median(b["cont"][c] / a["cont"][c]) if c.any() else np.nan
        dm_a = np.median(o.flux[c] / a["spec"][c]) if c.any() else np.nan
        dm_b = np.median(o.flux[c] / b["spec"][c]) if c.any() else np.nan
        dc, dl = np.sum(chi_b[c] - chi_a[c]), np.sum(chi_b[ln] - chi_a[ln])
        rows.append((0.5 * (lo + hi), dc, dl, ratio, dm_a, dm_b))
        lines.append(f"{lo:5.0f}-{hi:5.0f} {int(c.sum()):6d} {dc:8.1f} {int(ln.sum()):6d} "
                     f"{dl:8.1f} {ratio:8.4f} {dm_a:8.4f} {dm_b:8.4f}")  # fmt: skip
    c, ln = o.mask & ~is_line, o.mask & is_line
    lines.append(f"{'all':>11} {int(c.sum()):6d} {np.sum((chi_b - chi_a)[c]):8.1f} "
                 f"{int(ln.sum()):6d} {np.sum((chi_b - chi_a)[ln]):8.1f}")  # fmt: skip

    out = args.state.with_name("ridge_shape.txt")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    np.savez(
        out.with_suffix(".npz"),
        rest=rest, flux=o.flux, unc=o.uncertainty, mask=o.mask, is_line=is_line,
        **{f"{k}_n{n:g}": f[k] for n, f in fits.items() for k in ("theta", "spec", "cont", "resp")},
    )  # fmt: skip

    r = np.array(rows)
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].step(r[:, 0], r[:, 3], where="mid", color="k")
    ax[0].set_ylabel("continuum n=0 / n=-4")
    ax[1].step(r[:, 0], r[:, 4], where="mid", label="n = -4")
    ax[1].step(r[:, 0], r[:, 5], where="mid", label="n = 0")
    ax[1].axhline(1, color="0.6", lw=0.8)
    ax[1].set_ylabel("median data / model")
    ax[1].legend()
    ax[2].bar(r[:, 0], r[:, 1], width=180, label="continuum")
    ax[2].bar(r[:, 0], r[:, 2], width=90, label="lines")
    ax[2].plot(r[:, 0], np.cumsum(r[:, 1] + r[:, 2]), color="k", label="cumulative")
    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].set_ylabel("chi2(n=0) - chi2(n=-4)")
    ax[2].set_xlabel("rest wavelength [A]")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
