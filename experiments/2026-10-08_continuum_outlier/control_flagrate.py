"""How often the continuum flows flag DESI galaxies like 39627757533007793.

Analogues share its Dn4000, stellar mass and redshift in the FastSpecFit catalogue. If they are
rarely flagged, being massive and quiescent does not explain the flag. The flows are the
corrected [OII] runs in results/wide_flow_corrected/, which exist only on the Mac. The catalogue
is the same file on the Mac and on ls6 (md5 ae54e69f, checked 2026-10-08).

Run from the repository root with
``uv run python experiments/2026-10-08_continuum_outlier/control_flagrate.py``.
"""

import numpy as np
import torch
from astropy.io import fits

from hubersed.paths import PATHS

TID = 39627757533007793
FLOWS = ["cont10", "cont15"]
COLS = ["TARGETID", "Z", "LOGMSTAR", "DN4000", "VDISP", "RCHI2", "RCHI2_CONT", "RCHI2_LINE", "AV"]


def load_vac():
    """Return the FastSpecFit columns in COLS as a dict of arrays."""
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True) as h:
        d = h["FASTSPEC"].data
        return {c: np.asarray(d[c]) for c in COLS}


def load_flow(name):
    """Return target IDs, log p and the 0.1% mock threshold of one continuum flow."""
    f = PATHS["RESULTS"] / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{name}latent_snr3.pt"
    d = torch.load(f, weights_only=False)
    thr = [float(np.asarray(v)) for k, v in d.items() if "thresh" in k.lower() and "held" not in k]
    assert len(thr) == 1, list(d.keys())
    return np.asarray(d["desi_target_ids"], np.int64), np.asarray(d["log_p_desi"], float), thr[0]


def main():
    vac = load_vac()
    row = {t: i for i, t in enumerate(vac["TARGETID"])}
    g = row[TID]
    print("galaxy:", {c: float(vac[c][g]) for c in COLS[1:]})
    for name in FLOWS:
        tid, lp, thr = load_flow(name)
        keep = np.array([t in row for t in tid])
        tid, lp = tid[keep], lp[keep]
        v = {c: vac[c][[row[t] for t in tid]] for c in COLS[1:]}
        me = lp[tid == TID][0]
        print(
            f"\n{name}: threshold {thr:.2f}, galaxy log p {me:.2f}, flagged {me <= thr}, "
            f"rank {100 * np.mean(lp <= me):.2f}% of {tid.size}"
        )
        twin = (
            (v["DN4000"] >= 1.80)
            & (v["DN4000"] <= 1.96)
            & (v["LOGMSTAR"] >= 10.9)
            & (v["LOGMSTAR"] <= 11.3)
            & (v["Z"] >= 0.14)
            & (v["Z"] <= 0.20)
        )
        sets = {
            "analogues": twin,
            "analogues, VDISP 280-330": twin & (v["VDISP"] >= 280) & (v["VDISP"] <= 330),
            "Dn4000 >= 1.8, logM >= 10.8": (v["DN4000"] >= 1.8) & (v["LOGMSTAR"] >= 10.8),
            "Dn4000 < 1.5": v["DN4000"] < 1.5,
        }
        for label, s in sets.items():
            print(
                f"  {label}: n {s.sum()}, flagged {(lp[s] <= thr).sum()} "
                f"({100 * np.mean(lp[s] <= thr):.2f}%), median log p {np.median(lp[s]):.2f}, "
                f"galaxy is lowest: {me <= lp[s].min()}"
            )
        if name == FLOWS[0]:
            print("  galaxy percentile among analogues:")
            for c in ["RCHI2", "RCHI2_CONT", "RCHI2_LINE", "AV"]:
                x = v[c][twin]
                print(
                    f"    {c} {vac[c][g]:.4f}: {100 * np.mean(x <= vac[c][g]):.1f}th, "
                    f"analogue median {np.median(x):.3f}"
                )


if __name__ == "__main__":
    main()
