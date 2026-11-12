"""Compare the v2-mock flows with the old Cue-mock flows that gave the 310, per encoder.

QUESTION: Why does the 15-latent flow flag more DESI galaxies with the v2 mocks (2,006 vs 788) while
6- and 10-latent flag fewer, and which galaxies does only the 15-latent flow flag?
HYPOTHESIS: none stated before the run.
INPUTS: results/2026-11-09_mocks_v2_outliers/flows/ (new, post_mac.sh), results/
noised_cue_meanzero_wide_flow/ (old, the 310 run), data/latents/ (DESI and v2 mock latents),
data/fastspec-iron-sv3-bright.fits, data/dataset_z_snr.npz, results/prescreened_sample_v1.npz.
SEED: none (flows are loaded, not retrained).
COMMAND: ``uv run python experiments/2026-11-09_mocks_v2_outliers/compare_flows.py``.
RESULT: 6/10/15 flag 1,894/733/2,006 (old 2,480/1,779/788). The new 15-latent flags (1,771 with
fastspec rows) are low-mass (log M* 8.6), low-z (0.06) star-forming dwarfs with moderate EW(Ha) 45 A
and narrow lines: EmFit Ha sigma median 26 km/s (all scored: 90). The v2 mock eline_sigma prior,
TopHat(10, 450), has median 230 and only 5% below 32 km/s, so narrow-line dwarfs sit in thin mock
space; 5% of DESI fall below log p -20.3 vs ~0.3% of mocks. Assumption: EmFit sigma (flag 0) stands
for the line width the encoder sees. cont10: 3,105 flagged (old 2,667); 61% of DESI below the
v2 mock p5 log p (old mocks 28%; full encoders 17-23%), so typical DESI continua sit off the mock
bulk. Their 10 nearest mocks have dust2 1.35 vs prior median 0.68 (+1.1 sd), dust_index -1.85 vs
-1.05 (-1.0 sd), logzsol -0.62 vs -1.05 (+0.5 sd): DESI is redder and more metal-rich than most
mocks (dust2 ClippedNormal(0.3, 1, 0, 4), logzsol U(-2.5, 0.5)). Not checked: whether a
continuum-shape mismatch (C3K ER, smoothing outside rest 3320-9000 A) adds to this.
Flag rule (6&10&15)|(cont10&cont15) after the prescreen: new 205 (full 100, cont 111, both 6) vs
old 345 (the 310 after SIMBAD); 93 in both, 112 new only; 76 of the 310 still flagged. SIMBAD not
yet run on the new set.
FIGURES: none (tables in compare_flows.txt).
"""

# %%
import h5py
import numpy as np
import torch
from astropy.io import fits
from scipy.stats import spearmanr

from hubersed.detect.flow_model import build_flow
from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
NEW = RES / "2026-11-09_mocks_v2_outliers" / "flows"
OLD = RES / "noised_cue_meanzero_wide_flow"
TAGS = ("6", "10", "15", "cont10", "cont15")
Q = (0.1, 1, 5, 50, 95)
lines = []


def load_out(d, t):
    """Return the flow output dict of tag t in folder d, or None."""
    p = d / f"desi_outliers_flow_nsf_{t}latent_snr3.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def pct(x):
    """Return the Q percentiles of x as a short string."""
    return " ".join(f"p{q:g} {v:7.1f}" for q, v in zip(Q, np.percentile(x, Q), strict=True))


# %% per encoder: counts, thresholds, DESI log p
out = {}
for t in TAGS:
    o, n = load_out(OLD, t), load_out(NEW, t)
    if n is None:
        continue
    out[t] = (o, n)
    assert np.array_equal(o["desi_target_ids"], n["desi_target_ids"]), t
    for name, d in (("old", o), ("new", n)):
        lp = d["log_p_desi"]
        lines.append(
            f"{t:>6} {name}: flagged {len(d['outlier_target_ids']):5d}, threshold "
            f"{d['threshold']:7.2f}, c2st {d['c2st']:.3f}, DESI log p {pct(lp)}"
        )

# %% new 15-latent: mock log p from the saved flow, against DESI
f = torch.load(NEW / "flow_nsf_15latent.pt", weights_only=False)
nde = build_flow(f["method"], f["dim"], f["hidden"], f["num_transforms"], f["num_bins"])
nde.load_state_dict(f["state_dict"])
nde.eval()
mock, _, _ = load_latents(PATHS["ROOT"] / f["mock_file"])
xs = ((mock - f["scaler_mean"]) / f["scaler_scale"]).astype(np.float32)
with torch.no_grad():
    lp_mock = np.concatenate(
        [
            nde.log_prob(torch.from_numpy(xs[i : i + 65536])).numpy()
            for i in range(0, len(xs), 65536)
        ]
    )
n15 = out["15"][1]
lines.append(f"15 new: mock log p {pct(lp_mock)}")
lines.append(
    f"15 new: recomputed 0.1% mock quantile {np.quantile(lp_mock, 0.001):.2f} "
    f"(saved {n15['threshold']:.2f})"
)


# %% every new flow: median log p of mocks vs DESI (a shift means DESI sits off the mock bulk)
def mock_logp(t, folder=NEW):
    """Return log p of a flow's own mock latents under the saved flow of tag t in folder."""
    g = torch.load(folder / f"flow_nsf_{t}latent.pt", weights_only=False, map_location="cpu")
    m = build_flow(g["method"], g["dim"], g["hidden"], g["num_transforms"], g["num_bins"])
    m.load_state_dict(g["state_dict"])
    m.eval()
    x, _, _ = load_latents(PATHS["ROOT"] / g["mock_file"])
    x = ((x - g["scaler_mean"]) / g["scaler_scale"]).astype(np.float32)
    with torch.no_grad():
        return np.concatenate(
            [
                m.log_prob(torch.from_numpy(x[i : i + 65536])).numpy()
                for i in range(0, len(x), 65536)
            ]
        )


for t in out:
    lm, ld = mock_logp(t), out[t][1]["log_p_desi"]
    lines.append(
        f"{t:>6} new: mock log p p50 {np.median(lm):6.1f}, DESI p50 {np.median(ld):6.1f}, "
        f"DESI below the mock p5 {np.mean(ld < np.percentile(lm, 5)):.3f}"
    )

# The old cont10 mock latents are still on the Mac (noised_cue_meanzero_wide), so cont10 has a
# baseline; the old 6/10/15 mock latents are on TACC only (mac_cleanup_2026-11-10.md).
lm, ld = mock_logp("cont10", OLD), out["cont10"][0]["log_p_desi"]
lines.append(
    f"cont10 old: mock log p p50 {np.median(lm):6.1f}, DESI p50 {np.median(ld):6.1f}, "
    f"DESI below the mock p5 {np.mean(ld < np.percentile(lm, 5)):.3f}"
)

# %% cont10: which mock parameters the nearest mocks of DESI galaxies have, vs the whole prior
from sklearn.neighbors import KDTree  # noqa: E402

g = torch.load(NEW / "flow_nsf_cont10latent.pt", weights_only=False, map_location="cpu")
xm, mid, _ = load_latents(PATHS["ROOT"] / g["mock_file"])
xd, dtid, _ = load_latents(PATHS["ROOT"] / out["cont10"][1]["desi_file"])
xm = (xm - g["scaler_mean"]) / g["scaler_scale"]
xd = (xd - g["scaler_mean"]) / g["scaler_scale"]
pick = np.random.default_rng(0).choice(len(xd), 20000, replace=False)
_, nn = KDTree(xm).query(xd[pick], k=10)
with h5py.File(PATHS["DATA"] / "prospector_model" / "mocks_v2_500000.h5", "r") as hm:
    labels = [x.decode() if isinstance(x, bytes) else x for x in hm.attrs["theta_labels"]]
    theta = hm["priors/theta"][:][np.asarray(mid, np.int64)]
    zmock = hm["priors/redshifts"][:][np.asarray(mid, np.int64)]
cols = np.column_stack([theta, zmock])
names = [*labels, "z"]
matched = cols[nn.ravel()]
shift = (np.median(matched, 0) - np.median(cols, 0)) / np.std(cols, 0)
for j in np.argsort(-np.abs(shift))[:10]:
    lines.append(
        f"cont10 kNN: {names[j]:>16} prior median {np.median(cols[:, j]):8.3f}, DESI-matched "
        f"{np.median(matched[:, j]):8.3f}, shift {shift[j]:+.2f} sd"
    )

# %% cont10 kNN dust vs fastspecfit AV of the same galaxies (does dust stand in for shape?)
fs = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)["FASTSPEC"].data
fsrow = {int(t): i for i, t in enumerate(fs["TARGETID"])}
j2 = names.index("dust2")
av_knn = 1.086 * np.median(cols[nn, j2], axis=1)  # diffuse A_V of the 10 nearest mocks
ptid = np.asarray(dtid, np.int64)[pick]
ok = np.array([int(t) in fsrow for t in ptid])
av_fs = np.full(len(ptid), np.nan)
av_fs[ok] = np.asarray(fs["AV"], float)[[fsrow[int(t)] for t in ptid[ok]]]
good = ok & np.isfinite(av_fs)
d = av_knn[good] - av_fs[good]
lines.append(
    f"cont10 kNN A_V (1.086 dust2) median {np.median(av_knn[good]):.2f} vs fastspecfit AV "
    f"{np.median(av_fs[good]):.2f} on {good.sum()}; difference median {np.median(d):+.2f} "
    f"(p16 {np.percentile(d, 16):+.2f}, p84 {np.percentile(d, 84):+.2f}); Spearman "
    f"{spearmanr(av_knn[good], av_fs[good]).statistic:.2f}"
)

# %% who only the 15-latent flow flags (new), and how they look
tid = np.asarray(n15["desi_target_ids"], np.int64)
flag = {t: {int(x) for x in out[t][1]["outlier_target_ids"]} for t in ("6", "10", "15")}
old15 = {int(x) for x in out["15"][0]["outlier_target_ids"]}
only15 = flag["15"] - flag["6"] - flag["10"]
lines.append(
    f"new 15 flagged {len(flag['15'])}: only-15 {len(only15)}, also old-15 "
    f"{len(flag['15'] & old15)}, only-15 and old-15 {len(only15 & old15)}"
)

h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)["FASTSPEC"].data
row = {int(t): i for i, t in enumerate(h["TARGETID"])}
zs = np.load(PATHS["DATA"] / "dataset_z_snr.npz")
zrow = {int(t): i for i, t in enumerate(zs["tid"])}
keep = set(np.load(RES / "prescreened_sample_v1.npz")["keep"].tolist())


def describe(name, ids):
    """Append medians (p16, p84) of z, S/N, log M*, Dn4000, Halpha EW and the parent fraction."""
    ids = [i for i in ids if i in row and i in zrow]
    r = np.array([row[i] for i in ids])
    zr = np.array([zrow[i] for i in ids])
    cols = {
        "z": zs["z"][zr],
        "S/N": zs["snr"][zr],
        "logM*": np.asarray(h["LOGMSTAR"], float)[r],
        "Dn4000": np.asarray(h["DN4000"], float)[r],
        "EW(Ha)": np.asarray(h["HALPHA_EW"], float)[r],
    }
    s = " ".join(
        f"{k} {np.nanmedian(v):.2f} ({np.nanpercentile(v, 16):.2f}, {np.nanpercentile(v, 84):.2f})"
        for k, v in cols.items()
    )
    lines.append(
        f"{name:>16} N {len(ids):6d} in parent {np.mean([i in keep for i in ids]):.2f}  {s}"
    )


describe("all scored", tid.tolist())
describe("new 15 flagged", flag["15"])
describe("only-15 (new)", only15)
describe("old 15 flagged", old15)
describe("new 6&10&15", flag["6"] & flag["10"] & flag["15"])

# %% the cont10 flow: who the new flow flags, compared with the old one
if "cont10" in out:
    c_new = {int(x) for x in out["cont10"][1]["outlier_target_ids"]}
    c_old = {int(x) for x in out["cont10"][0]["outlier_target_ids"]}
    lines.append(
        f"cont10: new {len(c_new)}, old {len(c_old)}, both {len(c_new & c_old)}, "
        f"new only {len(c_new - c_old)}, old only {len(c_old - c_new)}"
    )
    describe("cont10 new", c_new)
    describe("cont10 new only", c_new - c_old)
    describe("cont10 old only", c_old - c_new)
    describe("cont10 both", c_new & c_old)

# %% line widths: EmFit Halpha narrow sigma of the groups vs the v2 mock eline_sigma prior draws
from astropy.table import Table  # noqa: E402

em = Table.read(RES / "2026-11-10_line_width_split" / "emfit_widths_sv3_bright.fits")
emrow = {int(t): i for i, t in enumerate(em["targetid"])}
sig = np.ma.filled(np.ma.asarray(em["ha_n_sigma"], float), np.nan)
sflag = np.ma.filled(np.ma.asarray(em["ha_n_sigma_flag"], float), np.nan)


def width(name, ids):
    """Append the EmFit narrow Halpha sigma (flag 0) quantiles of a group."""
    v = np.array([sig[emrow[i]] for i in ids if i in emrow and sflag[emrow[i]] == 0])
    lines.append(f"{name:>16} EmFit Ha narrow sigma N {v.size:6d} {pct(v)}")


width("all scored", tid.tolist())
width("new 15 flagged", flag["15"])
width("only-15 (new)", only15)
width("new 6&10&15", flag["6"] & flag["10"] & flag["15"])
with h5py.File(PATHS["DATA"] / "prospector_model" / "mocks_v2_500000.h5", "r") as hm:
    labels = [x.decode() if isinstance(x, bytes) else x for x in hm.attrs["theta_labels"]]
    es = hm["priors/theta"][:, labels.index("eline_sigma")]
lines.append(f"{'v2 mock prior':>16} eline_sigma N {es.size} {pct(es)}")

# %% the flag rule (6 & 10 & 15) | (cont10 & cont15), before and after the prescreen, vs the 310
import pandas as pd  # noqa: E402

t310 = set(pd.read_csv(RES / "map_fits_outliers310_v2" / "classification_v2.csv")["target_id"])
for name, k in (("old", 0), ("new", 1)):
    fl = {t: {int(x) for x in out[t][k]["outlier_target_ids"]} for t in TAGS}
    full = fl["6"] & fl["10"] & fl["15"]
    cont = fl["cont10"] & fl["cont15"]
    fk, ck = full & keep, cont & keep
    lines.append(
        f"rule {name}: full {len(full)} cont {len(cont)} | after prescreen full {len(fk)} "
        f"cont {len(ck)} both {len(fk & ck)} union {len(fk | ck)} | in the 310 {len((fk | ck) & t310)}"
    )
    if k == 1:
        new_union = fk | ck
    else:
        old_union = fk | ck
lines.append(
    f"rule: after-prescreen union old {len(old_union)}, new {len(new_union)}, "
    f"in both {len(old_union & new_union)}, new only {len(new_union - old_union)}"
)

text = "\n".join(lines)
print(text)
(RES / "2026-11-09_mocks_v2_outliers" / "compare_flows.txt").write_text(text + "\n")
