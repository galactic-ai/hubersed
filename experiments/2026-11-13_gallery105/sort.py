"""Copy each reviewed gallery105 figure into a folder named after its review class, and summarise.

Classes come from review.tsv (one row per reviewed figure, written during the review with Nikhil).
Figures are copied, not moved, so seed.py still finds all 105 at the top level. The script only adds
copies; when a class changes it lists the old copy for removal by hand.
INPUTS: review.tsv; results/2026-11-13_gallery105/<NN>_<TARGETID>.png (analysis.py).
COMMAND: ``uv run --no-sync python experiments/2026-11-13_gallery105/sort.py``.
OUTPUT: results/2026-11-13_gallery105/<review_class>/<NN>_<TARGETID>.png and review_summary.txt (class by
branch, figure lists, unsure counts, PSB redshifts, EmFit Halpha sigma of full-branch emission-line galaxies).
RESULT: 2026-11-13, all 105 reviewed: emission_line 58, psb 24, agn 10, bad_data 5, quiescent 5,
star 2, composite 1. Every psb is in the cont branch; 46 of 49 full-branch targets are emission-line
galaxies with median EmFit Halpha sigma ~26 km/s.
"""

import shutil
from pathlib import Path

import pandas as pd

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-13_gallery105"
REVIEW = Path(__file__).with_name("review.tsv")


def main():
    """Rebuild the class folders from review.tsv."""
    rev = pd.read_csv(REVIEW, sep="\t")
    assert rev["nn"].is_unique, "duplicate figure numbers in review.tsv"
    figs = {int(p.name.split("_")[0]): p for p in OUT.glob("[0-9]*_*.png")}
    want = {figs[int(nn)].name: cls for nn, cls in zip(rev["nn"], rev["review_class"], strict=True)}
    for name, cls in want.items():
        (OUT / cls).mkdir(exist_ok=True)
        shutil.copy2(OUT / name, OUT / cls / name)
    stale = [p for p in OUT.glob("*/*.png") if p.parent.name != "cutouts"
             and want.get(p.name) != p.parent.name]  # fmt: skip
    for p in stale:
        print(f"stale copy, remove by hand: {p.relative_to(OUT)}")
    print(rev["review_class"].value_counts().to_string())
    tab = pd.read_csv(OUT / "gallery105.tsv", sep="\t")[["nn", "branch", "z", "em_ha_n_sigma"]]
    m = rev.merge(tab, on="nn")
    lines = [pd.crosstab(m["review_class"], m["branch"], margins=True).to_string(), ""]
    for cls, g in m.groupby("review_class"):
        lines.append(f"{cls} ({len(g)}, unsure {int((~g['review_sure']).sum())}): "
                     + " ".join(f"{n:02d}" for n in sorted(g["nn"])))  # fmt: skip
    psb = m[m["review_class"] == "psb"]["z"]
    el = m[(m["review_class"] == "emission_line") & (m["branch"] != "cont")][
        "em_ha_n_sigma"
    ].dropna()
    lines += [
        "",
        f"psb z: min {psb.min():.3f}, median {psb.median():.3f}, max {psb.max():.3f}",
        f"full-branch emission_line with EmFit Halpha sigma: N {el.size}, median {el.median():.1f} km/s, "
        f"<= 40 km/s {(el <= 40).mean():.2f}",
    ]
    (OUT / "review_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"sorted {len(rev)} of {len(figs)} figures; not yet reviewed: "
          f"{sorted(set(figs) - set(rev['nn'].astype(int)))[:10]} ...")  # fmt: skip


if __name__ == "__main__":
    main()
