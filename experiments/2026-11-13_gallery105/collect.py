"""Merge Nikhil's visual labels into gallery105.tsv and cross-check them with the measured columns.

QUESTION: How do the visual classes of the 105 compare with the FastSpecFit line class, HdA and the
Goto-like flag of seed.py?
HYPOTHESIS: none registered.
INPUTS: visual_labels.txt (Nikhil 2026-11-13, by eye from the gallery105 figures, verbatim, one line
per figure number); results/2026-11-13_gallery105/gallery105.tsv (seed.py, run that first).
METHOD: visual_text keeps each label verbatim. visual_class is the first of these keywords in the
text: psb, agn, quiescent, emission, data error (bad_data); unsure when none appears. visual_sure
is False when the text has a "?". visual_image collects blend ("two galaxies"), crowded ("crowd" or
"star") and diffuse. These rules are mechanical; edit visual_class by hand where they misread.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_gallery105/collect.py``.
RESULT: pending.
FIGURES: none (tables in results/2026-11-13_gallery105/collect.txt).
"""

import re
from pathlib import Path

import pandas as pd

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-13_gallery105"
LABELS = Path(__file__).with_name("visual_labels.txt")
CLASS_KEYS = {
    "psb": "psb",
    "agn": "agn",
    "quiescent": "quiescent",
    "emission": "emission_line",
    "data error": "bad_data",
}
IMAGE_KEYS = {"two galaxies": "blend", "crowd": "crowded", "star": "crowded", "diffuse": "diffuse"}


def parse(text):
    """Return the visual class, sure flag and image flags of one label."""
    low = text.lower()
    hits = [(low.find(k), v) for k, v in CLASS_KEYS.items() if k in low]
    cls = min(hits)[1] if hits else "unsure"
    image = sorted({v for k, v in IMAGE_KEYS.items() if k in low})
    return cls, "?" not in text, ";".join(image)


def main():
    """Merge the labels and write the cross-check tables."""
    labels = {}
    for line in LABELS.read_text().splitlines():
        m = re.match(r"\s*(\d+)\.\s*(.*?)\s*$", line)
        labels[int(m.group(1))] = m.group(2)
    assert sorted(labels) == list(range(1, 106)), len(labels)
    d = pd.read_csv(OUT / "gallery105.tsv", sep="\t", keep_default_na=False, na_values=[""])
    d["visual_text"] = [labels[int(n)] for n in d["nn"]]
    parsed = [parse(t) for t in d["visual_text"]]
    d["visual_class"] = [p[0] for p in parsed]
    d["visual_sure"] = [p[1] for p in parsed]
    d["visual_image"] = [p[2] for p in parsed]
    d["category"] = d["visual_class"]
    d.to_csv(OUT / "gallery105.tsv", sep="\t", index=False)

    d["hda_gt5"] = d["hda"] > 5
    lines = [
        f"visual classes: {d['visual_class'].value_counts().to_dict()}",
        f"sure (no '?'): {d['visual_sure'].sum()} of {len(d)}",
        "",
        "visual_class x FastSpecFit class",
        pd.crosstab(d["visual_class"], d["fs_class"], margins=True).to_string(),
        "",
        "visual_class x HdA > 5 A and goto_like",
        pd.crosstab(d["visual_class"], [d["hda_gt5"], d["goto_like"]], margins=True).to_string(),
        "",
        "visual psb: per target",
        d[d["visual_class"] == "psb"][
            ["nn", "z", "hda", "hda_err", "fs_ew_oii", "fs_ew_ha", "fs_dn4000", "goto_like"]
        ].to_string(index=False),
        "",
        "goto_like or HdA - 2 err > 5, not visual psb",
        d[(d["goto_like"] | (d["hda"] - 2 * d["hda_err"] > 5)) & (d["visual_class"] != "psb")][
            ["nn", "visual_text", "hda", "hda_err", "fs_ew_oii", "fs_class"]
        ].to_string(index=False),
        "",
        "visual agn: FastSpecFit and EmFit",
        d[d["visual_class"] == "agn"][
            ["nn", "visual_text", "fs_class", "fs_broad", "fs_bpt", "fs_whan", "em_prob_broad"]
        ].to_string(index=False),
        "",
        "FastSpecFit AGN not visual agn",
        d[(d["fs_class"] == "AGN") & (d["visual_class"] != "agn")][
            ["nn", "visual_text", "fs_broad", "fs_bpt", "fs_whan"]
        ].to_string(index=False),
    ]
    text = "\n".join(lines)
    (OUT / "collect.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
