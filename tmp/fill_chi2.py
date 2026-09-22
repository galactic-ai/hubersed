"""Refill the MAP chi2_red column of results/cont_outlier20_table.md from the pkl fits."""
import glob, pickle, re, pathlib

chi2 = {}
for f in glob.glob("results/cont_map_fits/*.pkl"):
    r = pickle.load(open(f, "rb"))
    chi2[str(r["target_id"])] = r["stats"]["chi2_red"]

p = pathlib.Path("results/cont_outlier20_table.md")
out = []
for line in p.read_text().splitlines():
    m = re.match(r"^\| *\d+ \| `(\d+)` \|", line)
    if m and m.group(1) in chi2:
        line = re.sub(r"\| *(—|[\d.]+) \|$", f"| {chi2[m.group(1)]:.2f} |", line)
    out.append(re.sub(r"^MAP fits complete: \d+/20\.", f"MAP fits complete: {len(chi2)}/20.", line))
p.write_text("\n".join(out) + "\n")
print(f"{len(chi2)}/20 filled")
