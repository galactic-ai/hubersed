#!/bin/bash
# Fetch the iron spectra file of each healpix in targets.txt to /scratch, as
# 2026-10-31_screen_all/sky_stage.sh. Run on the stampede3 login node from the repository root:
#   bash experiments/2026-10-31_boss_39633325949715572/stage.sh
set -euo pipefail
HERE=experiments/2026-10-31_boss_39633325949715572
BASE=https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/sv3/bright
S=/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_spectra
mkdir -p "$S"
while read -r tid hp; do
    f="$S/spectra-sv3-bright-$hp.fits.gz"
    [ -s "$f" ] || { curl -sSf -o "$f.part" "$BASE/$((hp / 100))/$hp/$(basename "$f")" && mv "$f.part" "$f"; }
    echo "$tid $(ls -la "$f" | awk '{print $5}') $f"
done < "$HERE/targets.txt"
echo STAGED
