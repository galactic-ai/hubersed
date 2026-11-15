#!/bin/bash
# Fetch the iron spectra files of urls.txt (healpix 25944, 9841 and 11231, gallery105 figures 05,
# 67 and 99) to /scratch, skipping files already there. Run on the stampede3 login node from the
# repository root (or a copy of it):
#   bash experiments/2026-11-13_exposures_gallery105/stage.sh
set -euo pipefail
HERE=experiments/2026-11-13_exposures_gallery105
S=/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_spectra
mkdir -p "$S"
while read -r url; do
    f="$S/$(basename "$url")"
    if [ ! -s "$f" ]; then
        curl -sSf -o "$f.part" "$url" && mv "$f.part" "$f"
    fi
    echo "$(ls -la "$f" | awk '{print $5}') $f"
done < "$HERE/urls.txt"
echo STAGED
