#!/bin/bash
# Fetch the iron spectra files of urls.txt to /scratch, skipping files already there. Run on the
# stampede3 login node from the repository root:
#   bash experiments/2026-11-05_exposures_39627770174637084/stage.sh
set -euo pipefail
HERE=experiments/2026-11-05_exposures_39627770174637084
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
