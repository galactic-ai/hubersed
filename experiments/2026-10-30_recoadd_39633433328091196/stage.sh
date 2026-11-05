#!/bin/bash
# Fetch the iron files of iron_urls.txt to Corral (skipping files already there), then copy them to
# /scratch, since compute nodes do not mount Corral. Run on the stampede3 login node from the
# repository root: bash experiments/2026-10-30_recoadd_39633433328091196/stage.sh
set -euo pipefail
HERE=experiments/2026-10-30_recoadd_39633433328091196
C=/corral/utexas/AST25023/nikhilgaruda/hubersed/data
S=/scratch/11006/nikhilgaruda/hubersed_data
while read -r url dest; do
    if [ ! -s "$C/$dest" ]; then
        mkdir -p "$(dirname "$C/$dest")"
        curl -sSf -o "$C/$dest.part" "$url" && mv "$C/$dest.part" "$C/$dest"
    fi
    mkdir -p "$(dirname "$S/$dest")"
    rsync -a "$C/$dest" "$S/$dest"
done < "$HERE/iron_urls.txt"
ls -la "$S/desi_iron_spectra/spectra-sv3-bright-10231.fits.gz"
for e in 00087837 00087838 00087839; do
    echo "$e $(ls "$S/desi_iron_exposures/20210509/$e" | wc -l) files"
done
echo STAGED
