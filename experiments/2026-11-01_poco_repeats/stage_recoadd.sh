#!/bin/bash
# Fetch the iron files of recoadd_urls.txt to Corral (skipping files already there), then copy them
# to /scratch, since compute nodes do not mount Corral. As 2026-10-30_recoadd_39633433328091196/stage.sh.
# Run on the stampede3 login node from the repository root:
#   nohup bash experiments/2026-11-01_poco_repeats/stage_recoadd.sh > /scratch/11006/nikhilgaruda/logs/stage_recoadd.log 2>&1 &
set -euo pipefail
HERE=experiments/2026-11-01_poco_repeats
C=/corral/utexas/AST25023/nikhilgaruda/hubersed/data
S=/scratch/11006/nikhilgaruda/hubersed_data
echo "start $(date)"
while read -r url dest; do
    if [ ! -s "$C/$dest" ]; then
        mkdir -p "$(dirname "$C/$dest")"
        curl -sSf -o "$C/$dest.part" "$url" && mv "$C/$dest.part" "$C/$dest"
    fi
    mkdir -p "$(dirname "$S/$dest")"
    rsync -a "$C/$dest" "$S/$dest"
done < "$HERE/recoadd_urls.txt"
while read -r tid hp night expid petal fiber tile; do
    echo "$tid $night $expid $(ls "$S/desi_iron_exposures/$night/$(printf %08d "$expid")" | grep -c -- "-$petal-\|[brz]$petal-") files of petal $petal"
done < "$HERE/recoadd_exposures.txt"
ls -la $S/desi_iron_spectra/ | grep -E "25977|9425|15343|10157|10199|11231"
echo "STAGED $(date)"
