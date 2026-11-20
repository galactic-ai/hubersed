#!/bin/bash
# Fetch the iron cframes b1 and r1 of exposure 87113 (target 39627770174637084) to Corral, then copy
# them, its fluxcalib and its sframe to /scratch, as 2026-11-14_poco_psb_sfh/stage_recoadd.sh. Run on
# the login node: bash experiments/2026-11-16_exposure_line_centroids/stage.sh
set -euo pipefail
URL=https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/exposures/20210503/00087113
C=/corral/utexas/AST25023/nikhilgaruda/hubersed/data/desi_iron_exposures/20210503/00087113
S=/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_exposures/20210503/00087113
for f in cframe-b1-00087113.fits.gz cframe-r1-00087113.fits.gz; do
    [ -s "$C/$f" ] || { curl -sSf -o "$C/$f.part" "$URL/$f" && mv "$C/$f.part" "$C/$f"; }
done
mkdir -p "$S"
for p in cframe fluxcalib sframe; do
    for arm in b r; do
        rsync -a "$C/$p-${arm}1-00087113.fits.gz" "$S/"
    done
done
ls -la "$S" | grep -E "(cframe|fluxcalib|sframe)-[br]1"
echo "STAGED $(date)"
