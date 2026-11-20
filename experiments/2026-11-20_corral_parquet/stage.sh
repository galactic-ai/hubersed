#!/bin/bash
# Copy the noisy mock pickles from Corral to /scratch, since compute nodes do not mount Corral.
# About 120 GB. Run on a login node from the repository root:
#   nohup bash experiments/2026-11-20_corral_parquet/stage.sh > <log> 2>&1 &
set -euo pipefail
C=/corral/utexas/AST25023/nikhilgaruda/hubersed/data/prospector_model
S=/scratch/11006/nikhilgaruda/hubersed_data/pickles
mkdir -p $S
for d in noised_cuev2 noised_cue_meanzero_wide; do
    rsync -a $C/$d/ $S/$d/
    echo "$d: $(ls $S/$d | wc -l) files, $(du -sh $S/$d | cut -f1), $(date)"
done
