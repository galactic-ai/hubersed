#!/bin/bash
# Copy the Parquet datasets from /scratch to Corral after convert.slurm, with their READMEs.
# The READMEs are copied to $D/<name>/README.md beforehand. Run on a login node:
#   nohup bash experiments/2026-11-20_corral_parquet/publish.sh > <log> 2>&1 &
set -euo pipefail
D=/scratch/11006/nikhilgaruda/hubersed_data/datasets
C=/corral/utexas/AST25023/nikhilgaruda/hubersed/data/datasets
mkdir -p $C
for n in desi_sv3_bright_v1 mocks_cuev2_noised mocks_cue_noised; do
    rsync -a $D/$n/ $C/$n/
    echo "$n: $(ls $C/$n/data | wc -l) Parquet files, $(du -sh $C/$n | cut -f1), $(date)"
done
