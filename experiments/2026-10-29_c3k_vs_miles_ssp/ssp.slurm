#!/bin/bash
# Dust-free SSP grids from the C3K_HR (project venv) and MILES (miles_build) FSPS builds, with the
# same SPS_HOME as the 10-08 runs (run.sh). One process at a time, a few minutes each.
# Submit from the repository root on ls6:
#   sbatch experiments/2026-10-29_c3k_vs_miles_ssp/ssp.slurm
#SBATCH -J ssp1029
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/ssp1029_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_c3k_vs_miles_ssp
cd $REPO
export SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $EXP/ssp.py $EXP/ssp.slurm
uv run --no-sync python $EXP/ssp.py --name c3k_hr
PYTHONPATH=/scratch/11006/nikhilgaruda/hubersed_tmp/miles_build/fsps_site \
    uv run --no-sync python $EXP/ssp.py --name miles
echo "end $(date)"
