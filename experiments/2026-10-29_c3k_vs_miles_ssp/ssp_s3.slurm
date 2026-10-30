#!/bin/bash
# Dust-free SSP grids from the C3K ER (c3k_er_afe_v3) and MILES (miles_ls6) FSPS builds, both with
# SPS_HOME sps_home_er, which holds both libraries. One process at a time on one stampede3 skx-dev
# node.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-29_c3k_vs_miles_ssp/ssp_s3.slurm
#SBATCH -J ssp_s3
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/ssp_s3_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_c3k_vs_miles_ssp
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
BUILDS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
cd $REPO
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $EXP/ssp.py $EXP/ssp_s3.slurm
PYTHONPATH=$BUILDS/c3k_er_afe_v3/fsps_site taskset -c 0 $PY $EXP/ssp.py --name c3k_er
PYTHONPATH=$BUILDS/miles_ls6/fsps_site taskset -c 0 $PY $EXP/ssp.py --name miles
echo "end $(date)"
