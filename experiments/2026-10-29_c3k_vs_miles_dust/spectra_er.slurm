#!/bin/bash
# C3K ER model spectra of 39633433328091196 at the MILES and C3K ER best points (spectra.py --lib
# er), on one stampede3 skx-dev node, one process. Env as
# 2026-10-29_continuum_39633433328091196_map/map.slurm.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-29_c3k_vs_miles_dust/spectra_er.slurm
#SBATCH -J specer1029
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/specer1029_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_c3k_vs_miles_dust
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $EXP/spectra.py $EXP/spectra_er.slurm
taskset -c 0 $PY $EXP/spectra.py --lib er
echo "end $(date)"
