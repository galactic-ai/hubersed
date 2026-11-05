#!/bin/bash
# skx-dev job of this experiment: prepare.py (spender mask on SV3 and main, main resolution matrix),
# then fit.py check (setup and timing of warm calls on the main data).
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-29_continuum_main_survey_map/check.slurm
#SBATCH -J main1029chk
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/main1029chk_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_continuum_main_survey_map
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site:$REPO/experiments/2026-10-09_nautilus_c3k_er_broadline
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), start $(date)"
sha256sum $EXP/prepare.py $EXP/fit.py $EXP/check.slurm experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
echo "--- prepare $(date +%T)"
$PY $EXP/prepare.py
echo "--- check $(date +%T)"
taskset -c 0 $PY $EXP/fit.py check
echo "end $(date)"
