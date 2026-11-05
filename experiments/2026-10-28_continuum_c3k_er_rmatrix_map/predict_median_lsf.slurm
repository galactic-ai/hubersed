#!/bin/bash
# predict_median_lsf.py on one stampede3 skx-dev node, one process.
# Submit with this folder's predict_median_lsf.py and fit.py copied to $DIR:
#   sbatch --export=ALL,DIR=<folder> predict_median_lsf.slurm
#SBATCH -J er1028lsf
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/er1028lsf_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), start $(date)"
sha256sum $DIR/predict_median_lsf.py $DIR/fit.py
taskset -c 0 $PY $DIR/predict_median_lsf.py
echo "end $(date)"
