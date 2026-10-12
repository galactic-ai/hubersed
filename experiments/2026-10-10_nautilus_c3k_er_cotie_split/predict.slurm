#!/bin/bash
# Run predict.py for the converged C/O tie fit on one development node, with the ER build and
# SPS_HOME of chain.slurm. Copied from 2026-10-09_nautilus_c3k_er_broadline/predict.slurm without
# the analysis step, since this experiment has no analysis.py yet. Submit from the repository root
# with
#   sbatch experiments/2026-10-10_nautilus_c3k_er_cotie_split/predict.slurm
#SBATCH -J cotie_predict
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
SCRIPTS=$REPO/experiments/2026-10-10_nautilus_c3k_er_cotie_split
FSPS_SITE=${FSPS_SITE:-/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site}
cd $REPO
export PYTHONPATH=$FSPS_SITE
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), scripts $SCRIPTS, fsps $FSPS_SITE, start $(date)"
$REPO/.venv/bin/python $SCRIPTS/predict.py --forbidden-broad shared
echo "end $(date)"
