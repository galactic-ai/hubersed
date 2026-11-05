#!/bin/bash
# Run predict.py and then analysis.py for the converged ER fit on one development node, with the
# ER build and SPS_HOME of chain.slurm. SCRIPTS is the folder holding fit.py, predict.py and
# analysis.py (default this experiment). Submit from the repository root with
#   sbatch experiments/2026-10-09_nautilus_c3k_er_broadline/predict.slurm
#SBATCH -J er_predict
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
SCRIPTS=${SCRIPTS:-$REPO/experiments/2026-10-09_nautilus_c3k_er_broadline}
FSPS_SITE=${FSPS_SITE:-/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site}
cd $REPO
export PYTHONPATH=$FSPS_SITE
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), scripts $SCRIPTS, fsps $FSPS_SITE, start $(date)"
# ANALYSIS_ONLY=1 skips predict.py when its output already exists
if [ "${ANALYSIS_ONLY:-0}" != 1 ]; then
    $REPO/.venv/bin/python $SCRIPTS/predict.py --forbidden-broad shared || exit 1
    echo "predict done $(date)"
fi
$REPO/.venv/bin/python $SCRIPTS/analysis.py
echo "end $(date)"
