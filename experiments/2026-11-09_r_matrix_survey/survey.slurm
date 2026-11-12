#!/bin/bash
# survey.py run on one skx-dev node: merged DR1 coadd R and row sigma for the targets in
# targets.txt. Submit from the repository root on stampede3:
#   sbatch experiments/2026-11-09_r_matrix_survey/survey.slurm
#SBATCH -J rsurvey
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/rsurvey_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-11-09_r_matrix_survey
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), start $(date)"
sha256sum $EXP/survey.py $EXP/survey.slurm $EXP/targets.txt
$PY $EXP/survey.py run
status=$?
echo "python exit $status, end $(date)"
exit $status
