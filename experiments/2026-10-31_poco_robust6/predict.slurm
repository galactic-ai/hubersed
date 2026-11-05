#!/bin/bash
# Max-likelihood spectra (predict.py) of the targets given as arguments, one process on one skx-dev node.
# Submit from the repository root on stampede3, after their prod jobs:
#   sbatch experiments/2026-10-31_poco_robust6/predict.slurm <TARGETID> [<TARGETID> ...]
#SBATCH -J pred1031
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/pred1031_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-31_poco_robust6
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $EXP/predict.py $EXP/fit.py $EXP/predict.slurm
for tid in "$@"; do
    taskset -c 0 $PY $EXP/predict.py --tid $tid || echo "predict failed for $tid"
done
echo "end $(date)"
