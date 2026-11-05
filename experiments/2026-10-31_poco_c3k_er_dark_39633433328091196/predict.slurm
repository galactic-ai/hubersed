#!/bin/bash
# Spectrum of the max-likelihood pocoMC sample (predict.py), one process on one skx-dev node.
# Env as test.slurm. Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/predict.slurm
#SBATCH -J pocopred1031
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/pocopred1031_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $EXP/predict.py $EXP/fit.py $EXP/predict.slurm
$PY $EXP/predict.py
echo "exit $?, end $(date)"
