#!/bin/bash
# For every target of targets.txt: the DR1 resolution matrix (resolution_matrix.py), then fit.py check
# in one process (model build and lnlike at prior draws). After stage.sh. Env as the 10-31 test.slurm.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-31_poco_robust6/prep.slurm
#SBATCH -J prep1031
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/prep1031_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-31_poco_robust6
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $EXP/*.py $EXP/*.slurm $EXP/targets.txt experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/fit.py
while read -r tid hp chunk; do
    echo "== $tid $(date +%T)"
    taskset -c 0 $PY $EXP/resolution_matrix.py --tid $tid || exit 1
    taskset -c 0 $PY $EXP/fit.py check --tid $tid --n-calls 3 || exit 1
done < $EXP/targets.txt
echo "end $(date)"
