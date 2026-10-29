#!/bin/bash
# Test of this experiment on one stampede3 skx-dev node, as
# 2026-10-28_continuum_c3k_er_rmatrix_map/test.slurm: save the resolution matrix, check the setup
# and time warm calls (fit.py check), then a short start-0 MAP and its collect step with tag
# _test. One process at a time. hr_maxl.npz (hypers.py, made on the Mac) must be in the results
# folder first.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-29_continuum_39633433328091196_map/test.slurm
#SBATCH -J c4test1029
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/c4test1029_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_continuum_39633433328091196_map
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
TEST_MAXFEV=${TEST_MAXFEV:-2000}
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), start $(date)"
free -g
sha256sum $EXP/fit.py $EXP/resolution_matrix.py $EXP/test.slurm \
    experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
test -f results/2026-10-29_continuum_39633433328091196_map/hr_maxl.npz

echo "--- resolution matrix $(date +%T)"
$PY $EXP/resolution_matrix.py
echo "--- check $(date +%T)"
/usr/bin/time -v taskset -c 0 $PY $EXP/fit.py check 2>&1 | grep -v "^\s*\(Page\|Socket\|Signals\|Swaps\|File\|Average\|Voluntary\|Involuntary\|Exit\|Minor\|Major\)"
echo "--- short MAP, start 0, maxfev $TEST_MAXFEV $(date +%T)"
timeout 3600 taskset -c 0 $PY $EXP/fit.py fit --start 0 --maxfev $TEST_MAXFEV --tag _test
echo "--- collect $(date +%T)"
taskset -c 0 $PY $EXP/fit.py collect --n-starts 1 --tag _test
echo "end $(date)"
