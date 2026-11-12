#!/bin/bash
# fit.py predict on one skx-dev node: max-likelihood model, 100 posterior draws and chi of one
# finished step, with that step's FSPS build. Writes results/<exp>/<STEM>/predict.npz.
# Submit from the repository root on stampede3:
#   STEP=<3|4|5> sbatch --export=ALL experiments/2026-11-07_emline_ladder_39627770174637084/predict.slurm
#SBATCH -J ladpredict
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/ladpredict_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-11-07_emline_ladder_39627770174637084
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
STEP=${STEP:-5}
cd $REPO
R=/work/11006/nikhilgaruda/ls6/research
EXTRA=
case $STEP in  # FSPS build and SPS_HOME of each step, as 2026-11-01_poco_repeats/ladder.slurm
    1) BUILD=miles_ls6; SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5; EXTRA=--no-free ;;
    2) BUILD=c3k_hr_afe; SPS_HOME=$R/sps_home_hr_afe ;;
    3 | 4 | 5 | 6 | 7 | 8) BUILD=c3k_er_afe_v3; SPS_HOME=$R/sps_home_er ;;
    *) echo "unknown STEP $STEP"; exit 1 ;;
esac
export SPS_HOME PYTHONPATH=$FSPS/$BUILD/fsps_site:$REPO/experiments/2026-10-15_broadline_free_amplitude:$REPO/experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID step $STEP build $BUILD on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/fit.py $EXP/predict.slurm
$PY -c "import prospect, sedpy; from prospect.fitting import ProcessLikelihood; print('prospect', prospect.__file__, 'sedpy', sedpy.__file__)"
(while sleep 120; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
MON=$!
timeout --kill-after=60 3300 $PY $EXP/fit.py predict --step $STEP --pool 0 $EXTRA
status=$?
echo "python exit $status, end $(date)"
kill $MON
exit $status
