#!/bin/bash
# Setup check of fit.py on one skx-dev node: SSP build of every slot, speclib free, pool sized from
# the commit room, Cue built after the fork in the parent and in each worker, lnL parent == workers,
# call time, one pocoMC batch over the whole pool and the commit room after it.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-11-07_emline_ladder_39627770174637084/check.slurm
#SBATCH -J step5check
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/step5check_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-11-07_emline_ladder_39627770174637084
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
cd $REPO
export PYTHONPATH=$FSPS/c3k_er_afe_v3/fsps_site:$REPO/experiments/2026-10-15_broadline_free_amplitude:$REPO/experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/fit.py $EXP/check.slurm
$PY -c "import prospect, sedpy; from prospect.fitting import ProcessLikelihood; print('prospect', prospect.__file__, 'sedpy', sedpy.__file__)"
(while sleep 120; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
MON=$!
timeout --kill-after=60 5000 $PY $EXP/fit.py check --pool 0
echo "python exit $?, end $(date)"
kill $MON
