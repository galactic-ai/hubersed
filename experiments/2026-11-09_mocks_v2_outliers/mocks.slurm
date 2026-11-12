#!/bin/bash
# mocks.py on one node: ACTION=rbank merges the survey R matrices; ACTION=run writes mocks START to
# STOP with the C3K ER build (pilot: START=0 STOP=1000). Submit from the repository root on
# stampede3, for example:
#   ACTION=rbank sbatch --export=ALL -p skx -t 00:30:00 experiments/2026-11-09_mocks_v2_outliers/mocks.slurm
#   ACTION=run START=0 STOP=1000 sbatch --export=ALL experiments/2026-11-09_mocks_v2_outliers/mocks.slurm
# Full run 2026-11-09: 25 jobs of 20,000 (START=0,20000,...,480000), GIB=8, -t 02:30:00 on spr.
#SBATCH -J mocks2
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/mocks2_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-11-09_mocks_v2_outliers
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
BUILD=c3k_er_afe_v3
ACTION=${ACTION:-run}
START=${START:-0}
STOP=${STOP:-1000}
GIB=${GIB:-8}  # commit per worker in GiB, sizes the pool (pilot 11-09: ~5.6 measured)
cd $REPO
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export PYTHONPATH=$FSPS/$BUILD/fsps_site:$REPO/experiments/2026-10-15_broadline_free_amplitude:$REPO/experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID $ACTION $START-$STOP gib $GIB on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/mocks.py $EXP/mocks.slurm experiments/2026-11-07_emline_ladder_39627770174637084/fit.py
(while sleep 300; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
MON=$!
if [ "$ACTION" = rbank ]; then
    $PY $EXP/mocks.py rbank
else
    $PY $EXP/mocks.py run --start $START --stop $STOP --gib-worker $GIB
fi
status=$?
echo "python exit $status, end $(date)"
kill $MON
exit $status
