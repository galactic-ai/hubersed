#!/bin/bash
# Ladder step 5 production: fit.py run --mpi on spr, one MPI rank per node, each with a pool forked
# after the SSP build and the speclib free; rank 0 also runs pocoMC. Resumes from the newest state in
# results/2026-11-07_emline_ladder_39627770174637084/<STEM>/. Python stops TLIM s into the job; an
# unfinished run then requeues itself, at most 5 times (REQUEUE=0 turns that off).
# GIB is the commit budget per worker (Cue included). Submit from the repository root on stampede3:
#   STEP=<3|4|5> sbatch --export=ALL [--dependency=afterok:<check job>] experiments/2026-11-07_emline_ladder_39627770174637084/prod.slurm
#SBATCH -J step5prod
#SBATCH -p spr
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A TG-PHY240056
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /scratch/11006/nikhilgaruda/logs/step5prod_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-11-07_emline_ladder_39627770174637084
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
GIB=${GIB:-10}
STEP=${STEP:-5}
TLIM=${TLIM:-13800}  # s, leaves 10 min of the 4 h
RUN=${SLURM_RESTART_COUNT:-0}
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
export I_MPI_PIN=0 PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID step $STEP build $BUILD run $RUN on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), GIB $GIB, start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/fit.py $EXP/prod.slurm
(while sleep 600; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
MON=$!
timeout --kill-after=60 $TLIM ibrun $PY $EXP/fit.py run --step $STEP --mpi --gib-worker $GIB $EXTRA
status=$?
kill $MON
if [ $status -eq 124 ] || [ $status -eq 137 ]; then
    echo "stopped at the time limit, end $(date)"
    if [ "${REQUEUE:-1}" = 1 ] && [ "$RUN" -lt 5 ]; then
        echo "requeueing job $SLURM_JOB_ID (run $RUN of at most 5 requeues)"
        scontrol requeue $SLURM_JOB_ID || echo "requeue failed"
    fi
else
    echo "python exited with status $status, end $(date)"
fi
exit $status
