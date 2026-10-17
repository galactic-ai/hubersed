#!/bin/bash
# pocoMC run of the free-line model (poco.py) on one whole node. Each job resumes the newest state.
# Python stops TLIM s into the job; an unfinished run then requeues itself, at most 5 times.
# REQUEUE=0 turns that off. MAILTO=<address> mails a short report at the end of each run.
# Other queues: -p gpu-a100-dev -t 02:00:00 TLIM=6900, or -p development (as run.slurm).
# Several nodes: sbatch -N 2 -n 2 runs one MPI rank per node through ibrun (poco.py --mpi), each
# with its own pool of POOL workers. NMAXSTEPS sets --n-max-steps.
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 06:00:00
#SBATCH -A AST25022
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-15_broadline_free_amplitude
SETUP=experiments/2026-10-13_nautilus_resolution_matrix
TID=39627770174637084  # for file names only
VARIANT=${VARIANT:-free}
POOL=${POOL:-128}
NEFF=${NEFF:-512}
NACTIVE=${NACTIVE:-256}
SAVE=${SAVE:-2}
TLIM=${TLIM:-21000}  # s, below the wall time
FSPS_SITE=${FSPS_SITE:-/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site}
cd $REPO
RUN=${SLURM_RESTART_COUNT:-0}
LOG=$REPO/${SLURM_JOB_NAME}_${TID}_${SLURM_JOB_ID}_r${RUN}.log
LAUNCH=""
OPTS=""
if [ "${SLURM_NTASKS:-1}" -gt 1 ]; then
    LAUNCH=ibrun
    OPTS="--mpi"
    export I_MPI_PIN=0  # the forked workers must keep the whole node
fi
[ -n "$NMAXSTEPS" ] && OPTS="$OPTS --n-max-steps $NMAXSTEPS"
echo "job $SLURM_JOB_ID run $RUN on $SLURM_JOB_NODELIST, ${SLURM_NTASKS:-1} ranks, variant $VARIANT," \
    "pool $POOL per rank, n_effective $NEFF, n_active $NACTIVE, n_max_steps ${NMAXSTEPS:-default}," \
    "save every $SAVE, limit $TLIM s, start $(date)"
export PYTHONPATH=$FSPS_SITE:$REPO/$SETUP:$REPO/$EXP
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
report() {
    [ -n "$MAILTO" ] || return 0
    {
        echo "job $SLURM_JOB_ID run $RUN on $(hostname): $1, $(date)"
        tr '\r' '\n' < "$LOG" | grep -a beta | tail -1
        grep -a -E '^done in' "$LOG" | tail -1
        echo "log $LOG"
    } | mail -s "$SLURM_JOB_NAME $SLURM_JOB_ID run $RUN: $1" "$MAILTO" || true
}
timeout --kill-after=60 $TLIM $LAUNCH $REPO/.venv/bin/python $EXP/poco.py --variant $VARIANT \
    --pool $POOL --n-effective $NEFF --n-active $NACTIVE --save-every $SAVE $OPTS > "$LOG" 2>&1
status=$?
if [ $status -eq 124 ] || [ $status -eq 137 ]; then
    echo "stopped at the time limit, end $(date)"
    report "time limit"
    if [ "${REQUEUE:-1}" = 1 ] && [ "$RUN" -lt 5 ]; then
        echo "requeueing job $SLURM_JOB_ID (run $RUN of at most 5 requeues)"
        scontrol requeue $SLURM_JOB_ID || echo "requeue failed"
    fi
else
    echo "python exited with status $status, end $(date)"
    report "exit $status"
fi
