#!/bin/bash
# Nautilus run of the free-line model (fit_fork.py). Each job resumes the checkpoint, so it runs
# either as one long job or as a chain of development-queue links with the same name, e.g.
#   sbatch -J freeline --export=ALL,VARIANT=free experiments/2026-10-15_broadline_free_amplitude/run.slurm
#   sbatch -J freeline --dependency=singleton --export=ALL,VARIANT=free <same script>   (next link)
# For one long job override the queue on the command line: -p normal -t 48:00:00 and TIMEOUT
# below the wall time (TIMEOUT=170000). The first job that ends without "Stopped before
# convergence" writes a marker and cancels the pending links of its name.
# Same as 2026-10-13_nautilus_resolution_matrix/chain.slurm (c3k_er_afe_v3, n_batch 960) with pool 96
# and n_live 2000 (Nikhil 2026-10-15; ~165 GiB was free with pool 64),
# EXP, VARIANT replacing LSF and MASK, the 10-13 folder on PYTHONPATH for model_setup.py,
# and no RESUBMIT (TACC refuses sbatch from compute nodes).
# A job that stops before convergence requeues itself (scontrol requeue), so it runs again without
# a new submission. ls6 has JobRequeue 0 (hence --requeue here) and MaxBatchRequeue 5, so each
# submitted job runs at most 6 times; REQUEUE=0 turns this off. Logs get the restart count.
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-15_broadline_free_amplitude
SETUP=experiments/2026-10-13_nautilus_resolution_matrix
TID=39627770174637084  # fixed in model_setup.py; used only in the marker and log names
VARIANT=${VARIANT:-free}
POOL=${POOL:-96}
NLIVE=${NLIVE:-2000}
TIMEOUT=${TIMEOUT:-6000}  # sampler timeout in s; keep it below the -t wall time
NBATCH=${NBATCH:-960}
FLIVE=${FLIVE:-0.01}
FSPS_SITE=${FSPS_SITE:-/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site}
cd $REPO
MARKER=$REPO/CONVERGED_${SLURM_JOB_NAME}_$TID
RUN=${SLURM_RESTART_COUNT:-0}
LOG=$REPO/${SLURM_JOB_NAME}_${TID}_${SLURM_JOB_ID}_r${RUN}.log
echo "job $SLURM_JOB_ID run $RUN on $(hostname), TID $TID, variant $VARIANT, pool $POOL, n_live $NLIVE," \
    "n_batch $NBATCH, timeout $TIMEOUT, f_live $FLIVE, start $(date)"
if [ -e "$MARKER" ]; then
    echo "$MARKER exists, cancelling pending links"
    scancel -u $USER -n $SLURM_JOB_NAME -t PENDING
    exit 0
fi
export PYTHONPATH=$FSPS_SITE:$REPO/$SETUP
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# the venv's python directly, not uv run (uv run hung on a stale lock on 2026-10-10)
$REPO/.venv/bin/python $EXP/fit_fork.py --variant $VARIANT --n-live $NLIVE \
    --pool $POOL --n-batch $NBATCH --timeout $TIMEOUT --f-live $FLIVE > "$LOG" 2>&1
if grep -q "Stopped before convergence" "$LOG"; then
    echo "not converged, end $(date)"
    if [ "${REQUEUE:-1}" = 1 ] && [ "$RUN" -lt 5 ]; then
        echo "requeueing job $SLURM_JOB_ID (run $RUN of at most 5 requeues)"
        scontrol requeue $SLURM_JOB_ID || echo "requeue failed, the next link continues"
    fi
else
    echo "ended without 'Stopped before convergence' (converged or failed), $(date)" > "$MARKER"
    echo "converged or failed, wrote $MARKER, cancelling pending links, end $(date)"
    scancel -u $USER -n $SLURM_JOB_NAME -t PENDING
fi
