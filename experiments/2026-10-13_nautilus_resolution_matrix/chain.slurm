#!/bin/bash
# One link of a chain of development-queue jobs; each link resumes the nautilus checkpoint.
# Submit several links with the same job name and --dependency=singleton, for example
#   for i in 1 2 3 4 5 6; do
#     sbatch -J rm1013_mask --dependency=singleton --export=ALL \
#       experiments/2026-10-13_nautilus_resolution_matrix/chain.slurm
#   done
# The first link that converges writes a marker and cancels the pending links of its name.
# Same as the 2026-10-10_nautilus_c3k_er_cotie_split chain (fit_fork.py, c3k_er_afe_v3, pool 64,
# n_batch 960, timeout 6000 s) except for EXP and the options: LSF and MASK replace FB and SPLIT,
# which model_setup.py fixes at shared and 45 km/s, and the target is fixed there too.
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-13_nautilus_resolution_matrix
TID=39627770174637084  # fixed in model_setup.py; used only in the marker and log names
LSF=${LSF:-rmatrix}
MASK=${MASK:-mask}  # mask: the missing-line mask of the rmatrix mask MAP; nomask keeps those pixels
POOL=${POOL:-64}
TIMEOUT=${TIMEOUT:-6000}  # sampler timeout in s; keep it below the -t wall time
NBATCH=${NBATCH:-960}
# FLIVE ends the exploration phase when f_live drops to it; 0.01 is the nautilus default
FLIVE=${FLIVE:-0.01}
# SCRIPT=fit_fork.py forks the pool after one FSPS setup and SSP build; it needs an fsps build
# with get_ssp_slot (python-fsps 2c5168f), set with FSPS_SITE
SCRIPT=${SCRIPT:-fit_fork.py}
FSPS_SITE=${FSPS_SITE:-/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site}
# RESUBMIT=1 makes a link that stops before convergence submit the next link itself, so the
# chain runs on without anyone adding links; a link that fails writes the marker instead
# CORES pins the run to a core list so JAX sizes its thread pools to it. Use "+" between ranges
# (sbatch --export splits on commas), and spread the cores over both sockets (0-63 and 64-127).
PIN=${CORES:+taskset -c ${CORES//+/,}}
cd $REPO
MARKER=$REPO/CONVERGED_${SLURM_JOB_NAME}_$TID
LOG=$REPO/${SLURM_JOB_NAME}_${TID}_${SLURM_JOB_ID}.log
echo "link $SLURM_JOB_ID on $(hostname), TID $TID, LSF $LSF, MASK $MASK," \
    "pool $POOL, n_batch $NBATCH, timeout $TIMEOUT, f_live $FLIVE, script $SCRIPT," \
    "cores '${CORES:-all}'," \
    "start $(date)"
if [ -e "$MARKER" ]; then
    echo "$MARKER exists, cancelling pending links"
    scancel -u $USER -n $SLURM_JOB_NAME -t PENDING
    exit 0
fi
export PYTHONPATH=$FSPS_SITE
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# the venv's python directly, not uv run: on 2026-10-10 uv run hung on a stale lock on ~/.cache/uv
$PIN $REPO/.venv/bin/python $EXP/$SCRIPT --lsf $LSF --mask $MASK \
    --pool $POOL --n-batch $NBATCH --timeout $TIMEOUT --f-live $FLIVE > "$LOG" 2>&1
if grep -q "Stopped before convergence" "$LOG"; then
    echo "not converged, end $(date)"
    if [ "${RESUBMIT:-0}" = 1 ]; then
        # drop this job's SLURM_ variables so they do not leak into the next job
        env $(env | grep -o '^SLURM_[A-Za-z0-9_]*' | sed 's/^/-u /') sbatch -J $SLURM_JOB_NAME \
            --dependency=singleton --export=ALL $REPO/$EXP/chain.slurm
    fi
else
    echo "ended without 'Stopped before convergence' (converged or failed), $(date)" > "$MARKER"
    echo "converged or failed, wrote $MARKER, cancelling pending links, end $(date)"
    scancel -u $USER -n $SLURM_JOB_NAME -t PENDING
fi
