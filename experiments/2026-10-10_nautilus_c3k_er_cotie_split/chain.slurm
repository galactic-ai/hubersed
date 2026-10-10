#!/bin/bash
# One link of a chain of development-queue jobs; each link resumes the nautilus checkpoint.
# Submit several links with the same job name and --dependency=singleton, for example
#   for i in 1 2 3 4 5 6; do
#     sbatch -J er1010_cotie --dependency=singleton --export=ALL,FB=shared,TID=39627770174637084 \
#       experiments/2026-10-10_nautilus_c3k_er_cotie_split/chain.slurm
#   done
# The first link that converges writes a marker and cancels the pending links of its name.
# Same as the 2026-10-09_nautilus_c3k_er_broadline chain except for EXP and the defaults below,
# which are the settings the converged ER link used (fit_fork.py, c3k_er_afe_v3, pool 96, n_batch
# 960, timeout 6000 s). That chain in turn is the 2026-09-30_nautilus_c3k_broadline_split45 chain
# with a different FSPS build, SPS_HOME, account and pool. With fit_fork.py the workers share the
# FSPS setup copy on write; at pool 96 the converged ER link (3501933) grew to 215-218 GiB of the
# 250 GiB node after exploration, so watch the first link's memory.
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-10_nautilus_c3k_er_cotie_split
TID=${TID:-39627770174637084}
FB=${FB:?set FB=free or FB=shared}
SPLIT=${SPLIT:-45}
POOL=${POOL:-96}
TIMEOUT=${TIMEOUT:-6000}  # sampler timeout in s; keep it below the -t wall time
NBATCH=${NBATCH:-960}
# SCRIPT=fit_fork.py forks the pool after one FSPS setup and SSP build; it needs an fsps build
# with get_ssp_slot (python-fsps 2c5168f), set with FSPS_SITE
SCRIPT=${SCRIPT:-fit_fork.py}
FSPS_SITE=${FSPS_SITE:-/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site}
# RESUBMIT=1 makes a link that stops before convergence submit the next link itself, so the
# chain runs on without anyone adding links; a link that fails writes the marker instead
# CORES pins the run to a core list so JAX sizes its thread pools to it. Use "+" between ranges
# (sbatch --export splits on commas), and spread the cores over both sockets (0-63 and 64-127):
# pinning everything to one socket fills that NUMA node's memory, the kernel then evicts the
# shared libraries, and JAX compiles thrash on page faults (seen 2026-10-01, CORES=0-30).
PIN=${CORES:+taskset -c ${CORES//+/,}}
cd $REPO
MARKER=$REPO/CONVERGED_${SLURM_JOB_NAME}_$TID
LOG=$REPO/${SLURM_JOB_NAME}_${TID}_${SLURM_JOB_ID}.log
echo "link $SLURM_JOB_ID on $(hostname), TID $TID, FB $FB, split $SPLIT," \
    "pool $POOL, n_batch $NBATCH, timeout $TIMEOUT, script $SCRIPT, cores '${CORES:-all}'," \
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
$PIN $REPO/.venv/bin/python $EXP/$SCRIPT --tid $TID --forbidden-broad $FB --sigma-split $SPLIT \
    --pool $POOL --n-batch $NBATCH --timeout $TIMEOUT > "$LOG" 2>&1
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
