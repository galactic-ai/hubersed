#!/bin/bash
# One link of a chain of development-queue jobs; each link resumes the nautilus checkpoint.
# Submit several links with the same job name and --dependency=singleton, for example
#   for i in 1 2 3 4 5 6; do
#     sbatch -J b0930_free --dependency=singleton --export=ALL,FB=free,TID=39627770174637084 \
#       experiments/2026-09-30_nautilus_c3k_broadline_split45/chain.slurm
#   done
# The first link that converges writes a marker and cancels the pending links of its name.
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A AST26017
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-09-30_nautilus_c3k_broadline_split45
TID=${TID:-39627770174637084}
FB=${FB:?set FB=free or FB=shared}
SPLIT=${SPLIT:-45}
EXTRA=${EXTRA:-}  # e.g. EXTRA=--keep-sky-mask
POOL=${POOL:-24}
TIMEOUT=${TIMEOUT:-6300}  # sampler timeout in s; keep it below the -t wall time
# CORES pins the run to a core list so JAX sizes its thread pools to it. Use "+" between ranges
# (sbatch --export splits on commas), and spread the cores over both sockets (0-63 and 64-127):
# pinning everything to one socket fills that NUMA node's memory, the kernel then evicts the
# shared libraries, and JAX compiles thrash on page faults (seen 2026-10-01, CORES=0-30).
PIN=${CORES:+taskset -c ${CORES//+/,}}
cd $REPO
MARKER=$REPO/CONVERGED_${SLURM_JOB_NAME}_$TID
LOG=$REPO/${SLURM_JOB_NAME}_${TID}_${SLURM_JOB_ID}.log
echo "link $SLURM_JOB_ID on $(hostname), TID $TID, FB $FB, split $SPLIT, extra '$EXTRA'," \
    "pool $POOL, timeout $TIMEOUT, cores '${CORES:-all}', start $(date)"
if [ -e "$MARKER" ]; then
    echo "$MARKER exists, cancelling pending links"
    scancel -u $USER -n $SLURM_JOB_NAME -t PENDING
    exit 0
fi
export SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
$PIN uv run --no-sync python $EXP/fit.py --tid $TID --forbidden-broad $FB --sigma-split $SPLIT \
    $EXTRA --pool $POOL --n-batch 480 --timeout $TIMEOUT > "$LOG" 2>&1
if grep -q "Stopped before convergence" "$LOG"; then
    echo "not converged, end $(date)"
else
    echo "ended without 'Stopped before convergence' (converged or failed), $(date)" > "$MARKER"
    echo "converged or failed, wrote $MARKER, cancelling pending links, end $(date)"
    scancel -u $USER -n $SLURM_JOB_NAME -t PENDING
fi
