#!/bin/bash
#SBATCH -J broad0928_dev
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/broad0928_dev_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-09-28_nautilus_c3k_broadline_shared
TID=${TID:-39628357133926597}
cd $REPO
MARKER=$REPO/CONVERGED_broad0928_$TID
LOG=$REPO/broad0928_${TID}_dev_${SLURM_JOB_ID}.out
echo "link $SLURM_JOB_ID on $(hostname), TID $TID, start $(date), ulimit -u $(ulimit -u)"
if [ -e "$MARKER" ]; then
    echo "$MARKER exists, cancelling pending links"
    scancel -u $USER -n broad0928_dev -t PENDING
    exit 0
fi
export SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
uv run --no-sync python $EXP/fit.py --tid $TID --pool 24 --n-batch 480 --timeout 6300 \
    > "$LOG" 2>&1
if grep -q "Stopped before convergence" "$LOG"; then
    echo "not converged, end $(date)"
else
    echo "ended without 'Stopped before convergence' (converged or failed), $(date)" > "$MARKER"
    echo "converged or failed, wrote $MARKER, cancelling pending links, end $(date)"
    scancel -u $USER -n broad0928_dev -t PENDING
fi
