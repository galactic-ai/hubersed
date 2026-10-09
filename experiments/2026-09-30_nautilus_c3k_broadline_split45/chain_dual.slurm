#!/bin/bash
# One link of a chain that runs several fits side by side on one development node.
# FITS lists them as TID:MODE pairs; each fit gets an equal block of cores (taskset), so JAX sizes its
# thread pools to that block instead of 128, and POOL workers. A fit whose marker exists is skipped.
# Submit links with the same job name and --dependency=singleton, for example
#   for i in 1 2 3; do
#     sbatch -J b0930_dual --dependency=singleton \
#       --export=ALL,FITS="39627770174637084:free 39628357133926597:shared" \
#       experiments/2026-09-30_nautilus_c3k_broadline_split45/chain_dual.slurm
#   done
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A AST26017
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-09-30_nautilus_c3k_broadline_split45
FITS=${FITS:?set FITS="TID:MODE TID:MODE"}
POOL=${POOL:-16}
SPLIT=${SPLIT:-45}
cd $REPO
export SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "link $SLURM_JOB_ID on $(hostname), FITS '$FITS', pool $POOL, split $SPLIT, start $(date)"

# fits still to run (no marker yet); the 128 cores are split into one contiguous block per fit,
# so with two fits each gets one socket and with three the middle one straddles both
active=()
for f in $FITS; do
    TID=${f%%:*}
    FB=${f##*:}
    if [ -e "$REPO/CONVERGED_${SLURM_JOB_NAME}_${TID}_${FB}" ]; then
        echo "$TID $FB already converged, skipping"
    else
        active+=("$f")
    fi
done
n=${#active[@]}
CORES=()
for ((k = 0; k < n; k++)); do
    CORES+=("$((k * 128 / n))-$(((k + 1) * 128 / n - 1))")
done
i=0
pids=()
logs=()
for f in "${active[@]}"; do
    TID=${f%%:*}
    FB=${f##*:}
    MARKER=$REPO/CONVERGED_${SLURM_JOB_NAME}_${TID}_${FB}
    LOG=$REPO/${SLURM_JOB_NAME}_${TID}_${FB}_${SLURM_JOB_ID}.log
    taskset -c ${CORES[$i]} uv run --no-sync python $EXP/fit.py --tid $TID --forbidden-broad $FB \
        --sigma-split $SPLIT --pool $POOL --n-batch 480 --timeout 6300 > "$LOG" 2>&1 &
    pids+=($!)
    logs+=("$LOG|$MARKER|$TID $FB")
    echo "started $TID $FB on cores ${CORES[$i]}, log $LOG"
    i=$((i + 1))
done
for p in "${pids[@]}"; do wait $p; done

left=0
for entry in "${logs[@]}"; do
    LOG=${entry%%|*}
    rest=${entry#*|}
    MARKER=${rest%%|*}
    NAME=${rest#*|}
    if grep -q "Stopped before convergence" "$LOG"; then
        echo "$NAME not converged"
        left=$((left + 1))
    else
        echo "ended without 'Stopped before convergence' (converged or failed), $(date)" > "$MARKER"
        echo "$NAME converged or failed, wrote $MARKER"
    fi
done
if [ $left -eq 0 ]; then
    echo "all fits done, cancelling pending links, end $(date)"
    scancel -u $USER -n $SLURM_JOB_NAME -t PENDING
fi
