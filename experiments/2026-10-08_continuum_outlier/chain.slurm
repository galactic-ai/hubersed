#!/bin/bash
# One link of a chain of development jobs. It runs up to PER_NODE fits per node through run.sh.
# Submit several links with the same name and --dependency=singleton, for example
#   for i in 1 2 3; do
#     sbatch -J co1008 --dependency=singleton \
#       --export=ALL,FITS="miles:miles:on c3k_hr:miles:on c3k_hr:full:on miles:miles:on:off" \
#       experiments/2026-10-08_continuum_outlier/chain.slurm
#   done
# FITS is a list of LIB:WINDOW:NEBULAR[:DEGRADE], with DEGRADE on (default) or off. Fits that
# already converged are dropped first. The rest are dealt round-robin over the nodes, and each takes
# the next core slot on its node. Ask for enough nodes (-N on the sbatch line) to hold them all.
# Memory sets the limit. Peaks measured 2026-10-08 at pool 24: about 51 GB for a nebular-off fit
# (sacct MaxRSS) and about 90 GB for a nebular-on fit (3.6 GB per JAX worker). With four fits on
# one 250 GB node (about 270 GB in total) 13 workers of one fit died at once, most likely out of
# memory, and its pool then hung.
# So PER_NODE is 2. Threads are about 2100 per nebular-off fit and 2600 per nebular-on fit,
# out of 16384 per user. With POOL=48 a nebular-on fit needs about 175 GB, so set PER_NODE=1.
#SBATCH -p development
#SBATCH -N 3
#SBATCH -n 3
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-08_continuum_outlier
FITS=${FITS:?set FITS, e.g. "miles:miles:on c3k_hr:full:off"}
POOL=${POOL:-24}
PER_NODE=${PER_NODE:-2}
TIMEOUT=${TIMEOUT:-6300}  # keep below the 2 h wall time so nautilus saves first
# Core slots of POOL cores, half on each socket (0-63 and 64-127). One-socket pinning made JAX
# thrash on 2026-10-01.
half=$((POOL / 2))
SLOTS=()
for k in $(seq 0 $((64 / half - 1))); do
    a=$((k * half)) b=$(((k + 1) * half - 1))
    SLOTS+=("$a-$b,$((a + 64))-$((b + 64))")
done
cd $REPO
echo "link $SLURM_JOB_ID on $SLURM_JOB_NODELIST, fits '$FITS', pool $POOL, start $(date)"

todo=()
for f in $FITS; do
    IFS=: read lib window neb deg <<< "$f"
    name=co1008_${lib}_${window}_neb${neb}
    [ "${deg:-on}" = off ] && name=${name}_nodeg
    [ -e CONVERGED_$name ] || todo+=("$f")
done
if [ ${#todo[@]} -eq 0 ]; then
    echo "all fits converged, cancelling pending links"
    scancel -u $USER -n $SLURM_JOB_NAME -t PENDING
    exit 0
fi

i=0
for f in "${todo[@]}"; do
    IFS=: read lib window neb deg <<< "$f"
    node=$((i % SLURM_NNODES))
    slot=$((i / SLURM_NNODES))
    if [ $slot -ge $PER_NODE ]; then
        echo "no slot left for $f, skipped"
        continue
    fi
    srun --overlap -N1 -n1 --relative=$node \
        bash $EXP/run.sh $lib $window $neb $POOL ${SLOTS[$slot]} $TIMEOUT ${deg:-on} &
    echo "$f on node $node, cores ${SLOTS[$slot]}"
    i=$((i + 1))
done
wait
echo "end $(date)"
