#!/bin/bash
# One link of a chain of development jobs that runs up to 4 fits per node, each through run.sh.
# Submit several links with the same name and --dependency=singleton, for example
#   for i in 1 2 3; do
#     sbatch -J co1008 --dependency=singleton \
#       --export=ALL,FITS="miles:miles:on c3k_hr:miles:on c3k_hr:miles:off c3k_hr:full:on c3k_hr:full:off" \
#       experiments/2026-10-08_continuum_outlier/chain.slurm
#   done
# FITS is a list of LIB:WINDOW:NEBULAR[:DEGRADE], with DEGRADE on (default) or off. Fits are dealt
# round-robin over the nodes, and each takes the next core slot on its node. Fits that already
# converged skip themselves (run.sh checks the marker).
# Limits per node, measured 2026-10-08 on c309-006 with pool 24: about 2100 threads for a nebular-off
# fit and 2600 for a nebular-on fit, out of 16384 per user, and roughly 40 GB of memory each.
# So 4 fits per node at pool 24 fit with room to spare.
#SBATCH -p development
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-08_continuum_outlier
FITS=${FITS:?set FITS, e.g. "miles:miles:on c3k_hr:full:off"}
POOL=${POOL:-24}
TIMEOUT=${TIMEOUT:-6300}  # keep below the 2 h wall time so nautilus saves first
# Core slots spread over both sockets (0-63 and 64-127). One-socket pinning made JAX thrash on 2026-10-01.
SLOTS=(0-11,64-75 12-23,76-87 24-35,88-99 36-47,100-111)
cd $REPO
echo "link $SLURM_JOB_ID on $SLURM_JOB_NODELIST, fits '$FITS', pool $POOL, start $(date)"

left=0
for f in $FITS; do
    IFS=: read lib window neb deg <<< "$f"
    name=co1008_${lib}_${window}_neb${neb}
    [ "${deg:-on}" = off ] && name=${name}_nodeg
    [ -e CONVERGED_$name ] || left=$((left + 1))
done
if [ $left -eq 0 ]; then
    echo "all fits converged, cancelling pending links"
    scancel -u $USER -n $SLURM_JOB_NAME -t PENDING
    exit 0
fi

i=0
for f in $FITS; do
    IFS=: read lib window neb deg <<< "$f"
    node=$((i % SLURM_NNODES))
    slot=$((i / SLURM_NNODES))
    if [ $slot -ge ${#SLOTS[@]} ]; then
        echo "no core slot left for $f, skipped"
        continue
    fi
    srun --overlap -N1 -n1 --relative=$node bash $EXP/run.sh $lib $window $neb $POOL ${SLOTS[$slot]} $TIMEOUT ${deg:-on} &
    echo "$f on node $node, cores ${SLOTS[$slot]}"
    i=$((i + 1))
done
wait
echo "end $(date)"
