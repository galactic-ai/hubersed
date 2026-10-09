#!/bin/bash
# Run or resume one fit of this experiment on an ls6 compute node, for example as a step in
# another job of ours:
#   srun --jobid=<job> --overlap -p development -A AST25022 -t 02:00:00 -N1 -n1 \
#       bash experiments/2026-10-08_continuum_outlier/run.sh 39627757533007793 miles miles on 24 \
#       16-27,80-91 3000
# Arguments are TID LIB WINDOW NEBULAR POOL CORES TIMEOUT [DEGRADE [AFE]], with DEGRADE on
# (default) or off and AFE none (default), free or zero.
# Spread CORES over both sockets (0-63 and 64-127), or JAX compiles thrash (seen 2026-10-01).
# Keep TIMEOUT below the time the job has left, so nautilus saves before the step is killed.
# MILES runs use the MILES build of python-fsps, made by build_miles_fsps.sh, put first on PYTHONPATH.
# Alpha runs use the AFE_FLAG build (-mcmodel=medium) and an SPS_HOME that also holds the alpha
# spectra, both of 2026-10-02 (knowledge/_evidence/2026-10-01_afe_build/README.md).
set -euo pipefail
TID=$1 LIB=$2 WINDOW=$3 NEB=$4 POOL=$5 CORES=$6 TIMEOUT=$7 DEG=${8:-on} AFE=${9:-none}
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-08_continuum_outlier
NAME=co1008_${TID}_${LIB}_${WINDOW}_neb${NEB}
[ "$DEG" = off ] && NAME=${NAME}_nodeg
[ "$AFE" != none ] && NAME=${NAME}_afe$AFE
cd $REPO
if [ -e CONVERGED_$NAME ]; then
    echo "CONVERGED_$NAME exists"
    exit 0
fi
export SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
if [ "$LIB" = miles ]; then
    export PYTHONPATH=/scratch/11006/nikhilgaruda/hubersed_tmp/miles_build/fsps_site
fi
if [ "$AFE" != none ]; then
    export PYTHONPATH=/scratch/11006/nikhilgaruda/hubersed_tmp/afe_build/fsps_site
    export SPS_HOME=/scratch/11006/nikhilgaruda/hubersed_tmp/afe_build/sps_home
fi
LOG=$REPO/${NAME}_${SLURM_JOB_ID}_$(date +%m%d%H%M).log
echo "$NAME on $(hostname), job $SLURM_JOB_ID, pool $POOL, cores $CORES, timeout $TIMEOUT, start $(date)"
taskset -c "$CORES" uv run --no-sync python $EXP/fit.py --tid "$TID" --lib "$LIB" --window "$WINDOW" \
    --nebular "$NEB" --degrade "$DEG" --afe "$AFE" --pool "$POOL" --n-batch 480 --timeout "$TIMEOUT" > "$LOG" 2>&1 || true
if grep -q "Stopped before convergence" "$LOG"; then
    echo "not converged, end $(date)"
else
    echo "ended without 'Stopped before convergence' (converged or failed), $(date)" > CONVERGED_$NAME
    echo "converged or failed, wrote CONVERGED_$NAME, end $(date)"
fi
