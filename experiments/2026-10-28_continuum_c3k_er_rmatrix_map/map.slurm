#!/bin/bash
# Multi-start MAP of this experiment on one stampede3 icx node (80 cores, 256 GB). Starts
# 0..NSTART-1 run at once, each pinned to one core, spread over the node. Each process peaks at
# 21.6 GiB (test.slurm, job 3655311), so 7 starts need about 151 GiB. A warm call takes 0.048 s,
# but 2000 Powell calls took 1043 s there, since FSPS (zcontinuous=1) builds each SSP slot of
# metallicity and afe the first time logzsol or afe moves into it (fsps.f90 lines 244-258 of
# the c3k_er_afe_v3 source). The wall time is set long for that. A start killed by the timeout
# saves nothing. The collect step then keeps the best start.
# Submit from the repository root on stampede3, after test.slurm wrote resolution_matrix.npz:
#   sbatch experiments/2026-10-28_continuum_c3k_er_rmatrix_map/map.slurm
# With Cue nebular emission (needs the continuum MAP's <TID>_map.npz for start 0):
#   sbatch --export=ALL,NEB=on experiments/2026-10-28_continuum_c3k_er_rmatrix_map/map.slurm
#SBATCH -J er1028map
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/er1028map_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-28_continuum_c3k_er_rmatrix_map
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
NSTART=${NSTART:-7}
NEB=${NEB:-off}
LOGTAG=$([ "$NEB" = on ] && echo _cue || true)
MAXFEV=${MAXFEV:-150000}
JITTER=${JITTER:-0.02}
TLIM=${TLIM:-106800}  # s per start, leaves 10 min for collect
OUT=$REPO/results/2026-10-28_continuum_c3k_er_rmatrix_map
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"  # Cue is JAX
CORES=($($PY -c "import os; print(*sorted(os.sched_getaffinity(0)))"))
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), $NSTART starts," \
    "maxfev $MAXFEV, jitter $JITTER, nebular $NEB, ${#CORES[@]} cores, start $(date)"
free -g
sha256sum $EXP/fit.py $EXP/map.slurm

# memory and load every 10 min
(while true; do
    echo "--- $(date +%T) $(free -g | awk '/Mem/ {print "used " $3 " of " $2 " GB"}')," \
        "load $(cut -d' ' -f1 /proc/loadavg)"
    sleep 600
done) &
MON=$!

STEP=$((${#CORES[@]} / NSTART))
PIDS=()
for k in $(seq 0 $((NSTART - 1))); do
    c=${CORES[$((k * STEP))]}
    timeout --kill-after=60 $TLIM taskset -c $c $PY $EXP/fit.py fit --start $k \
        --n-starts $NSTART --maxfev $MAXFEV --jitter $JITTER --nebular $NEB \
        > $OUT/fit_s$k$LOGTAG.log 2>&1 &
    PIDS+=($!)
    echo "start $k on core $c, pid $!"
    sleep 20  # stagger the imports
done
wait "${PIDS[@]}" || true
kill $MON
echo "fits done $(date)"
grep -h "nfev\|peak memory" $OUT/fit_s*$LOGTAG.log
$PY $EXP/fit.py collect --n-starts $NSTART --nebular $NEB
echo "end $(date)"
