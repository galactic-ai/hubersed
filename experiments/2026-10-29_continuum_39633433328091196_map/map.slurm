#!/bin/bash
# Multi-start continuum MAP of 39633433328091196 on one stampede3 icx node, as
# 2026-10-29_continuum_recoadd/map.slurm (7 starts, one core each, collect at the end). Continuum
# only. A start killed by the timeout saves nothing.
# Submit from the repository root on stampede3, after test.slurm wrote resolution_matrix.npz:
#   sbatch experiments/2026-10-29_continuum_39633433328091196_map/map.slurm
#SBATCH -J c4map1029
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/c4map1029_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_continuum_39633433328091196_map
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
NSTART=${NSTART:-7}
TAG=${TAG:-}  # suffix of the output files
STAGGER=${STAGGER:-20}  # s between start-ups
MAXFEV=${MAXFEV:-150000}
JITTER=${JITTER:-0.02}
TLIM=${TLIM:-106800}  # s per start, leaves 10 min for collect
OUT=$REPO/results/2026-10-29_continuum_39633433328091196_map
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CORES=($($PY -c "import os; print(*sorted(os.sched_getaffinity(0)))"))
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), $NSTART starts," \
    "maxfev $MAXFEV, jitter $JITTER, tag [$TAG], stagger $STAGGER s, tlim $TLIM s, ${#CORES[@]} cores, start $(date)"
free -g
sha256sum $EXP/fit.py $EXP/map.slurm experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py

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
        --n-starts $NSTART --maxfev $MAXFEV --jitter $JITTER --tag "$TAG" \
        > $OUT/fit_s$k$TAG.log 2>&1 &
    PIDS+=($!)
    echo "start $k on core $c, pid $!"
    [ $k -lt $((NSTART - 1)) ] && sleep $STAGGER  # stagger the start-ups
done
wait "${PIDS[@]}" || true
kill $MON
echo "fits done $(date)"
grep -h "nfev\|peak memory" $OUT/fit_s*$TAG.log
$PY $EXP/fit.py collect --n-starts $NSTART --tag "$TAG"
echo "end $(date)"
