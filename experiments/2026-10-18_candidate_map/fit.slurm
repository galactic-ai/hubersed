#!/bin/bash
# Multi-start MAP of one candidate (2026-10-18_candidate_map.py) on one vm-small node.
# Starts 0..NSTART-1 run at once, each pinned to its own pair of cores. Start 0 is the
# unjittered start, so it repeats the Mac quick MAP in the ls6 environment.
# After the fits, the report step writes <TID>_map_s<K>.txt and .png for each finished start.
# FIT_TID stays unset: map.py reads it at import for the #14 start file.
# Submit from the repository root on ls6:
#   sbatch --export=ALL,TID=39628357133926597 experiments/2026-10-18_candidate_map.slurm
#SBATCH -J candmap
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/candmap_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
SCRIPT=experiments/2026-10-18_candidate_map.py
TID=${TID:?set TID}
NSTART=${NSTART:-8}
MAXFEV=${MAXFEV:-120000}
JITTER=${JITTER:-0.1}
TLIM=${TLIM:-56400}  # s per fit, leaves 30 min for the reports
FSPS_SITE=${FSPS_SITE:-/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site}
OUT=$REPO/results/2026-10-18_candidate_map
cd $REPO
unset FIT_TID
export PYTHONPATH=$FSPS_SITE:$REPO/experiments/2026-10-15_broadline_free_amplitude:$REPO/experiments/2026-10-13_nautilus_resolution_matrix
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
PY=$REPO/.venv/bin/python
mkdir -p $OUT

CORES=($($PY -c "import os; print(*sorted(os.sched_getaffinity(0)))"))
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), tid $TID, $NSTART starts," \
    "maxfev $MAXFEV, jitter $JITTER, ${#CORES[@]} cores, start $(date)"
free -g
sha256sum $SCRIPT experiments/2026-10-18_candidate_map.slurm

# Memory and load every 10 min, so a too-big start count shows up in the log.
(while true; do
    echo "--- $(date +%T) $(free -g | awk '/Mem/ {print "used " $3 " of " $2 " GB"}')," \
        "load $(cut -d' ' -f1 /proc/loadavg)"
    ps -u $USER -o pid=,rss=,pcpu=,args= | awk '/candidate_map.py fit/ && !/awk/ {
        printf "  pid %s rss %.1f GB cpu %s%%\n", $1, $2/1048576, $3}'
    sleep 600
done) &
MON=$!

PIDS=()
for k in $(seq 0 $((NSTART - 1))); do
    a=${CORES[$(((2 * k) % ${#CORES[@]}))]}
    b=${CORES[$(((2 * k + 1) % ${#CORES[@]}))]}
    timeout --kill-after=60 $TLIM taskset -c $a,$b $PY $SCRIPT fit --tid $TID --start $k \
        --n-starts $NSTART --maxfev $MAXFEV --jitter $JITTER > $OUT/${TID}_fit_s$k.log 2>&1 &
    PIDS+=($!)
    echo "start $k on cores $a,$b, pid $!"
    sleep 20  # stagger the imports
done
wait "${PIDS[@]}"
kill $MON
echo "fits done $(date)"
grep -h "nfev" $OUT/${TID}_fit_s*.log

for k in $(seq 0 $((NSTART - 1))); do
    if [ -f $OUT/${TID}_map_free_s$k.npz ]; then
        $PY $SCRIPT report --tid $TID --start $k > /dev/null 2>> $OUT/${TID}_report.log &
    else
        echo "start $k has no result"
    fi
done
wait
grep "chi2/" $OUT/${TID}_map_s*.txt
echo "end $(date)"
