#!/bin/bash
# MILES multi-start continuum MAP on the recoadded SV3 spectrum of 39627757533007793, as
# 2026-11-01_recoadd_afe0/map.slurm (fit.py check first, then 7 starts, one core each, collect at
# the end), with the MILES build of python-fsps. It runs the fit.py files of SRC, a copy of
# experiments/ holding 2026-10-28_continuum_c3k_er_rmatrix_map, 2026-10-29_continuum_recoadd and
# this folder, so the shared clone is not changed.
#   sbatch -p spr --export=ALL,SRC=<copy> map.slurm
# STARTS (for example "4 5 6") runs only those starts of NSTART; COLLECT=0 skips collect.
#SBATCH -J milesmap
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/milesmap_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=${SRC:?set SRC to the experiments copy}/2026-11-01_miles_recoadd_map
export RECOADD_SURVEY=sv3
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
NSTART=${NSTART:-7}
STARTS=${STARTS:-$(seq 0 $((NSTART - 1)))}
COLLECT=${COLLECT:-1}
NEB=${NEB:-off}
LOGTAG=$([ "$NEB" = on ] && echo _cue || true)
TAG=${TAG:-}  # suffix of the output files
STAGGER=${STAGGER:-20}  # s between start-ups
MAXFEV=${MAXFEV:-150000}
JITTER=${JITTER:-0.02}
TLIM=${TLIM:-9600}  # s per start, leaves 20 min for collect
OUT=$REPO/results/2026-11-01_miles_recoadd_map
mkdir -p $OUT
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/miles_ls6/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/fsps_5
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"  # Cue is JAX
CORES=($($PY -c "import os; print(*sorted(os.sched_getaffinity(0)))"))
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), $NSTART starts," \
    "survey $RECOADD_SURVEY, maxfev $MAXFEV, jitter $JITTER, nebular $NEB, tag [$TAG], stagger $STAGGER s, tlim $TLIM s, ${#CORES[@]} cores, start $(date)"
free -g
echo "--- check $(date +%T)"
taskset -c 0 $PY $EXP/fit.py check --n-calls 7 || exit 1
sha256sum $EXP/fit.py $SRC/2026-10-29_continuum_recoadd/fit.py $SRC/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py

# memory and load every 10 min
(while true; do
    echo "--- $(date +%T) $(free -g | awk '/Mem/ {print "used " $3 " of " $2 " GB"}')," \
        "load $(cut -d' ' -f1 /proc/loadavg)"
    sleep 600
done) &
MON=$!

STEP=$((${#CORES[@]} / NSTART))
PIDS=()
LAST=$(echo $STARTS | awk '{print $NF}')
for k in $STARTS; do
    c=${CORES[$((k * STEP))]}
    timeout --kill-after=60 $TLIM taskset -c $c $PY $EXP/fit.py fit --start $k \
        --n-starts $NSTART --maxfev $MAXFEV --jitter $JITTER --nebular $NEB --tag "$TAG" \
        > $OUT/fit_s$k$LOGTAG$TAG.log 2>&1 &
    PIDS+=($!)
    echo "start $k on core $c, pid $!"
    [ $k -ne $LAST ] && sleep $STAGGER  # stagger the start-ups
done
wait "${PIDS[@]}" || true
kill $MON
echo "fits done $(date)"
grep -h "nfev\|peak memory" $OUT/fit_s*$LOGTAG$TAG.log
[ "$COLLECT" = 1 ] && $PY $EXP/fit.py collect --n-starts $NSTART --nebular $NEB --tag "$TAG"
echo "end $(date)"
