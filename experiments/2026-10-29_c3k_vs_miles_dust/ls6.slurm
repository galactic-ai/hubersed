#!/bin/bash
# The ls6 (MILES build) steps of Hahn's tests, in one development job to stay within the queue's
# submit limit: MILES spectra at both best points (spectra.py --lib miles), then the MILES MAPs of
# 2026-10-29_nodust_map_39633433328091196/fit.py with dust on and off (check, 7 starts each on one
# core each, collect). MILES env as 2026-10-08_continuum_outlier/run.sh. A start killed by the
# timeout saves nothing.
# Submit from the repository root on ls6:
#   sbatch experiments/2026-10-29_c3k_vs_miles_dust/ls6.slurm
#SBATCH -J miles1029
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/miles1029_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_c3k_vs_miles_dust
FIT=experiments/2026-10-29_nodust_map_39633433328091196/fit.py
NSTART=7
TLIM=5700  # s per start, leaves time for the checks and collect
OUT=$REPO/results/2026-10-29_nodust_map_39633433328091196
cd $REPO
mkdir -p $OUT/miles_dust $OUT/miles_nodust
export PYTHONPATH=/scratch/11006/nikhilgaruda/hubersed_tmp/miles_build/fsps_site
export SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
PY="uv run --no-sync python"
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
free -g
sha256sum $EXP/spectra.py $EXP/ls6.slurm $FIT experiments/2026-10-08_continuum_outlier/fit.py \
    experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
taskset -c 0 $PY $EXP/spectra.py --lib miles
taskset -c 0 $PY $FIT check --lib miles --dust on || exit 1
taskset -c 0 $PY $FIT check --lib miles --dust off || exit 1
PIDS=()
c=0
for dust in on off; do
    sub=$([ $dust = on ] && echo miles_dust || echo miles_nodust)
    for k in $(seq 0 $((NSTART - 1))); do
        timeout --kill-after=60 $TLIM taskset -c $c $PY $FIT fit --lib miles --dust $dust \
            --start $k --n-starts $NSTART > $OUT/$sub/fit_s$k.log 2>&1 &
        PIDS+=($!)
        echo "dust $dust start $k on core $c, pid $!"
        c=$((c + 2))
        sleep 10
    done
done
wait "${PIDS[@]}" || true
echo "fits done $(date)"
grep -h "nfev\|peak memory" $OUT/miles_*/fit_s*.log
$PY $FIT collect --lib miles --dust on --n-starts $NSTART
$PY $FIT collect --lib miles --dust off --n-starts $NSTART
echo "end $(date)"
