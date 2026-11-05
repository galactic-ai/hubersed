#!/bin/bash
# Follow-up 2, MILES side: MAPs of 39633433328091196 with the logzsol prior capped at +0.2 (fit.py
# --lib miles --zmax 0.2), dust on and off, 7 starts each on one core each, on one stampede3 icx node,
# then collect both. Env as 2026-10-29_c3k_vs_miles_dust/miles_s3.slurm (ls6 MILES build copied to
# fsps_builds/miles_ls6, SPS_HOME sps_home_er). A start killed by the timeout saves nothing.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-29_nodust_map_39633433328091196/miles_z.slurm
#SBATCH -J miles_z
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/miles_z_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_nodust_map_39633433328091196
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
NSTART=7
TLIM=6000  # s per start
OUT=$REPO/results/2026-10-29_nodust_map_39633433328091196
cd $REPO
mkdir -p $OUT/miles_dust_zmax0.2 $OUT/miles_nodust_zmax0.2
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/miles_ls6/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORES=($($PY -c "import os; print(*sorted(os.sched_getaffinity(0)))"))
echo "job $SLURM_JOB_ID on $(hostname), ${#CORES[@]} cores, start $(date)"
sha256sum $EXP/fit.py $EXP/miles_z.slurm experiments/2026-10-08_continuum_outlier/fit.py \
    experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
taskset -c ${CORES[0]} $PY $EXP/fit.py check --lib miles --dust on --zmax 0.2 || exit 1
taskset -c ${CORES[0]} $PY $EXP/fit.py check --lib miles --dust off --zmax 0.2 || exit 1
STEP=$((${#CORES[@]} / (2 * NSTART)))
PIDS=()
i=0
for dust in on off; do
    sub=$([ $dust = on ] && echo miles_dust_zmax0.2 || echo miles_nodust_zmax0.2)
    for k in $(seq 0 $((NSTART - 1))); do
        c=${CORES[$((i * STEP))]}
        timeout --kill-after=60 $TLIM taskset -c $c $PY $EXP/fit.py fit --lib miles --dust $dust \
            --zmax 0.2 --start $k --n-starts $NSTART > $OUT/$sub/fit_s$k.log 2>&1 &
        PIDS+=($!)
        echo "dust $dust start $k on core $c, pid $!"
        i=$((i + 1))
        sleep 10
    done
done
wait "${PIDS[@]}" || true
echo "fits done $(date)"
grep -h "nfev\|peak memory" $OUT/miles_*_zmax0.2/fit_s*.log
$PY $EXP/fit.py collect --lib miles --dust on --zmax 0.2 --n-starts $NSTART
$PY $EXP/fit.py collect --lib miles --dust off --zmax 0.2 --n-starts $NSTART
echo "end $(date)"
