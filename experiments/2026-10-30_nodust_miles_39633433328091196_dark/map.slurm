#!/bin/bash
# MAPs of the SV3 dark spectrum of 39633433328091196 (fit.py runs er_nodust, miles_dust, miles_nodust),
# 4 starts each on one core each, on one stampede3 icx node, then collect. Env as
# 2026-10-29_nodust_map_39633433328091196/er_z.slurm (C3K ER) and miles_z.slurm (MILES), set per
# process. Memory is about 4 x 22 GiB for C3K ER and 8 x 1.5 GiB for MILES. A start killed by the
# timeout saves nothing. Needs dark_spectrum.npz, resolution_matrix.npz and the dark MAP in
# results/2026-10-30_continuum_39633433328091196_dark_map. Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-30_nodust_miles_39633433328091196_dark/map.slurm
#SBATCH -J ndmiles1030
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/ndmiles1030_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-30_nodust_miles_39633433328091196_dark
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
NSTART=4
TLIM=5400  # s per start, leaves 30 min for checks and collect
OUT=$REPO/results/2026-10-30_nodust_miles_39633433328091196_dark
cd $REPO
mkdir -p $OUT/er_nodust $OUT/miles_dust_zmax0.2 $OUT/miles_nodust_zmax0.2
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CORES=($($PY -c "import os; print(*sorted(os.sched_getaffinity(0)))"))
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), ${#CORES[@]} cores, start $(date)"
free -g
sha256sum $EXP/fit.py $EXP/map.slurm experiments/2026-10-29_nodust_map_39633433328091196/fit.py \
    experiments/2026-10-30_continuum_39633433328091196_dark_map/fit.py \
    experiments/2026-10-29_continuum_39633433328091196_map/fit.py \
    experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py experiments/2026-10-08_continuum_outlier/fit.py

run() {  # run a command with the FSPS build of library $1 (er or miles)
    local lib=$1
    shift
    if [ "$lib" = er ]; then
        env PYTHONPATH=$FSPS/c3k_er_afe_v3/fsps_site "$@"
    else
        env PYTHONPATH=$FSPS/miles_ls6/fsps_site \
            XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" "$@"
    fi
}

run er taskset -c ${CORES[0]} $PY $EXP/fit.py check --run er_nodust || exit 1
run miles taskset -c ${CORES[0]} $PY $EXP/fit.py check --run miles_dust || exit 1
run miles taskset -c ${CORES[0]} $PY $EXP/fit.py check --run miles_nodust || exit 1

STEP=$((${#CORES[@]} / (3 * NSTART)))
PIDS=()
i=0
for r in er_nodust miles_dust miles_nodust; do
    lib=${r%%_*}
    sub=$r
    [ $lib = miles ] && sub=${r}_zmax0.2
    for k in $(seq 0 $((NSTART - 1))); do
        c=${CORES[$((i * STEP))]}
        run $lib timeout --kill-after=60 $TLIM taskset -c $c $PY $EXP/fit.py fit --run $r \
            --start $k --n-starts $NSTART > $OUT/$sub/fit_s$k.log 2>&1 &
        PIDS+=($!)
        echo "$r start $k on core $c, pid $!"
        i=$((i + 1))
        sleep 20
    done
done
wait "${PIDS[@]}" || true
echo "fits done $(date)"
grep -h "nfev\|peak memory" $OUT/*/fit_s*.log
for r in er_nodust miles_dust miles_nodust; do
    run ${r%%_*} $PY $EXP/fit.py collect --run $r --n-starts $NSTART
done
echo "end $(date)"
