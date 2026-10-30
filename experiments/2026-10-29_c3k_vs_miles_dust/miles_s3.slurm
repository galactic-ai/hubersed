#!/bin/bash
# The MILES steps of Hahn's tests on one stampede3 icx node, in place of ls6.slurm: check the
# MILES build (log L at the 10-08 max-L must match), MILES spectra at both best points
# (spectra.py --lib miles), then the MILES MAPs of 2026-10-29_nodust_map_39633433328091196/fit.py
# with dust on and off (7 starts each, one core each) and collect.
# FSPS is the ls6 MILES build of 2026-10-08_continuum_outlier/build_miles_fsps.sh, its fsps_site
# copied from ls6 /scratch to fsps_builds/miles_ls6 on /work (2026-10-29). SPS_HOME is sps_home_er,
# which holds the MILES spectra; the check step tests that it reproduces the 10-08 run.
# A start killed by the timeout saves nothing.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-29_c3k_vs_miles_dust/miles_s3.slurm
#SBATCH -J miles_s3
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/miles_s3_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_c3k_vs_miles_dust
FIT=experiments/2026-10-29_nodust_map_39633433328091196/fit.py
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
NSTART=7
TLIM=12600  # s per start
OUT=$REPO/results/2026-10-29_nodust_map_39633433328091196
cd $REPO
mkdir -p $OUT/miles_dust $OUT/miles_nodust
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/miles_ls6/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CORES=($($PY -c "import os; print(*sorted(os.sched_getaffinity(0)))"))
echo "job $SLURM_JOB_ID on $(hostname), ${#CORES[@]} cores, start $(date)"
free -g
sha256sum $EXP/spectra.py $EXP/miles_s3.slurm $FIT experiments/2026-10-08_continuum_outlier/fit.py \
    experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
$PY -c "import fsps; print(fsps.__file__, fsps.StellarPopulation(zcontinuous=1).libraries)"
taskset -c ${CORES[0]} $PY $FIT check --lib miles --dust on || exit 1
taskset -c ${CORES[0]} $PY $FIT check --lib miles --dust off || exit 1
taskset -c ${CORES[0]} $PY $EXP/spectra.py --lib miles || exit 1
STEP=$((${#CORES[@]} / (2 * NSTART)))
PIDS=()
i=0
for dust in on off; do
    sub=$([ $dust = on ] && echo miles_dust || echo miles_nodust)
    for k in $(seq 0 $((NSTART - 1))); do
        c=${CORES[$((i * STEP))]}
        timeout --kill-after=60 $TLIM taskset -c $c $PY $FIT fit --lib miles --dust $dust \
            --start $k --n-starts $NSTART > $OUT/$sub/fit_s$k.log 2>&1 &
        PIDS+=($!)
        echo "dust $dust start $k on core $c, pid $!"
        i=$((i + 1))
        sleep 10
    done
done
wait "${PIDS[@]}" || true
echo "fits done $(date)"
grep -h "nfev\|peak memory" $OUT/miles_*/fit_s*.log
$PY $FIT collect --lib miles --dust on --n-starts $NSTART
$PY $FIT collect --lib miles --dust off --n-starts $NSTART
echo "end $(date)"
