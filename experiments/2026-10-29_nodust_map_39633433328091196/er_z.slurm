#!/bin/bash
# Follow-up 2: C3K ER MAP of 39633433328091196 with the logzsol prior capped at +0.2 (fit.py
# --lib er --zmax 0.2), dust on or off by DUST (default off), on one stampede3 icx node: check,
# 7 starts on one core each, collect. Env and start layout as
# 2026-10-29_continuum_39633433328091196_map/map.slurm. A start killed by the timeout saves nothing.
# Submit from the repository root on stampede3:
#   sbatch --export=ALL,DUST=on experiments/2026-10-29_nodust_map_39633433328091196/er_z.slurm
#   sbatch --export=ALL,DUST=off experiments/2026-10-29_nodust_map_39633433328091196/er_z.slurm
#SBATCH -J er_z
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/er_z_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-29_nodust_map_39633433328091196
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
NSTART=${NSTART:-7}  # starts; 4 fit in an spr node's memory (~22 GiB each)
DUST=${DUST:-off}
TLIM=13200  # s per start, leaves 20 min for check and collect
OUT=$REPO/results/2026-10-29_nodust_map_39633433328091196/er_${DUST/on/dust}_zmax0.2
OUT=${OUT/er_off_zmax0.2/er_nodust_zmax0.2}
cd $REPO
mkdir -p $OUT
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CORES=($($PY -c "import os; print(*sorted(os.sched_getaffinity(0)))"))
echo "job $SLURM_JOB_ID on $(hostname), dust $DUST, ${#CORES[@]} cores, start $(date)"
free -g
sha256sum $EXP/fit.py $EXP/er_z.slurm experiments/2026-10-29_continuum_39633433328091196_map/fit.py \
    experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
taskset -c ${CORES[0]} $PY $EXP/fit.py check --lib er --dust $DUST --zmax 0.2 || exit 1
STEP=$((${#CORES[@]} / NSTART))
PIDS=()
for k in $(seq 0 $((NSTART - 1))); do
    c=${CORES[$((k * STEP))]}
    timeout --kill-after=60 $TLIM taskset -c $c $PY $EXP/fit.py fit --lib er --dust $DUST --zmax 0.2 --start $k \
        --n-starts $NSTART > $OUT/fit_s$k.log 2>&1 &
    PIDS+=($!)
    echo "start $k on core $c, pid $!"
    sleep 20
done
wait "${PIDS[@]}" || true
echo "fits done $(date)"
grep -h "nfev\|peak memory" $OUT/fit_s*.log
$PY $EXP/fit.py collect --lib er --dust $DUST --zmax 0.2 --n-starts $NSTART
echo "end $(date)"
