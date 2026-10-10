#!/bin/bash
# Test fit_fork.py on ls6 before the er1009_shared chain uses it. Copies the chain's live checkpoint
# into a scratch folder, resumes the copy there with a forked pool of POOL workers for TIMEOUT s,
# and logs every 2 min the memory of each process (Rss, Pss and private from smaps_rollup), the
# node's used memory and the latest sampler row. It writes no CONVERGED marker and never writes
# the live checkpoint. Submit with, for example,
#   sbatch -J forktest --export=ALL,POOL=48 experiments/2026-10-09_nautilus_c3k_er_broadline/test_fork.slurm
# Output goes to /work/11006/nikhilgaruda/ls6/research/forktest_<job id>/ (fit.log and mem.log).
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/%x_%j.out

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-09_nautilus_c3k_er_broadline
TID=39627770174637084
STEM=${TID}_er_broad_shared_split45_rest9000_seed0
LIVE=$REPO/results/2026-10-09_nautilus_c3k_er_broadline
POOL=${POOL:-48}
NBATCH=${NBATCH:-480}
TIMEOUT=${TIMEOUT:-2400}  # sampler timeout in s; setup and warm-up need the rest of the hour
# /work, as the chain uses: on 2026-10-10 a compute node wrote to /scratch at about 0.1 MB/s
OUT=/work/11006/nikhilgaruda/ls6/research/forktest_$SLURM_JOB_ID
mkdir -p "$OUT"
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# the chain writes the live checkpoint after every batch, so copy until the copy opens cleanly
for i in 1 2 3 4 5; do
    cp "$LIVE/$STEM.h5" "$LIVE/$STEM.json" "$OUT/"
    if uv run --no-sync python -c "import h5py, sys; h5py.File(sys.argv[1], 'r')['sampler'].attrs['n_like']" \
        "$OUT/$STEM.h5"; then
        break
    fi
    sleep 5
done
echo "test $SLURM_JOB_ID on $(hostname), pool $POOL, n_batch $NBATCH, timeout $TIMEOUT, start $(date)"

uv run --no-sync python $EXP/fit_fork.py --tid $TID --forbidden-broad shared --pool $POOL \
    --n-batch $NBATCH --timeout $TIMEOUT --out "$OUT" > "$OUT/fit.log" 2>&1 &
RUN=$!
while kill -0 $RUN 2>/dev/null; do
    {
        echo "== $(date +%T)"
        for p in $(pgrep -u "$USER" -f "$EXP/fit_fork.py"); do
            awk -v p="$p" '/^(Rss|Pss|Private_Clean|Private_Dirty):/ {s[$1] += $2}
                END {printf "%s rss %.2f pss %.2f private %.2f GiB\n", p, s["Rss:"] / 1048576,
                     s["Pss:"] / 1048576, (s["Private_Clean:"] + s["Private_Dirty:"]) / 1048576}' \
                /proc/"$p"/smaps_rollup 2>/dev/null || true
        done
        free -g | sed -n 2p
        tr '\r' '\n' < "$OUT/fit.log" | grep -E '^(Sampling|Stopped)' | tail -1 || true
    } >> "$OUT/mem.log"
    sleep 120
done
wait $RUN || echo "fit_fork.py exited with $?"
echo "end $(date)"
