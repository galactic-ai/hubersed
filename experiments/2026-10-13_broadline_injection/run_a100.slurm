#!/bin/bash
#SBATCH -J inj1013
#SBATCH -A AST25023
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 03:00:00
#SBATCH -o inj1013_%j.out
#
# Run the injection-mock fits (fit.py) on one whole node of our own, adding workers in chunks.
#
# The workers do not share one SSP build: each fit.py process builds its own C3K_ER SSP set at
# startup, about 3.7 GB per process on the Mac. Starting them all at once ran a node out of memory
# on 2026-10-13. So workers start CHUNK at a time. After each chunk has built (WAIT s), the peak
# memory of the largest worker (VmHWM) is read, and the next chunk starts only if CHUNK times that
# peak plus MARGIN_GIB still fits in MemAvailable, up to MAXW workers. A guard kills the newest
# chunk and stops adding if MemAvailable drops below FLOOR_GIB. Killed jobs keep their claim files
# and are not redone; list them with ``fit.py status`` after the run.
#
# Environment as the 2026-10-13_nautilus_resolution_matrix chain.slurm: c3k_er_afe_v3 build,
# sps_home_er, one thread per process, the venv's python.
#
# Submit from the ls6 repo root:
#   sbatch experiments/2026-10-13_broadline_injection/run_a100.slurm
set -u
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
MAXW=${MAXW:-120}
CHUNK=${CHUNK:-8}
WAIT=${WAIT:-300}
MARGIN_GIB=${MARGIN_GIB:-40}
FLOOR_GIB=${FLOOR_GIB:-15}
LOGS=$REPO/results/2026-10-13_broadline_injection/logs_a100_${SLURM_JOB_ID}
mkdir -p "$LOGS"
cd "$REPO" || exit 1
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site:$REPO/experiments/2026-10-13_nautilus_resolution_matrix
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

avail_gib() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
peak_gib() {
    # largest VmHWM over the running workers, in GiB, rounded up
    local m=0 k
    for p in "${PIDS[@]}"; do
        k=$(awk '/VmHWM/ {print $2}' /proc/$p/status 2>/dev/null) || continue
        [ -n "$k" ] && [ "$k" -gt "$m" ] && m=$k
    done
    echo $(( (m + 1048575) / 1048576 ))
}
alive() {
    local n=0
    for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done
    echo $n
}

echo "job $SLURM_JOB_ID on $(hostname), $(nproc) cores, MemTotal $(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo) GiB," \
    "MemAvailable $(avail_gib) GiB, MAXW $MAXW CHUNK $CHUNK WAIT $WAIT MARGIN $MARGIN_GIB FLOOR $FLOOR_GIB, start $(date)"

PIDS=()
STOP=$LOGS/stop_adding
start_chunk() {
    local first=${#PIDS[@]} i
    LAST_CHUNK=()
    for i in $(seq "$first" $((first + CHUNK - 1))); do
        "$REPO/.venv/bin/python" "$REPO/experiments/2026-10-13_broadline_injection/fit.py" run \
            --seeds -1 0 1 2 3 4 > "$LOGS/worker$i.log" 2>&1 &
        PIDS+=($!)
        LAST_CHUNK+=($!)
    done
    printf '%s\n' "${LAST_CHUNK[@]}" > "$LOGS/last_chunk_pids"
    echo "$(date) started workers $first-$((first + CHUNK - 1)), MemAvailable $(avail_gib) GiB"
}

# guard: kill the newest chunk if memory runs low, and stop adding
(
    while true; do
        a=$(avail_gib)
        if [ "$a" -lt "$FLOOR_GIB" ]; then
            echo "$(date) GUARD MemAvailable $a GiB < $FLOOR_GIB, killing newest chunk"
            touch "$STOP"
            xargs -r kill < "$LOGS/last_chunk_pids"
            sleep 60
        fi
        sleep 5
    done
) &
GUARD=$!

start_chunk
while true; do
    sleep "$WAIT"
    n=$(alive)
    a=$(avail_gib)
    pk=$(peak_gib)
    echo "$(date) alive $n of ${#PIDS[@]}, peak per worker ${pk} GiB, MemAvailable $a GiB"
    if [ "$n" -eq 0 ]; then
        echo "no workers alive, stop adding"
        break
    fi
    if [ "$n" -lt "${#PIDS[@]}" ]; then
        echo "some workers exited (no jobs left or an error), stop adding"
        break
    fi
    if [ -e "$STOP" ] || [ $(( ${#PIDS[@]} + CHUNK )) -gt "$MAXW" ] \
        || [ $(( a - CHUNK * pk )) -lt "$MARGIN_GIB" ]; then
        echo "stop adding at ${#PIDS[@]} workers"
        break
    fi
    start_chunk
done

for p in "${PIDS[@]}"; do wait "$p"; done
kill "$GUARD" 2>/dev/null
echo "all workers done, $(ls "$REPO/results/2026-10-13_broadline_injection/fits" | wc -l) fit files, end $(date)"
