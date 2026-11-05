#!/bin/bash
# Run the pending skx recoadds on skx-dev when it starts them first (Nikhil, 2026-11-03). For each
# TARGETID:skx-jobid[:skx-dev-jobid] argument, submit a skx-dev copy of recoadd.slurm while skx-dev has
# a free submit slot (4 per user). When the copy is RUNNING and the skx job still PENDING, cancel the
# skx job; when the skx job is RUNNING first, cancel the pending copy, so one job writes each output
# folder. Ends when every target has one job running or done. Run on the stampede3 login node from
# the repository root:
#   nohup bash experiments/2026-11-01_poco_repeats/recoadd_devswap.sh TID:JOB[:DEVJOB] ... > <log> 2>&1 &
# TACC's sbatch prints its banner even with --parsable, so the job id is the last numeric line.
E=experiments/2026-11-01_poco_repeats
SRC=/scratch/11006/nikhilgaruda/ladder_src
declare -A SKX DEV
for p in "$@"; do
    IFS=: read -r t j d <<< "$p"
    SKX[$t]=$j
    [ -n "$d" ] && DEV[$t]=$d
done
state() { squeue -h -j "$1" -o %T 2>/dev/null; }
while true; do
    left=0
    for tid in "${!SKX[@]}"; do
        s=$(state "${SKX[$tid]}")
        d=${DEV[$tid]:-}
        ds=$([ -n "$d" ] && state "$d")
        if [ "$s" = PENDING ]; then
            if [ -z "$d" ] && [ "$(squeue -h -u $USER -p skx-dev | wc -l)" -lt 4 ]; then
                d=$(TID=$tid SRC=$SRC sbatch --parsable --export=ALL -p skx-dev -J recdev_$tid $E/recoadd.slurm 2>/dev/null | grep -E '^[0-9]+$' | tail -1)
                DEV[$tid]=$d
                echo "$(date +%T) $tid skx-dev copy $d submitted"
            elif [ "$ds" = RUNNING ]; then
                scancel "${SKX[$tid]}"
                echo "$(date +%T) $tid skx-dev $d running, cancelled skx ${SKX[$tid]}"
                unset "SKX[$tid]"
                continue
            elif [ -n "$d" ] && [ -z "$ds" ]; then
                echo "$(date +%T) $tid skx-dev copy $d ended unseen; skx ${SKX[$tid]} kept, check the copy's log"
                unset "SKX[$tid]"
                continue
            fi
            left=1
        elif [ -n "$d" ] && [ "$ds" = PENDING ]; then
            scancel "$d"
            echo "$(date +%T) $tid skx ${SKX[$tid]} $s, cancelled pending skx-dev copy $d"
            unset "SKX[$tid]"
        else
            echo "$(date +%T) $tid skx ${SKX[$tid]} ${s:-done}, skx-dev ${d:-none} ${ds:-}"
            unset "SKX[$tid]"
        fi
    done
    [ "$left" = 0 ] && [ ${#SKX[@]} = 0 ] && break
    sleep 60
done
echo "done $(date)"
