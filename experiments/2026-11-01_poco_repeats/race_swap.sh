#!/bin/bash
# Keep whichever of two queued copies of a fit starts first (Nikhil, 2026-11-04). Each argument is
# OLD:NEW:OLDPRED:NEWPRED, the job ids of the two fits and of the predict queued after each. When one
# fit is RUNNING and the other still PENDING, cancel the pending fit and its predict. If both are
# RUNNING, cancel NEW and its predict (it is still building, before any pocoMC state). Ends when every
# pair is settled. Run on the stampede3 login node from the repository root:
#   nohup bash experiments/2026-11-01_poco_repeats/race_swap.sh OLD:NEW:OLDPRED:NEWPRED ... > <log> 2>&1 &
state() { squeue -h -j "$1" -o %T 2>/dev/null; }
PAIRS=("$@")
while [ ${#PAIRS[@]} -gt 0 ]; do
    LEFT=()
    for p in "${PAIRS[@]}"; do
        IFS=: read -r o n op np <<< "$p"
        so=$(state "$o")
        sn=$(state "$n")
        if [ "$sn" = RUNNING ] && [ "$so" = RUNNING ]; then
            scancel "$n" "$np"
            echo "$(date +%T) both running; kept old $o, cancelled new $n and predict $np"
        elif [ "$so" = RUNNING ] || { [ -z "$so" ] && [ "$sn" = PENDING ]; }; then
            scancel "$n" "$np"
            echo "$(date +%T) old $o ${so:-gone}; cancelled new $n and predict $np"
        elif [ "$sn" = RUNNING ] || { [ -z "$sn" ] && [ "$so" = PENDING ]; }; then
            scancel "$o" "$op"
            echo "$(date +%T) new $n ${sn:-gone}; cancelled old $o and predict $op"
        elif [ -z "$so" ] && [ -z "$sn" ]; then
            echo "$(date +%T) $o and $n both gone, check them"
        else
            LEFT+=("$p")
        fi
    done
    PAIRS=("${LEFT[@]}")
    [ ${#PAIRS[@]} -gt 0 ] && sleep 60
done
echo "done $(date)"
