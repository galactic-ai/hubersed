#!/bin/bash
# Run ladder.slurm predicts inside the allocation of a running job (Nikhil, 2026-11-04), at most
# MAXPAR at a time, each through srun --overlap on its own core. Arguments: the job id, then
# TARGETID:STEP pairs. A predict still running when that job ends is killed with it. Run on the
# stampede3 login node from the repository root:
#   nohup bash experiments/2026-11-01_poco_repeats/predict_in_job.sh JOB TID:STEP ... > <log> 2>&1 &
JOB=$1
shift
PART=$(squeue -h -j "$JOB" -o %P)  # TACC's srun wants -p and -t even inside a job
LEFT=$(squeue -h -j "$JOB" -o %L)  # the job's time left, as the steps' limit
[ -n "$PART" ] || { echo "job $JOB is not running"; exit 1; }
MAXPAR=${MAXPAR:-3}
LOGS=/scratch/11006/nikhilgaruda/logs
EXP=experiments/2026-11-01_poco_repeats
i=0
for p in "$@"; do
    IFS=: read -r tid step <<< "$p"
    while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 10; done
    core=$((5 + 14 * (i % MAXPAR)))
    echo "$(date +%T) start $tid $step on core $core"
    (log=$LOGS/predin_${step}_${tid}_$JOB.log
     TID=$tid STEP=$step ACTION=predict CORE=$core srun --jobid="$JOB" -p "$PART" -t "$LEFT" --overlap \
         -N 1 -n 1 bash $EXP/ladder.slurm > "$log" 2>&1
     echo "$(date +%T) end $tid $step, $(grep -oE 'python exit [0-9]+' "$log" || echo 'no python exit line')") &
    i=$((i + 1))
done
wait
echo "done $(date)"
