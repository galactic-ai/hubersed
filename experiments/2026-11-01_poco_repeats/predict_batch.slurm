#!/bin/bash
# Several ladder.slurm predicts side by side on one node, each on its own core with its step's FSPS
# build. Arguments are TARGETID:STEP pairs. Submit from the repository root on stampede3:
#   sbatch experiments/2026-11-01_poco_repeats/predict_batch.slurm <TID>:<STEP> [...]
#SBATCH -J predbatch
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/predbatch_%j.log

cd /work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-11-01_poco_repeats
LOGS=/scratch/11006/nikhilgaruda/logs
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
i=0
for p in "$@"; do
    IFS=: read -r tid step <<< "$p"
    core=$((2 + 4 * i))
    (log=$LOGS/predbatch_${step}_${tid}_$SLURM_JOB_ID.log
     TID=$tid STEP=$step ACTION=predict CORE=$core bash $EXP/ladder.slurm > "$log" 2>&1
     echo "$(date +%T) $tid $step: $(grep -oE 'stored lnlike [0-9.]+, recomputed [0-9.]+' "$log" || echo 'no lnlike line'), $(grep -oE 'python exit [0-9]+' "$log")") &
    i=$((i + 1))
done
wait
echo "end $(date)"
