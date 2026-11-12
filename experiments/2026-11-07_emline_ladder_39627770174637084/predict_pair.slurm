#!/bin/bash
# Two predicts on one node, one after the other: predict.slurm with STEP=$A, then STEP=$B, each to
# its own log (side by side they ran out of commit room in the SSP build fork, job 3705373).
# Submit from the repository root on stampede3:
#   A=7 B=8 sbatch --export=ALL experiments/2026-11-07_emline_ladder_39627770174637084/predict_pair.slurm
#SBATCH -J ladpredict2
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/ladpredict2_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-11-07_emline_ladder_39627770174637084
LOGS=/scratch/11006/nikhilgaruda/logs
A=${A:-7}
B=${B:-8}
cd $REPO
echo "job $SLURM_JOB_ID steps $A and $B on $SLURM_JOB_NODELIST, start $(date)"
STEP=$A bash $EXP/predict.slurm > $LOGS/ladpredict2_${SLURM_JOB_ID}_step$A.log 2>&1
SA=$?
STEP=$B bash $EXP/predict.slurm > $LOGS/ladpredict2_${SLURM_JOB_ID}_step$B.log 2>&1
SB=$?
echo "step $A exit $SA, step $B exit $SB, end $(date)"
exit $((SA || SB))
