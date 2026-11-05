#!/bin/bash
# Flux calibration rerun of all units on two nodes at once (development queue allows one running
# job per user but up to 8 nodes). Each node takes half of units_all.csv (units_all_part0.csv and
# units_all_part1.csv, alternate rows). Same NAME as earlier runs, so finished units are skipped.
# Submit from this folder: sbatch --dependency=afterany:<jobid> fluxcal_sample_2node.slurm
#SBATCH -J fluxcal_2node
#SBATCH -p development
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH -o fluxcal_sample_2node_%j.out

cd /work/11006/nikhilgaruda/ls6/research/hubersed
export DESIMODEL=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/desimodel
PY=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv/bin/python
D=experiments/2026-10-13_fluxcal_sample
for k in 0 1; do
    srun -N 1 -n 1 --exclusive $PY $D/fluxcal_sample.py $D/units_all_part$k.csv all \
        --discard --s3-streams 50 --units-parallel 16 --tag _part$k &
done
wait
