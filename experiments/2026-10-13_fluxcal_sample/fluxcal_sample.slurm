#!/bin/bash
# Flux calibration rerun on a sample of exposure-petals (fluxcal_sample.py), development queue.
# Submit from this folder: sbatch fluxcal_sample.slurm (pilot), or for all units
# UNITS=experiments/2026-10-13_fluxcal_sample/units_all.csv NAME=all EXTRA=--discard sbatch fluxcal_sample.slurm.
# A resubmission with the same NAME skips finished units.
#SBATCH -J fluxcal_sample
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH -o fluxcal_sample_%j.out

cd /work/11006/nikhilgaruda/ls6/research/hubersed
export DESIMODEL=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/desimodel
PY=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv/bin/python
$PY experiments/2026-10-13_fluxcal_sample/fluxcal_sample.py \
    "${UNITS:-experiments/2026-10-13_fluxcal_sample/pilot_units.csv}" "${NAME:-pilot}" ${EXTRA}
