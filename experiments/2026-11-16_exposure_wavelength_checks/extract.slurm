#!/bin/bash
# Run extract.py on one skx-dev node, from the run copy, reading /scratch and writing to /work results.
# Submit from $RUN: sbatch experiments/2026-11-16_exposure_wavelength_checks/extract.slurm
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -A TG-PHY240056
#SBATCH -J skywave
#SBATCH -o /scratch/11006/nikhilgaruda/logs/skywave_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
RUN=/scratch/11006/nikhilgaruda/claude_runs/2026-11-14_poco_psb_sfh
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
H=$RUN/experiments/2026-11-16_exposure_wavelength_checks
E=$RUN/experiments/2026-10-09_nautilus_c3k_er_broadline
export IRON_EXPOSURES=/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_exposures
export RESULTS_ROOT=$REPO/results PYTHONPATH=$E
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $RUN
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $H/extract.py $E/ivar_recompute.py $E/targets.py
$PY $H/extract.py
echo "DONE $(date)"
