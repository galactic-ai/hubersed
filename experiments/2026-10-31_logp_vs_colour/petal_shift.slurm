#!/bin/bash
# Per-exposure colour shift by tile, petal and exposure over the unzipped spectra (petal_shift.py).
# Run after exp_extract.slurm. Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-31_logp_vs_colour/petal_shift.slurm
#SBATCH -J petal1031
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/petal1031_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd $REPO
HERE=experiments/2026-10-31_logp_vs_colour
DESIPY=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv_shared/bin/python
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $HERE/petal_shift.py $HERE/petal_shift.slurm
$DESIPY $HERE/petal_shift.py /scratch/11006/nikhilgaruda/hubersed_data/desi_iron_spectra/unzipped \
    results/2026-10-31_logp_vs_colour
echo "end $(date)"
