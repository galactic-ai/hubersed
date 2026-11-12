#!/bin/bash
# Extract the SV3 bright rows of the EmFit VAC (extract_emfit.py) on one skx-dev core. Run from the
# folder that holds extract_emfit.py:  sbatch extract_emfit.slurm
#SBATCH -J emfitx
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/emfitx_%j.log

set -euo pipefail
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
# data/ on the clone is a Corral symlink, which compute nodes do not mount; read the /scratch copy
VAC=/scratch/11006/nikhilgaruda/hubersed_data/emfit_v2.3.1/emfit-dr1-v2.3.1.fits
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum extract_emfit.py extract_emfit.slurm
$PY extract_emfit.py $VAC emfit_sv3_bright.fits
echo "end $(date)"
