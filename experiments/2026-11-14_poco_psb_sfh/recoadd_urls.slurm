#!/bin/bash
# Run recoadd_urls.py on one skx-dev node, since python does not run on login nodes. It reads the SV3
# coadds' EXP_FIBERMAP from /work and writes recoadd_urls.txt and recoadd_exposures.txt here.
# Submit from $RUN: sbatch experiments/2026-11-14_poco_psb_sfh/recoadd_urls.slurm
#SBATCH -J psburls
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/psburls_%j.log

set -euo pipefail
RUN=/scratch/11006/nikhilgaruda/claude_runs/2026-11-14_poco_psb_sfh
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $RUN/experiments/2026-11-14_poco_psb_sfh/recoadd_urls.py
$PY $RUN/experiments/2026-11-14_poco_psb_sfh/recoadd_urls.py
echo "DONE $(date)"
